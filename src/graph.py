import json
from typing import TypedDict, Any
from langgraph.graph import StateGraph, START, END
from src.tools.selector import select_tools

class SupportState(TypedDict, total=False):
    user_message: str
    route: str
    intent: str
    confidence: float
    reason: str
    context: str
    tool_results: list[dict]
    answer: str
    escalate: bool
    trace_id: str
    history: list[dict]
    max_tokens: int
    path: list[str]

def build_graph(router, qa, support, tools, observer):
    def route_node(state):
        with observer.span('route', {'message': state['user_message']}) as event:
            decision = router.route(state['user_message']).model_dump()
            event.update(decision)
        return {**decision, 'escalate': decision['route'] == 'escalate', 'path': ['route_node']}

    def qa_node(state):
        result = tools.invoke('knowledge_base_search', {'query': state['user_message'], 'limit': 1})
        matches = result['data'].get('matches', []) if result['ok'] else []
        context = matches[0]['text'] if matches else ''
        if not context:
            return {'answer': 'The supplied knowledge base does not contain enough evidence to answer.',
                    'context': '', 'tool_results': [result], 'path': state['path'] + ['qa_node']}
        with observer.span('model_b', {'question': state['user_message'], 'context': context}) as event:
            prediction = qa.answer(state['user_message'], context)
            event.update(prediction)
        text = prediction['text']
        answer = f"{text}\nSource: {matches[0]['id']} (course knowledge base)." if text else 'No supported answer span found.'
        return {'answer': answer, 'context': context, 'tool_results': [result], 'path': state['path'] + ['qa_node']}

    def tools_node(state):
        results = [tools.invoke(name, args) for name, args in select_tools(state['user_message'], state['intent'])]
        if any(not result['ok'] for result in results):
            return {'route': 'escalate', 'reason': 'A required diagnostic tool failed', 'tool_results': results,
                    'path': state['path'] + ['tools_node']}
        return {'tool_results': results, 'path': state['path'] + ['tools_node']}

    def support_node(state):
        with observer.span('model_c', {'message': state['user_message'], 'tools': state.get('tool_results', [])}) as event:
            answer = support.answer(state['user_message'], state.get('context', ''), state.get('tool_results', []),
                                    state.get('history', []), state.get('max_tokens', 180))
            # Preserve a visible provenance notice even if the small model drops it.
            if any(r['backend'] == 'deterministic_mock' for r in state.get('tool_results', [])):
                answer = 'Health evidence below is a deterministic fixture, not live production telemetry.\n' + answer
            event['answer'] = answer
        return {'answer': answer, 'path': state['path'] + ['support_node']}

    def escalation_node(state):
        result = tools.invoke('escalate_to_human', {'reason': state['reason'],
                            'evidence': json.dumps({'message': state['user_message'], 'trace_id': state.get('trace_id', ''),
                                                    'tools': state.get('tool_results', [])})[:8000]})
        answer = (f"Human review required. Escalation {result['data']['id']} recorded locally; no human has been notified. "
                  'Contact your authorized support owner and preserve relevant evidence. Avoid destructive changes.') if result['ok'] else (
                  'Human review required. The escalation record could not be saved. Contact your authorized support owner directly.')
        return {'answer': answer, 'escalate': True, 'tool_results': state.get('tool_results', []) + [result], 'path': state['path'] + ['escalation_node']}

    def guarded(name, node):
        def invoke(state):
            try:
                return node(state)
            except Exception:
                # Keep exception text out of user responses and persisted evidence.
                failed = {**state, 'route': 'escalate', 'intent': state.get('intent', 'general'),
                          'confidence': 0, 'reason': f'{name} failed',
                          'path': state.get('path', []) + [name]}
                if name == 'escalation_node':
                    return {**failed, 'escalate': True,
                            'answer': 'Human review required. Recording failed; contact your support owner directly.'}
                return failed
        return invoke

    graph = StateGraph(SupportState)
    for name, node in [('route_node', route_node), ('qa_node', qa_node), ('tools_node', tools_node),
                       ('support_node', support_node), ('escalation_node', escalation_node)]:
        graph.add_node(name, guarded(name, node))
    graph.add_edge(START, 'route_node')
    graph.add_conditional_edges('route_node', lambda s: s['route'],
                                {'qa': 'qa_node', 'tools': 'tools_node', 'support': 'support_node', 'escalate': 'escalation_node'})
    graph.add_conditional_edges('tools_node', lambda s: 'escalation_node' if s['route'] == 'escalate' else 'support_node')
    for name in ['qa_node', 'support_node']:
        graph.add_conditional_edges(name, lambda s: 'escalation_node' if s['route'] == 'escalate' else END)
    graph.add_edge('escalation_node', END)
    return graph.compile()
