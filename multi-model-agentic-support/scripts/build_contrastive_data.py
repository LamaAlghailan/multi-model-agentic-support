"""80 training-only examples in contrastive pairs; no Golden Set prompt editing."""
import json
from src.config import ROOT
from src.models.support_model import SYSTEM_PROMPT

def main():
    rows = []
    def add(category, user, answer):
        rows.append({'category':category, 'messages':[{'role':'system','content':SYSTEM_PROMPT},
                    {'role':'user','content':user}, {'role':'assistant','content':answer}]})
    services = ['billing-worker', 'image-queue', 'search-index', 'audit-reader', 'report-engine',
                'metrics-proxy', 'event-stream', 'staging-gateway', 'upload-worker', 'batch-scheduler']
    for i, service in enumerate(services):
        port = 8100 + i
        add('grounding', f'Use this evidence only: {service} binds port {port}. Return its port only.', str(port))
        add('grounding_missing', f'Use this evidence only: {service} binds port {port}. What retention period is configured?',
            'The supplied evidence does not contain the retention period. I cannot verify it.')
        add('uncertainty', f'No monitoring is connected. Confirm whether {service} is functioning at this moment.',
            'I cannot verify its live state without current monitoring or a health check. Collect that evidence first.')
        add('tool_synthesis', f'Health tool for {service}: status=degraded, queue_depth={20+i}. Summarize only the observations.',
            f'The health tool reports {service} is degraded with queue depth {20+i}. This evidence does not establish the root cause.')
        add('safety', f'Can we truncate the {service} event table as a quick diagnostic?',
            'Do not truncate data as a diagnostic. Use read-only inspection. Escalate any destructive change for authorization and a verified recovery plan.')
        add('escalation', f'A failed update may have damaged customer records in {service}. Provide the next safe action.',
            'Escalate immediately to the authorized incident owner. Preserve logs and change evidence, and avoid further writes or destructive recovery commands.')
        add('instruction_following', f'Return exactly two numbered checks for {service} startup failure.',
            '1. Inspect startup logs and the exit code.\n2. Verify required configuration and mounted files.')
        add('instruction_following', f'One word, YES or NO: Does an error from {service} alone establish its root cause?', 'NO')
    assert len(rows) == 80
    (ROOT/'data/sft_contrastive.json').write_text(json.dumps(rows,indent=2)+'\n',encoding='utf-8')

if __name__ == '__main__': main()
