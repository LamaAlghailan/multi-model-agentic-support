import secrets
import threading
import time
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, Header, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from src.config import Settings, MODEL_NAME
from src.schemas import ChatRequest
from src.runtime import create_runtime

def create_app(settings=None, runtime_factory=create_runtime):
    settings = settings or Settings()
    lock = threading.BoundedSemaphore(settings.max_concurrency)

    @asynccontextmanager
    async def lifespan(app):
        app.state.ready = False
        app.state.graph = app.state.observer = None
        try:
            app.state.graph, app.state.observer = runtime_factory(settings)
            app.state.ready = True
        except Exception:
            # Never expose exception text which could contain paths, credentials or user data.
            pass
        yield
        if app.state.observer:
            app.state.observer.close()

    app = FastAPI(title='Tuwaiq Technical Support Agent', lifespan=lifespan)

    def authenticate(authorization):
        if settings.api_key and not secrets.compare_digest(authorization or '', 'Bearer ' + settings.api_key):
            raise HTTPException(401, 'Invalid API key')

    @app.exception_handler(HTTPException)
    async def http_error(request, exc):
        return JSONResponse(status_code=exc.status_code, content={'error': {'message': exc.detail, 'type': 'request_error', 'code': str(exc.status_code)}})

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        return JSONResponse(status_code=422, content={'error': {'message': 'Invalid request schema', 'type': 'invalid_request_error', 'code': '422'}})

    @app.get('/health')
    def health():
        ready = getattr(app.state, 'ready', False)
        return JSONResponse(status_code=200 if ready else 503, content={'status': 'ok' if ready else 'not_ready',
                             'mode': settings.mode, 'models_verified': ready and settings.mode == 'production'})

    @app.get('/v1/models')
    def models(authorization: str | None = Header(default=None)):
        authenticate(authorization)
        return {'object': 'list', 'data': [{'id': MODEL_NAME, 'object': 'model', 'created': 0, 'owned_by': 'tuwaiq'}]}

    @app.post('/v1/chat/completions')
    def chat(req: ChatRequest, authorization: str | None = Header(default=None)):
        authenticate(authorization)
        if req.model != MODEL_NAME:
            raise HTTPException(404, 'Unknown model')
        if req.stream:
            raise HTTPException(400, 'Streaming is not supported')
        if req.stop:
            raise HTTPException(400, 'Custom stop sequences are not supported')
        if req.messages[-1].role != 'user':
            raise HTTPException(400, 'The last message must be a user message')
        if sum(len(m.content) for m in req.messages) > 24000:
            raise HTTPException(413, 'Conversation exceeds the supported size')
        if not getattr(app.state, 'ready', False):
            raise HTTPException(503, 'Agent not ready: verify model quality evidence and configuration')
        if not lock.acquire(blocking=False):
            raise HTTPException(429, 'Agent busy; retry later')
        trace_id = uuid.uuid4().hex
        started = time.perf_counter()
        try:
            with app.state.observer.span('support_request', {'trace_id': trace_id, 'messages': [m.model_dump() for m in req.messages]}) as event:
                result = app.state.graph.invoke({'user_message': req.messages[-1].content,
                         'history': [m.model_dump() for m in req.messages[:-1]], 'trace_id': trace_id,
                         'tool_results': [], 'context': '', 'max_tokens': req.max_completion_tokens or req.max_tokens})
                event.update(result)
        except ValueError:
            raise HTTPException(400, 'Input exceeds the model context budget or violates a contract') from None
        except Exception:
            raise HTTPException(503, 'Agent execution failed; contact support with the request time') from None
        finally:
            lock.release()
        return {'id': 'chatcmpl-' + trace_id, 'object': 'chat.completion', 'created': int(time.time()), 'model': MODEL_NAME,
                'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': result['answer']}, 'finish_reason': 'stop'}],
                'system_metadata': {'trace_id': trace_id, 'route': result['route'], 'intent': result['intent'],
                                    'confidence': result['confidence'], 'mode': settings.mode,
                                    'latency_seconds': time.perf_counter() - started}}
    return app

app = create_app()
