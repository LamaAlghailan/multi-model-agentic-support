"""Typed contracts with SQLite persistence and explicitly labelled fixture backends."""
import ast
import json
import math
import operator
import re
import sqlite3
from pathlib import Path
from typing import Literal
from pydantic import Field, ValidationError
from src.config import ROOT
from src.schemas import Contract, ToolResult
from src.observability import redact

class Search(Contract):
    query: str = Field(min_length=1, max_length=2000)
    limit: int = Field(default=3, ge=1, le=10)

class Ticket(Contract):
    title: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1, max_length=4000)
    priority: Literal['low', 'normal', 'high'] = 'normal'

class Service(Contract):
    service: Literal['api', 'database', 'gpu', 'network', 'deployment']

class Logs(Contract):
    text: str = Field(min_length=1, max_length=16000)

class Package(Contract):
    name: str = Field(pattern=r'^[a-zA-Z0-9_.-]{1,80}$')

class SQL(Contract):
    query: str = Field(min_length=1, max_length=4000)

class Calculate(Contract):
    expression: str = Field(min_length=1, max_length=200)

class Escalation(Contract):
    reason: str = Field(min_length=1, max_length=1000)
    evidence: str = Field(min_length=1, max_length=8000)

class Runbook(Contract):
    intent: Literal['authentication', 'network', 'deployment', 'database', 'gpu', 'api', 'package', 'general']

CONTRACTS = {
    'knowledge_base_search': Search, 'ticket_search': Search, 'ticket_create': Ticket,
    'system_health_check': Service, 'log_analyzer': Logs, 'documentation_search': Search,
    'package_lookup': Package, 'sql_query': SQL, 'calculator': Calculate, 'file_search': Search,
    'web_search': Search, 'escalate_to_human': Escalation, 'diagnostic_runbook': Runbook,
}

def calculate(expression):
    operations = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
                  ast.Div: operator.truediv, ast.USub: operator.neg, ast.UAdd: operator.pos}
    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            value = node.value
        elif isinstance(node, ast.BinOp) and type(node.op) in operations:
            value = operations[type(node.op)](visit(node.left), visit(node.right))
        elif isinstance(node, ast.UnaryOp) and type(node.op) in operations:
            value = operations[type(node.op)](visit(node.operand))
        else:
            raise ValueError('Only finite numeric arithmetic + - * / is supported')
        if not math.isfinite(value) or abs(value) > 1e15:
            raise ValueError('Calculation exceeds numeric bounds')
        return value
    tree = ast.parse(expression, mode='eval')
    if sum(1 for _ in ast.walk(tree)) > 80:
        raise ValueError('Expression too complex')
    return visit(tree.body)

class ToolRegistry:
    def __init__(self, database, observer=None):
        self.database = Path(database)
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self.observer = observer
        self.docs = json.loads((ROOT / 'data/kb.json').read_text(encoding='utf-8'))
        with self.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS tickets (id INTEGER PRIMARY KEY, title TEXT NOT NULL, description TEXT NOT NULL, priority TEXT NOT NULL, status TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS escalations (id INTEGER PRIMARY KEY, reason TEXT NOT NULL, evidence TEXT NOT NULL)')

    def connect(self):
        db = sqlite3.connect(self.database, timeout=5)
        db.row_factory = sqlite3.Row
        return db

    def invoke(self, name, payload):
        if self.observer:
            with self.observer.span('tool.' + name, payload) as record:
                result = self._invoke(name, payload)
                record.update(result)
                return result
        return self._invoke(name, payload)

    def _invoke(self, name, payload):
        backend = 'sqlite' if name in {'ticket_search', 'ticket_create', 'sql_query', 'escalate_to_human'} else 'local'
        if name == 'system_health_check':
            backend = 'deterministic_mock'
        if name == 'web_search':
            backend = 'unconfigured'
        try:
            if name not in CONTRACTS:
                raise ValueError('Unknown tool')
            args = CONTRACTS[name].model_validate(payload)
            data = getattr(self, name)(**args.model_dump())
            result = ToolResult(tool=name, ok=True, backend=backend, data=redact(data))
        except (ValueError, ValidationError, sqlite3.Error, ZeroDivisionError, SyntaxError, OverflowError) as exc:
            result = ToolResult(tool=name, ok=False, backend=backend, error=redact(str(exc))[:500])
        return result.model_dump()

    def knowledge_base_search(self, query, limit=3):
        tokens = set(re.findall(r'\w+', query.lower())) - {'the', 'what', 'which', 'is', 'a', 'in', 'to', 'of', 'i'}
        ranked = sorted(((len(tokens & set(re.findall(r'\w+', text.lower()))), key, text)
                         for key, text in self.docs.items()), reverse=True)
        return {'matches': [{'id': key, 'text': text, 'score': score, 'source': 'course_fixture'}
                            for score, key, text in ranked[:limit] if score > 0]}

    documentation_search = knowledge_base_search

    def ticket_search(self, query, limit=3):
        with self.connect() as db:
            rows = db.execute('SELECT * FROM tickets WHERE title LIKE ? OR description LIKE ? ORDER BY id LIMIT ?',
                              ('%' + query + '%', '%' + query + '%', limit)).fetchall()
        return {'tickets': [dict(r) for r in rows]}

    def ticket_create(self, title, description, priority='normal'):
        with self.connect() as db:
            cur = db.execute('INSERT INTO tickets(title,description,priority,status) VALUES(?,?,?,?)',
                             (redact(title), redact(description), priority, 'open'))
            return {'id': cur.lastrowid, 'status': 'open', 'priority': priority}

    def system_health_check(self, service):
        fixtures = {'api': {'status': 'healthy'}, 'database': {'status': 'degraded', 'connections_pct': 91},
                    'gpu': {'status': 'unknown'}, 'network': {'status': 'healthy'}, 'deployment': {'status': 'healthy'}}
        return {'service': service, **fixtures[service], 'is_live': False, 'notice': 'Fixture only; not production telemetry'}

    def log_analyzer(self, text):
        return {'findings': [label for label, pattern in {
            'database_connections': r'too many connections|pool.*(exhaust|98|100)',
            'out_of_memory': r'out of memory|oom', 'validation': r'\b422\b',
            'unavailable': r'\b503\b', 'authentication': r'\b401\b|unauthorized',
        }.items() if re.search(pattern, text, re.I)], 'inference': 'Pattern matches only; not a root-cause conclusion'}

    def package_lookup(self, name):
        from importlib.metadata import version, PackageNotFoundError
        try:
            return {'package': name, 'installed_version': version(name), 'scope': 'agent environment'}
        except PackageNotFoundError:
            raise ValueError('Package not installed in the agent environment') from None

    def sql_query(self, query):
        if not re.match(r'^\s*(SELECT|WITH)\b', query, re.I):
            raise ValueError('Only read-only SELECT queries are allowed')
        # Enforce read-only access in SQLite, not solely with keyword checks.
        uri = self.database.resolve().as_uri() + '?mode=ro'
        with sqlite3.connect(uri, uri=True) as db:
            db.row_factory = sqlite3.Row
            db.execute('PRAGMA query_only=ON')
            allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}
            def authorize(action, arg1, arg2, database, source):
                if action not in allowed or (action == sqlite3.SQLITE_FUNCTION and str(arg2).lower() == 'load_extension'):
                    return sqlite3.SQLITE_DENY
                if action == sqlite3.SQLITE_READ and arg1 != 'tickets':
                    return sqlite3.SQLITE_DENY
                return sqlite3.SQLITE_OK
            db.set_authorizer(authorize)
            db.set_progress_handler(lambda: 1, 100000)
            rows = db.execute(query).fetchmany(101)
        return {'rows': [dict(r) for r in rows[:100]], 'truncated': len(rows) > 100}

    def calculator(self, expression):
        return {'value': calculate(expression)}

    def file_search(self, query, limit=3):
        # Only indexed KB filenames; never inspect arbitrary host files.
        return {'files': [{'path': 'data/kb.json', 'document_id': key} for key in sorted(self.docs) if query.lower() in key.lower()][:limit],
                'scope': 'bundled knowledge base index'}

    def web_search(self, query, limit=3):
        raise ValueError('Web search provider is not configured; no live web results available')

    def escalate_to_human(self, reason, evidence):
        with self.connect() as db:
            cur = db.execute('INSERT INTO escalations(reason,evidence) VALUES(?,?)', (redact(reason), redact(evidence)))
        return {'id': cur.lastrowid, 'status': 'recorded_locally', 'reason': redact(reason),
                'evidence': redact(evidence), 'human_notified': False}

    def diagnostic_runbook(self, intent):
        steps = {
            'database': ['Inspect connection usage and lock waits.', 'Review slow transactions before changing limits.'],
            'gpu': ['Check CUDA availability.', 'Check model and tensor device placement.'],
            'deployment': ['Inspect startup logs and exit code.', 'Verify environment and port mapping.'],
            'authentication': ['Check expiry and intended token audience without sharing credentials.', 'Verify MFA and account status.'],
            'network': ['Check DNS and reachability.', 'Review firewall and proxy configuration.'],
            'api': ['Inspect request validation and server logs.', 'Check dependency health.'],
            'package': ['Compare installed versions with requirements.', 'Reproduce in an isolated environment.'],
            'general': ['Gather reproducible symptoms.', 'Collect relevant logs with secrets removed.'],
        }
        return {'steps': steps[intent], 'executed': False, 'notice': 'Read-only diagnostic guidance'}
