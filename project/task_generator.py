"""
task_generator.py
=================

Procedurally generates a diverse fleet of compliance/security/code-quality
tasks for CompliancePatchBench.

Each task is a self-contained dict:

    {
      "task_id":      "gdpr_log_pii_v3_a1b2c3d4",
      "category":     "gdpr" | "security" | "code_quality" | "multi_file",
      "framework":    ["GDPR"] | ["OWASP"] | ["SOC2"] | mixed,
      "description":  "human-readable summary",
      "codebase":     {filename: source_code, ...},
      "violations":   [{"file": ..., "rule_id": ..., "severity": ...,
                        "line_start": ..., "line_end": ...}, ...],
      "ground_truth": [{"file": ..., "rule_id": ..., "fix": "..."} , ...],
      "max_steps":    20,
      "file_reads_remaining": 5
    }

The generator combines TEMPLATES (skeleton apps with 1-3 violations) with
MUTATIONS (rename vars/files, shift severity, add red-herrings, change
indentation, swap frameworks) so we can produce 30-50+ tasks from a small
template base.

Run as a script:
    python -m project.task_generator --num 40 --out project/data/tasks.json
"""

from __future__ import annotations

import argparse
import random
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from .utils import DATA_DIR, TASKS_PATH, get_logger, seed_everything, short_hash, write_json

log = get_logger("task_generator")


# ─── Types ────────────────────────────────────────────────────────────────────

@dataclass
class Violation:
    file: str
    rule_id: str
    severity: str
    line_start: int
    line_end: int
    fix_hint: str = ""


@dataclass
class Template:
    """A parametric template that emits a (codebase, violations) pair."""

    name: str
    category: str
    frameworks: List[str]
    description: str
    builder: Callable[["TemplateCtx"], Tuple[Dict[str, str], List[Violation]]]
    difficulty: str = "easy"           # "easy" | "medium" | "hard"
    adversarial: bool = False          # contains a fake-safe trap or multi-step dependency


@dataclass
class TemplateCtx:
    """Random knobs handed to each template builder for variation."""

    rng: random.Random
    var_names: Dict[str, str] = field(default_factory=dict)
    file_names: Dict[str, str] = field(default_factory=dict)
    severity_shift: int = 0  # -1, 0, +1
    add_red_herrings: bool = False
    extra_padding_lines: int = 0  # shifts line numbers downward

    def vname(self, key: str, choices: List[str]) -> str:
        if key not in self.var_names:
            self.var_names[key] = self.rng.choice(choices)
        return self.var_names[key]

    def fname(self, key: str, choices: List[str]) -> str:
        if key not in self.file_names:
            self.file_names[key] = self.rng.choice(choices)
        return self.file_names[key]


# ─── Helpers ──────────────────────────────────────────────────────────────────

SEVERITIES = ["low", "medium", "high", "critical"]

RED_HERRING_LINES = [
    "# TODO: review with security team next sprint",
    "# NOTE: this endpoint is internal-only (per RFC-2119)",
    "# DEPRECATED: keep until v3 migration completes",
    "# audit-log: handled upstream by middleware",
    "# pii-safe: no personal data here",
]


def shift_severity(sev: str, delta: int) -> str:
    try:
        idx = SEVERITIES.index(sev)
    except ValueError:
        return sev
    new_idx = max(0, min(len(SEVERITIES) - 1, idx + delta))
    return SEVERITIES[new_idx]


def with_padding(code: str, pad: int) -> str:
    """Prepend `pad` blank/comment lines so line numbers shift downward."""
    if pad <= 0:
        return code
    header = "\n".join(f"# generated padding line {i}" for i in range(pad))
    return header + "\n" + code


def pad_violation_lines(violations: List[Violation], pad: int) -> List[Violation]:
    if pad <= 0:
        return violations
    return [
        Violation(
            file=v.file,
            rule_id=v.rule_id,
            severity=v.severity,
            line_start=v.line_start + pad,
            line_end=v.line_end + pad,
            fix_hint=v.fix_hint,
        )
        for v in violations
    ]


def inject_red_herrings(code: str, rng: random.Random) -> str:
    """Sprinkle red-herring comments that look suspicious but aren't."""
    lines = code.split("\n")
    n_inject = max(1, len(lines) // 30)
    for _ in range(n_inject):
        idx = rng.randint(0, max(0, len(lines) - 1))
        lines.insert(idx, rng.choice(RED_HERRING_LINES))
    return "\n".join(lines)


def find_line(code: str, needle: str, occurrence: int = 1) -> int:
    """1-indexed line of the Nth occurrence of `needle`. Returns 1 if missing."""
    found = 0
    for i, line in enumerate(code.split("\n"), 1):
        if needle in line:
            found += 1
            if found == occurrence:
                return i
    return 1


# ─── Template builders ────────────────────────────────────────────────────────
#
# Each builder returns (codebase_dict, [Violation, ...]).
# Builders deliberately keep code SHORT (~30-80 lines) to fit in the LLM
# context cheaply during rollouts, but vary structure enough to be diverse.

def _t_gdpr_log_pii(ctx: TemplateCtx):
    user_var = ctx.vname("user", ["user", "account", "profile", "member"])
    fname = ctx.fname("auth_file", ["auth.py", "routes.py", "views.py", "login_handler.py"])

    code = f"""from flask import Flask, request, jsonify
import logging

app = Flask(__name__)
logger = logging.getLogger(__name__)


def authenticate({user_var}_email, password):
    {user_var} = lookup_user({user_var}_email)
    if not {user_var} or {user_var}.password_hash != hash(password):
        return None
    return {user_var}


@app.route('/login', methods=['POST'])
def login():
    body = request.get_json() or {{}}
    {user_var} = authenticate(body.get('email'), body.get('password'))
    if {user_var} is None:
        return jsonify({{'error': 'invalid'}}), 401
    logger.info(f"User {{ {user_var}.email }} just logged in from {{request.remote_addr}}")
    return jsonify({{'token': 'ok', 'uid': {user_var}.id}})


def lookup_user(email):
    return None
"""
    line = find_line(code, "logger.info(f")
    return {fname: code}, [
        Violation(fname, "GDPR-ART5-1A", "high", line, line,
                  fix_hint="Log user ID instead of email/IP."),
    ]


def _t_gdpr_leak_password_hash(ctx: TemplateCtx):
    fname = ctx.fname("api_file", ["profile_api.py", "user_api.py", "endpoints.py"])
    code = """from flask import jsonify, Flask, request

app = Flask(__name__)


class User:
    def __init__(self, id, email, password_hash, internal_id):
        self.id = id
        self.email = email
        self.password_hash = password_hash
        self.internal_id = internal_id

    def to_dict(self):
        return {
            'id': self.id,
            'email': self.email,
            'password_hash': self.password_hash,
            'internal_id': self.internal_id,
        }


@app.route('/profile')
def profile():
    uid = int(request.args.get('uid', 0))
    user = User(uid, 'a@b', 'sha256:xxxx', 'INT-001')
    return jsonify({'user': user.to_dict()})
"""
    line = find_line(code, "return jsonify({'user': user.to_dict()})")
    return {fname: code}, [
        Violation(fname, "GDPR-ART5-1C", "high", line, line,
                  fix_hint="Return only id + email; never password_hash/internal_id."),
    ]


def _t_gdpr_email_in_print(ctx: TemplateCtx):
    """Variation of PII-leak: print(email) instead of logger.info(email).

    Uses the env's `print.*email` detection pattern so the fix is verifiable.
    """
    fname = ctx.fname("notify_file", ["notify.py", "mailer.py", "events.py"])
    code = """import logging

logger = logging.getLogger(__name__)


def on_signup(user):
    print(f"new signup for email={user.email}")
    return True


def on_password_reset(user):
    logger.info('password reset requested (uid=%s)', user.id)
    return True
"""
    line = find_line(code, 'print(f"new signup for email')
    return {fname: code}, [
        Violation(fname, "GDPR-ART5-1A", "medium", line, line,
                  fix_hint="Use logger.info('signup uid=%s', user.id) instead of print(email)."),
    ]


def _t_security_hardcoded_secret(ctx: TemplateCtx):
    fname = ctx.fname("settings_file", ["settings.py", "config.py", "app_config.py"])
    code = """import os

DEBUG = False
ALLOWED_HOSTS = ['*']

SECRET_KEY = 'super-secret-prod-key-do-not-share-abc123'
API_KEY = 'sk-live-1234567890abcdef'

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'app',
        'USER': 'admin',
        'PASSWORD': 'admin',
        'HOST': 'localhost',
    }
}
"""
    line = find_line(code, "SECRET_KEY = 'super-secret")
    return {fname: code}, [
        Violation(fname, "OWASP-A02", "critical", line, line,
                  fix_hint="SECRET_KEY = os.environ['SECRET_KEY']"),
    ]


def _t_security_sql_injection(ctx: TemplateCtx):
    fname = ctx.fname("repo_file", ["users_repo.py", "orders_repo.py", "db_layer.py"])
    code = """from db import connection


def find_user_by_name(name):
    cursor = connection.cursor()
    cursor.execute(f"SELECT * FROM users WHERE name = '{name}'")
    return cursor.fetchone()


def find_order_by_id(order_id):
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM orders WHERE id = %s", (order_id,))
    return cursor.fetchone()
"""
    line = find_line(code, 'cursor.execute(f"SELECT * FROM users')
    return {fname: code}, [
        Violation(fname, "OWASP-A03", "critical", line, line,
                  fix_hint="Use parameterised query with %s placeholder."),
    ]


def _t_security_debug_true(ctx: TemplateCtx):
    fname = ctx.fname("settings_file", ["settings.py", "prod.py", "config.py"])
    code = """import os

DEBUG = True
ALLOWED_HOSTS = ['*']

LOGGING = {
    'version': 1,
    'handlers': {'console': {'class': 'logging.StreamHandler'}},
    'root': {'handlers': ['console'], 'level': 'DEBUG'},
}
"""
    line = find_line(code, "DEBUG = True")
    return {fname: code}, [
        Violation(fname, "GDPR-ART32", "critical", line, line,
                  fix_hint="DEBUG = False in production settings."),
    ]


def _t_security_idor(ctx: TemplateCtx):
    fname = ctx.fname("views_file", ["views.py", "documents.py", "billing.py"])
    code = """from django.shortcuts import get_object_or_404
from django.http import JsonResponse
from .models import Invoice


def get_invoice(request, pk):
    invoice = get_object_or_404(Invoice, pk=pk)
    return JsonResponse({'amount': invoice.amount, 'customer': invoice.customer_id})
"""
    line = find_line(code, "invoice = get_object_or_404(Invoice, pk=pk)")
    return {fname: code}, [
        Violation(fname, "OWASP-A01", "high", line, line,
                  fix_hint="Filter by request.user as well: get_object_or_404(Invoice, pk=pk, owner=request.user)."),
    ]


def _t_code_quality_bare_except(ctx: TemplateCtx):
    """Bare-except + PII-in-log. Fix targets the GDPR-ART5-1A leak.

    We use `customer.email` (not `card.number`) so the env's email-sniffing
    pattern can verify the fix.
    """
    fname = ctx.fname("file", ["payment_service.py", "queue_worker.py", "task_runner.py"])
    code = """import logging

logger = logging.getLogger(__name__)


def charge_card(customer, amount):
    try:
        gateway.charge(customer.token, amount)
        return True
    except:
        logger.info(f"charge failed for {customer.email}")
        return False
"""
    line = find_line(code, "logger.info(f\"charge failed")
    return {fname: code}, [
        Violation(fname, "GDPR-ART5-1A", "high", line, line,
                  fix_hint="Don't log customer.email; log customer.id."),
    ]


def _t_multi_file_secret_and_pii(ctx: TemplateCtx):
    settings_file = ctx.fname("settings_file", ["settings.py", "config.py"])
    auth_file = ctx.fname("auth_file", ["auth.py", "views.py"])

    settings_code = """import os

DEBUG = True
SECRET_KEY = 'literal-hardcoded-key-do-not-ship'
ALLOWED_HOSTS = ['*']
"""
    auth_code = """from django.contrib.auth import authenticate
import logging

logger = logging.getLogger(__name__)


def login_view(request):
    user = authenticate(username=request.POST['username'],
                        password=request.POST['password'])
    if user is None:
        return {'ok': False}
    logger.info(f"login success for {user.email} from {request.META.get('REMOTE_ADDR')}")
    return {'ok': True, 'token': 'x'}
"""
    s_line = find_line(settings_code, "SECRET_KEY = 'literal-hardcoded-key")
    a_line = find_line(auth_code, "logger.info(f\"login success")
    d_line = find_line(settings_code, "DEBUG = True")

    return (
        {settings_file: settings_code, auth_file: auth_code},
        [
            Violation(settings_file, "OWASP-A02", "critical", s_line, s_line,
                      fix_hint="Load SECRET_KEY from os.environ."),
            Violation(settings_file, "GDPR-ART32", "critical", d_line, d_line,
                      fix_hint="DEBUG = False in production."),
            Violation(auth_file, "GDPR-ART5-1A", "high", a_line, a_line,
                      fix_hint="Log user id, never email/IP."),
        ],
    )


def _t_multi_file_idor_and_sql(ctx: TemplateCtx):
    views_file = ctx.fname("views_file", ["views.py", "api.py"])
    repo_file = ctx.fname("repo_file", ["repo.py", "queries.py"])

    views_code = """from django.shortcuts import get_object_or_404
from .models import Document
from .repo import get_user_search


def doc_view(request, pk):
    doc = get_object_or_404(Document, pk=pk)
    return {'title': doc.title, 'body': doc.body}


def search_view(request):
    return get_user_search(request.GET['q'])
"""
    repo_code = """from db import connection


def get_user_search(q):
    cur = connection.cursor()
    cur.execute(f"SELECT id, name FROM users WHERE name LIKE '%{q}%'")
    return cur.fetchall()
"""
    v_line = find_line(views_code, "doc = get_object_or_404(Document, pk=pk)")
    r_line = find_line(repo_code, 'cur.execute(f"SELECT id, name')
    return (
        {views_file: views_code, repo_file: repo_code},
        [
            Violation(views_file, "OWASP-A01", "high", v_line, v_line,
                      fix_hint="Add ownership filter when fetching Document."),
            Violation(repo_file, "OWASP-A03", "critical", r_line, r_line,
                      fix_hint="Use parameterised LIKE with %s."),
        ],
    )


def _t_security_api_key_leak(ctx: TemplateCtx):
    fname = ctx.fname("settings_file", ["config.py", "secrets.py", "env_loader.py"])
    code = """import os

DEBUG = False
ALLOWED_HOSTS = ['api.example.com']

API_KEY = 'sk-live-do-not-share-1234567890'

def get_client():
    return {'auth': API_KEY}
"""
    line = find_line(code, "API_KEY = 'sk-live")
    return {fname: code}, [
        Violation(fname, "OWASP-A02", "critical", line, line,
                  fix_hint="API_KEY = os.environ['API_KEY']"),
    ]


# ─── Adversarial templates (HARD) ────────────────────────────────────────────
#
# These either contain fake-safe traps (cheat fixes that pass CI but trigger
# the hidden compliance check) or multi-step dependencies that punish a
# greedy single-file fix.

def _t_adv_pii_hash_trap(ctx: TemplateCtx):
    """Easy fix is `hash(user.email)` — passes regex, fails hidden check."""
    fname = ctx.fname("auth_file", ["auth.py", "session.py"])
    code = """import logging
import hashlib

logger = logging.getLogger(__name__)


def on_login(user, request):
    # Tempting "fix": logger.info(hashlib.sha256(user.email.encode()).hexdigest())
    # That still leaks PII (deterministic hashing) — log the user id instead.
    logger.info(f"login ok for {user.email}")
    return True
"""
    line = find_line(code, 'logger.info(f"login ok')
    return {fname: code}, [
        Violation(fname, "GDPR-ART5-1A", "high", line, line,
                  fix_hint="logger.info('login uid=%s', user.id) — never hash/encode email."),
    ]


def _t_adv_secret_env_default(ctx: TemplateCtx):
    """Easy fix is `os.environ.get('SECRET_KEY', 'literal-fallback')` — still hardcoded."""
    fname = ctx.fname("settings_file", ["settings.py", "config.py"])
    code = """import os

DEBUG = False
SECRET_KEY = 'prod-literal-secret-do-not-share'
ALLOWED_HOSTS = ['*']
"""
    line = find_line(code, "SECRET_KEY = 'prod-literal-secret")
    return {fname: code}, [
        Violation(fname, "OWASP-A02", "critical", line, line,
                  fix_hint="SECRET_KEY = os.environ['SECRET_KEY'] — never use a string default."),
    ]


def _t_adv_partial_idor(ctx: TemplateCtx):
    """Two endpoints both with IDOR — fixing one leaves the other vulnerable."""
    fname = ctx.fname("views_file", ["views.py", "documents.py"])
    code = """from django.shortcuts import get_object_or_404
from django.http import JsonResponse
from .models import Document


def view_doc(request, pk):
    doc = get_object_or_404(Document, pk=pk)
    return JsonResponse({'title': doc.title})


def download_doc(request, pk):
    doc = get_object_or_404(Document, pk=pk)
    return JsonResponse({'body': doc.body})
"""
    l1 = find_line(code, "doc = get_object_or_404(Document, pk=pk)", occurrence=1)
    l2 = find_line(code, "doc = get_object_or_404(Document, pk=pk)", occurrence=2)
    return {fname: code}, [
        Violation(fname, "OWASP-A01", "high", l1, l1,
                  fix_hint="Add owner=request.user filter."),
        Violation(fname, "OWASP-A01", "high", l2, l2,
                  fix_hint="Add owner=request.user filter HERE TOO."),
    ]


def _t_adv_sql_strip_quote_trap(ctx: TemplateCtx):
    """Cheat fix: `name.replace(\"'\", \"\")` — still injectable via other chars."""
    fname = ctx.fname("repo_file", ["repo.py", "db_layer.py"])
    code = """from db import connection


def find_by_name(name):
    cur = connection.cursor()
    cur.execute(f"SELECT id FROM users WHERE name = '{name}'")
    return cur.fetchone()
"""
    line = find_line(code, 'cur.execute(f"SELECT id FROM users')
    return {fname: code}, [
        Violation(fname, "OWASP-A03", "critical", line, line,
                  fix_hint="Use parameterised query with %s — string sanitisation is not safe."),
    ]


def _t_adv_multi_file_secret_import(ctx: TemplateCtx):
    """Settings has the secret; auth imports it. Fixing settings breaks auth's import."""
    settings_file = ctx.fname("settings_file", ["settings.py", "config.py"])
    auth_file = ctx.fname("auth_file", ["auth_service.py", "tokens.py"])

    settings_code = """import os

DEBUG = False
SECRET_KEY = 'literal-secret-very-long-key-shipped-in-source'
ALLOWED_HOSTS = ['*']
"""
    auth_code = """from .settings import SECRET_KEY


def make_token(user_id):
    return f"{user_id}:{SECRET_KEY}"
"""
    s_line = find_line(settings_code, "SECRET_KEY = 'literal-secret")
    return (
        {settings_file: settings_code, auth_file: auth_code},
        [
            Violation(settings_file, "OWASP-A02", "critical", s_line, s_line,
                      fix_hint="Load SECRET_KEY from os.environ; downstream imports keep working."),
        ],
    )


def _t_adv_debug_env_default(ctx: TemplateCtx):
    """`DEBUG = os.environ.get('DEBUG', True)` — env-aware but defaults dangerous."""
    fname = ctx.fname("settings_file", ["settings.py", "config.py"])
    code = """import os

DEBUG = True
ALLOWED_HOSTS = ['*']

LOGGING = {'version': 1}
"""
    line = find_line(code, "DEBUG = True")
    return {fname: code}, [
        Violation(fname, "GDPR-ART32", "critical", line, line,
                  fix_hint="DEBUG = False — do not load DEBUG from env with True default."),
    ]


def _t_gdpr_dict_password_leak(ctx: TemplateCtx):
    """Direct password_hash key in a returned dict (no to_dict() wrapper)."""
    fname = ctx.fname("api_file", ["users_api.py", "session_api.py"])
    code = """from flask import Flask, jsonify, request

app = Flask(__name__)

DB = {}


@app.route('/me')
def me():
    uid = int(request.args.get('uid', 0))
    user = DB.get(uid, {'id': uid, 'email': 'x@y', 'password_hash': 'sha:abc'})
    return jsonify({'id': user['id'], 'email': user['email'], 'password_hash': user['password_hash']})
"""
    line = find_line(code, "return jsonify({'id': user['id'],")
    return {fname: code}, [
        Violation(fname, "GDPR-ART5-1C", "high", line, line,
                  fix_hint="Drop password_hash from the returned JSON."),
    ]


# Registry of templates the generator will sample from.
# difficulty drives sampling weights AND the heuristic's failure rate.
TEMPLATES: List[Template] = [
    # ── EASY (single-file, single-violation, no traps) ───────────────────────
    Template("gdpr_log_pii", "gdpr", ["GDPR"],
             "Login handler logs personal data (email, IP).",
             _t_gdpr_log_pii, difficulty="easy"),
    Template("gdpr_email_in_print", "gdpr", ["GDPR"],
             "Print statement leaks user email.",
             _t_gdpr_email_in_print, difficulty="easy"),
    Template("sec_hardcoded_secret", "security", ["OWASP"],
             "SECRET_KEY / API_KEY hardcoded in settings.",
             _t_security_hardcoded_secret, difficulty="easy"),
    Template("sec_debug_true", "security", ["GDPR", "OWASP"],
             "Production settings ship with DEBUG = True.",
             _t_security_debug_true, difficulty="easy"),
    Template("sec_api_key_leak", "security", ["OWASP"],
             "Hardcoded API_KEY in config file.",
             _t_security_api_key_leak, difficulty="easy"),
    Template("gdpr_dict_password_leak", "gdpr", ["GDPR"],
             "Endpoint serialises password_hash directly.",
             _t_gdpr_dict_password_leak, difficulty="easy"),

    # ── MEDIUM (single-file but trickier shape OR uses red-herrings) ─────────
    Template("gdpr_leak_pwd_hash", "gdpr", ["GDPR"],
             "Profile API returns password_hash + internal_id.",
             _t_gdpr_leak_password_hash, difficulty="medium"),
    Template("sec_sql_injection", "security", ["OWASP"],
             "Repo uses f-string SQL → injection.",
             _t_security_sql_injection, difficulty="medium"),
    Template("sec_idor", "security", ["OWASP"],
             "Object lookup without ownership check (IDOR).",
             _t_security_idor, difficulty="medium"),
    Template("cq_bare_except_pii", "code_quality", ["GDPR"],
             "Bare except masks errors AND logs customer email.",
             _t_code_quality_bare_except, difficulty="medium"),

    # ── HARD (multi-file or adversarial fake-safe traps) ─────────────────────
    Template("multi_secret_and_pii", "multi_file", ["GDPR", "OWASP"],
             "Settings + auth: secret + DEBUG + PII log.",
             _t_multi_file_secret_and_pii, difficulty="hard"),
    Template("multi_idor_and_sql", "multi_file", ["OWASP"],
             "Views + repo: IDOR plus SQL injection across files.",
             _t_multi_file_idor_and_sql, difficulty="hard"),
    Template("adv_pii_hash_trap", "adversarial", ["GDPR"],
             "Tempting cheat: hash(email). Hidden oracle catches this.",
             _t_adv_pii_hash_trap, difficulty="hard", adversarial=True),
    Template("adv_secret_env_default", "adversarial", ["OWASP"],
             "Tempting cheat: os.environ.get('SECRET_KEY','literal').",
             _t_adv_secret_env_default, difficulty="hard", adversarial=True),
    Template("adv_partial_idor", "adversarial", ["OWASP"],
             "Two IDOR endpoints — easy to fix one, miss the other.",
             _t_adv_partial_idor, difficulty="hard", adversarial=True),
    Template("adv_sql_strip_quote", "adversarial", ["OWASP"],
             "Tempting cheat: name.replace(\"'\",\"\") sanitisation.",
             _t_adv_sql_strip_quote_trap, difficulty="hard", adversarial=True),
    Template("adv_multi_file_secret_import", "adversarial", ["OWASP"],
             "Cross-file: fixing the secret without breaking the importer.",
             _t_adv_multi_file_secret_import, difficulty="hard", adversarial=True),
    Template("adv_debug_env_default", "adversarial", ["GDPR"],
             "Tempting cheat: DEBUG = os.environ.get('DEBUG', True).",
             _t_adv_debug_env_default, difficulty="hard", adversarial=True),
]


# ─── Mutations ────────────────────────────────────────────────────────────────

def _maybe_rename_files(codebase: Dict[str, str], rng: random.Random) -> Dict[str, str]:
    """Rename files with low probability — adds variety without breaking refs."""
    if rng.random() > 0.25 or len(codebase) > 1:
        # don't mess with multi-file tasks (refs would break)
        return codebase
    only = next(iter(codebase))
    suffix = rng.choice(["_v2", "_handler", "_app", ""])
    new_name = only.replace(".py", f"{suffix}.py")
    if new_name == only:
        return codebase
    return {new_name: codebase[only]}


def _retarget_violations(violations: List[Violation], old_to_new: Dict[str, str]) -> List[Violation]:
    out = []
    for v in violations:
        out.append(Violation(
            file=old_to_new.get(v.file, v.file),
            rule_id=v.rule_id,
            severity=v.severity,
            line_start=v.line_start,
            line_end=v.line_end,
            fix_hint=v.fix_hint,
        ))
    return out


def mutate(
    codebase: Dict[str, str],
    violations: List[Violation],
    ctx: TemplateCtx,
) -> Tuple[Dict[str, str], List[Violation]]:
    """Apply a stack of mutations: padding, red-herrings, severity-shift, rename."""
    # 1. add padding so line numbers shift away from the template defaults
    padded = {f: with_padding(c, ctx.extra_padding_lines) for f, c in codebase.items()}
    violations = pad_violation_lines(violations, ctx.extra_padding_lines)

    # 2. red-herrings — but only on a copy because they shift lines unpredictably
    if ctx.add_red_herrings:
        # inject only into files that have NO violations to preserve line accuracy
        files_with_v = {v.file for v in violations}
        for fname in list(padded.keys()):
            if fname not in files_with_v:
                padded[fname] = inject_red_herrings(padded[fname], ctx.rng)

    # 3. severity shift
    violations = [
        Violation(v.file, v.rule_id, shift_severity(v.severity, ctx.severity_shift),
                  v.line_start, v.line_end, v.fix_hint)
        for v in violations
    ]

    # 4. file rename (single-file only)
    old_files = list(padded.keys())
    renamed = _maybe_rename_files(padded, ctx.rng)
    if list(renamed.keys()) != old_files:
        mapping = dict(zip(old_files, renamed.keys()))
        violations = _retarget_violations(violations, mapping)
        padded = renamed

    return padded, violations


# ─── Top-level generation ─────────────────────────────────────────────────────

def _violation_to_dict(v: Violation) -> Dict:
    return {
        "file": v.file,
        "rule_id": v.rule_id,
        "severity": v.severity,
        "line_start": v.line_start,
        "line_end": v.line_end,
    }


def _ground_truth(violations: List[Violation]) -> List[Dict]:
    return [
        {
            "file": v.file,
            "rule_id": v.rule_id,
            "fix": v.fix_hint or "Apply the standard remediation for this rule.",
        }
        for v in violations
    ]


def generate_task(template: Template, variant_idx: int, rng: random.Random) -> Dict:
    """Generate ONE mutated task instance from a template."""
    ctx = TemplateCtx(
        rng=rng,
        severity_shift=rng.choice([-1, 0, 0, 0, 1]),
        add_red_herrings=rng.random() < 0.4,
        extra_padding_lines=rng.choice([0, 0, 2, 5, 8]),
    )

    base_codebase, base_violations = template.builder(ctx)
    codebase, violations = mutate(base_codebase, base_violations, ctx)

    fingerprint = template.name + "|" + str(variant_idx) + "|" + str(sorted(codebase.keys()))
    task_id = f"{template.name}_v{variant_idx}_{short_hash(fingerprint, 6)}"

    return {
        "task_id": task_id,
        "category": template.category,
        "framework": template.frameworks,
        "description": template.description,
        "difficulty": template.difficulty,
        "adversarial": template.adversarial,
        "codebase": codebase,
        "violations": [_violation_to_dict(v) for v in violations],
        "ground_truth": _ground_truth(violations),
        "max_steps": 12 + 4 * len(codebase),
        "file_reads_remaining": max(2, len(codebase) + 1),
        "mutation": {
            "severity_shift": ctx.severity_shift,
            "added_red_herrings": ctx.add_red_herrings,
            "extra_padding_lines": ctx.extra_padding_lines,
        },
    }


# Target difficulty mix — used when no explicit `difficulty_mix` is passed.
# Roughly matches a real audit: most violations are easy/obvious, fewer are
# adversarial cheats.
DIFFICULTY_MIX = {"easy": 0.50, "medium": 0.30, "hard": 0.20}


def generate_tasks(
    num: int,
    seed: int = 42,
    difficulty_mix: Optional[Dict[str, float]] = None,
) -> List[Dict]:
    """
    Generate `num` tasks honouring the configured `difficulty_mix`
    (defaults to 50/30/20 easy/medium/hard).

    Within each difficulty bucket we cycle through the matching templates so
    every template is exercised at least once before duplicates appear.
    """
    seed_everything(seed)
    rng = random.Random(seed)
    mix = difficulty_mix or DIFFICULTY_MIX

    # How many tasks per bucket? round() then top-up to exact `num`.
    targets = {d: int(round(num * w)) for d, w in mix.items()}
    while sum(targets.values()) < num:
        targets["easy"] += 1
    while sum(targets.values()) > num:
        targets["hard"] = max(0, targets["hard"] - 1)

    by_difficulty: Dict[str, List[Template]] = {"easy": [], "medium": [], "hard": []}
    for t in TEMPLATES:
        by_difficulty.setdefault(t.difficulty, []).append(t)

    tasks: List[Dict] = []
    seen_ids: set[str] = set()
    variant_counter: Dict[str, int] = {t.name: 0 for t in TEMPLATES}

    for diff, n in targets.items():
        pool = by_difficulty.get(diff, [])
        if not pool:
            log.warning("No templates registered for difficulty=%r — skipping %d tasks", diff, n)
            continue
        for i in range(n):
            template = pool[i % len(pool)]
            for _ in range(20):     # try a few variants if id collides
                variant_counter[template.name] += 1
                task = generate_task(template, variant_counter[template.name], rng)
                if task["task_id"] not in seen_ids:
                    seen_ids.add(task["task_id"])
                    tasks.append(task)
                    break

    log.info("Generated %d tasks across %d templates", len(tasks), len(TEMPLATES))
    return tasks


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate compliance/security tasks.")
    p.add_argument("--num", type=int, default=40, help="Number of tasks to generate (default 40)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    p.add_argument("--out", type=str, default=str(TASKS_PATH), help="Output JSON path")
    return p


def main() -> None:
    args = _build_argparser().parse_args()
    tasks = generate_tasks(args.num, args.seed)
    write_json(args.out, tasks)
    log.info("Wrote %d tasks → %s", len(tasks), args.out)

    counts: Dict[str, int] = {}
    diffs: Dict[str, int] = {}
    adv = 0
    for t in tasks:
        counts[t["category"]] = counts.get(t["category"], 0) + 1
        diffs[t["difficulty"]] = diffs.get(t["difficulty"], 0) + 1
        if t.get("adversarial"):
            adv += 1
    for cat, n in sorted(counts.items()):
        log.info("  by category   %-15s %d", cat, n)
    for d in ("easy", "medium", "hard"):
        log.info("  by difficulty %-15s %d", d, diffs.get(d, 0))
    log.info("  adversarial tasks: %d", adv)


if __name__ == "__main__":
    main()
