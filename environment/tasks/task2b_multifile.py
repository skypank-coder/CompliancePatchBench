from typing import Dict, List

# Task 2B: Multi-file dependency - fix in one file affects another

CODEBASE: Dict[str, str] = {
    "models.py": """class User:
    def __init__(self, user_id, email):
        self.id = user_id
        self.email = email
    
    def to_dict(self):
        # VIOLATION 1: GDPR-ART5-1C - Exposing email in serialization
        return {'id': self.id, 'email': self.email}
""",
    
    "api.py": """from flask import jsonify
from models import User

def get_user_profile(user_id):
    user = User(user_id, 'user@example.com')
    # VIOLATION 2: Depends on User.to_dict() which exposes email
    # If models.py is fixed but this isn't updated, it breaks
    return jsonify(user.to_dict())

def get_user_list():
    users = [User(1, 'a@example.com'), User(2, 'b@example.com')]
    # This also depends on to_dict()
    return jsonify([u.to_dict() for u in users])
"""
}

GROUND_TRUTH: List[Dict] = [
    {"file": "models.py", "rule_id": "GDPR-ART5-1C", "severity": "high", "line_start": 7, "line_end": 7},
    {"file": "api.py", "rule_id": "GDPR-ART5-1C", "severity": "medium", "line_start": 7, "line_end": 7},
]

def get_task() -> Dict:
    return {
        "task_id": "task2b_multifile_dependency",
        "codebase": CODEBASE,
        "ground_truth": GROUND_TRUTH,
        "framework": ["GDPR"],
        "file_reads_remaining": 5,
        "max_steps": 20,
        "description": "Fix cross-file GDPR violations - models.py fix affects api.py.",
    }
