from typing import Dict, List

# Task 1B: Two connected GDPR violations - both fixable in one trajectory

CODEBASE: Dict[str, str] = {
    "auth.py": """from flask import Flask, request, jsonify
import logging

app = Flask(__name__)
logger = logging.getLogger(__name__)

class User:
    def __init__(self, user_id, email, password_hash):
        self.id = user_id
        self.email = email
        self.password_hash = password_hash

@app.route('/login', methods=['POST'])
def login():
    email = request.json.get('email')
    password = request.json.get('password')
    
    # VIOLATION 1: GDPR-ART5-1A - Logging PII (email)
    logger.info(f"Login attempt for {email}")
    
    user = authenticate(email, password)
    if user:
        # VIOLATION 2: GDPR-ART5-1C - Exposing sensitive data
        return jsonify({'user': {'id': user.id, 'email': user.email, 'password_hash': user.password_hash}})
    return jsonify({'error': 'Invalid credentials'}), 401

def authenticate(email, password):
    # Mock authentication
    return User(1, email, 'hashed_password')
"""
}

GROUND_TRUTH: List[Dict] = [
    {"file": "auth.py", "rule_id": "GDPR-ART5-1A", "severity": "high", "line_start": 19, "line_end": 19},
    {"file": "auth.py", "rule_id": "GDPR-ART5-1C", "severity": "high", "line_start": 24, "line_end": 24},
]

def get_task() -> Dict:
    return {
        "task_id": "task1b_connected_violations",
        "codebase": CODEBASE,
        "ground_truth": GROUND_TRUTH,
        "framework": ["GDPR"],
        "file_reads_remaining": 3,
        "max_steps": 15,
        "description": "Fix 2 connected GDPR violations in authentication flow.",
    }
