from typing import Dict, List

# Task 5: FastAPI Microservice with 3 GDPR/OWASP violations

CODEBASE: Dict[str, str] = {
    "main.py": """from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
import os
import logging

app = FastAPI()
logger = logging.getLogger(__name__)

# VIOLATION 1: OWASP-A02 - Hardcoded secret
SECRET_KEY = "hardcoded-secret-key-12345"
API_TOKEN = "sk-1234567890abcdef"

class User(BaseModel):
    id: int
    email: str
    password_hash: str
    api_key: Optional[str] = None

@app.post("/api/login")
async def login(email: str, password: str):
    # VIOLATION 2: GDPR-ART5-1A - Logging PII
    logger.info(f"Login attempt from {email}")
    
    # Mock authentication
    user = User(
        id=1,
        email=email,
        password_hash="hashed",
        api_key="key-123"
    )
    
    # VIOLATION 3: GDPR-ART5-1C - Exposing sensitive data
    return {
        "user": {
            "id": user.id,
            "email": user.email,
            "password_hash": user.password_hash,
            "api_key": user.api_key
        },
        "token": SECRET_KEY
    }

@app.get("/api/users/{user_id}")
async def get_user(user_id: int):
    user = User(
        id=user_id,
        email="user@example.com",
        password_hash="hashed",
        api_key="key-456"
    )
    
    return user.dict()
"""
}

GROUND_TRUTH: List[Dict] = [
    {"file": "main.py", "rule_id": "OWASP-A02", "severity": "critical", "line_start": 11, "line_end": 12},
    {"file": "main.py", "rule_id": "GDPR-ART5-1A", "severity": "high", "line_start": 24, "line_end": 24},
    {"file": "main.py", "rule_id": "GDPR-ART5-1C", "severity": "high", "line_start": 35, "line_end": 43},
]

def get_task() -> Dict:
    return {
        "task_id": "task5_fastapi",
        "codebase": CODEBASE,
        "ground_truth": GROUND_TRUTH,
        "framework": ["GDPR", "OWASP"],
        "file_reads_remaining": 2,
        "max_steps": 15,
        "description": "Audit FastAPI microservice for GDPR/OWASP violations.",
    }
