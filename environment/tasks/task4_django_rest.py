from typing import Dict, List

# Task 4: Django REST API with 4 GDPR/OWASP violations

CODEBASE: Dict[str, str] = {
    "api/views.py": """from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from django.contrib.auth.models import User
from .models import UserProfile
from .serializers import UserProfileSerializer
import logging

logger = logging.getLogger(__name__)

class UserProfileViewSet(viewsets.ModelViewSet):
    queryset = UserProfile.objects.all()
    serializer_class = UserProfileSerializer
    
    @action(detail=False, methods=['post'])
    def register(self, request):
        email = request.data.get('email')
        password = request.data.get('password')
        
        # VIOLATION 1: GDPR-ART5-1A - Logging PII
        logger.info(f"New user registration: {email}")
        
        user = User.objects.create_user(
            username=email,
            email=email,
            password=password
        )
        
        profile = UserProfile.objects.create(
            user=user,
            phone=request.data.get('phone', '')
        )
        
        return Response({'status': 'created'}, status=status.HTTP_201_CREATED)
    
    @action(detail=True, methods=['get'])
    def profile_detail(self, request, pk=None):
        # VIOLATION 2: OWASP-A01 - Missing authorization check
        profile = UserProfile.objects.get(pk=pk)
        
        # VIOLATION 3: GDPR-ART5-1C - Exposing sensitive data
        return Response({
            'id': profile.id,
            'email': profile.user.email,
            'phone': profile.phone,
            'internal_id': profile.internal_id,
            'api_key': profile.api_key
        })
    
    @action(detail=False, methods=['get'])
    def search(self, request):
        query = request.query_params.get('q', '')
        
        # VIOLATION 4: OWASP-A03 - SQL injection via raw query
        profiles = UserProfile.objects.raw(
            f"SELECT * FROM api_userprofile WHERE phone LIKE '%{query}%'"
        )
        
        serializer = self.get_serializer(profiles, many=True)
        return Response(serializer.data)
"""
}

GROUND_TRUTH: List[Dict] = [
    {"file": "api/views.py", "rule_id": "GDPR-ART5-1A", "severity": "high", "line_start": 21, "line_end": 21},
    {"file": "api/views.py", "rule_id": "OWASP-A01", "severity": "high", "line_start": 38, "line_end": 38},
    {"file": "api/views.py", "rule_id": "GDPR-ART5-1C", "severity": "high", "line_start": 41, "line_end": 47},
    {"file": "api/views.py", "rule_id": "OWASP-A03", "severity": "critical", "line_start": 54, "line_end": 56},
]

def get_task() -> Dict:
    return {
        "task_id": "task4_django_rest",
        "codebase": CODEBASE,
        "ground_truth": GROUND_TRUTH,
        "framework": ["GDPR", "OWASP"],
        "file_reads_remaining": 3,
        "max_steps": 20,
        "description": "Audit Django REST API for GDPR/OWASP violations.",
    }
