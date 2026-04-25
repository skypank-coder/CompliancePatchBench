#!/bin/bash
# Pre-commit hook for CompliancePatchBench
# Place in .git/hooks/pre-commit and make executable

echo "Running compliance audit..."

# Start API server in background
uvicorn api.server:app --host 0.0.0.0 --port 7860 &
API_PID=$!
sleep 3

# Run audit on staged Python files
STAGED_FILES=$(git diff --cached --name-only --diff-filter=ACM | grep '\.py$')

if [ -z "$STAGED_FILES" ]; then
    echo "No Python files to audit"
    kill $API_PID
    exit 0
fi

# Run inference
python inference.py

RESULT=$?

# Cleanup
kill $API_PID

if [ $RESULT -ne 0 ]; then
    echo "❌ Compliance violations found. Commit blocked."
    echo "Run 'python inference.py' to see details."
    exit 1
fi

echo "✅ No compliance violations detected"
exit 0
