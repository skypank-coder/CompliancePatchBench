"""Hugging Face Spaces entry point for the CompliancePatchBench API."""

from api.server import app

if __name__ == "__main__":
    import os
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "7860")),
        reload=False
    )
