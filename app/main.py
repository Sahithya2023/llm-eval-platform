"""ASGI entry point.

Run the API with:
    uvicorn app.main:app --reload
"""

from app.api.app import create_app

app = create_app()
