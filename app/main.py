"""
Compatibility entry point for app.main.
Re-exports the FastAPI app from root main.py.
"""

from main import app

__all__ = ["app"]
