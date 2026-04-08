"""
server/app.py — OpenEnv-compatible server entry point.

This module provides the FastAPI application instance that serves
the Code Review RL Environment as a REST API.
"""

from __future__ import annotations

import sys
import os

# Add parent directory to path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import app

def main():
    """Start the server."""
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
