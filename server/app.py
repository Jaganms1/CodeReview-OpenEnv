"""
server/app.py — OpenEnv-compatible server entry point.

This module provides an alternative way to start the FastAPI
server, importing the app directly from server.py.
"""

from __future__ import annotations

import sys
import os

# Add parent directory to path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    """Start the server."""
    import uvicorn
    # Import server.py directly using importlib to avoid name conflict with server/ package
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "server_module",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "server.py")
    )
    server_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(server_module)

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(server_module.app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
