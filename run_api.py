"""Start the signal API with Module A and Module B subscribed.

    python run_api.py            -> http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import logging

import uvicorn

from module_b.api import build_router
from risk_engine.api import create_app
from risk_engine.config import API_HOST, API_PORT
from system import build_system


def make_app():
    sys_ = build_system()
    app = create_app(sys_.pipeline, sys_.store)
    app.include_router(build_router(sys_.stress))
    app.state.system = sys_
    return app


app = make_app()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host=API_HOST, port=API_PORT)
