"""
FastAPI Main Application Entrypoint (Section 3 & Section 5)
Mounts routers and implements the consistent response envelope: { data, error, request_id }
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
import uuid
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

import logging

# Ensure project root is in sys.path when starting app from backend directory
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Automatically load environment variables from root .env
env_file = root_dir / ".env"
if env_file.exists():
    load_dotenv(dotenv_path=env_file, override=True)
else:
    load_dotenv(find_dotenv(), override=True)

# Configure standard Python logging
log_level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
logging.basicConfig(
    level=log_level,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

from backend.app.api.admin import router as admin_router
from backend.app.api.auth import router as auth_router
from backend.app.api.chat import router as chat_router
from backend.app.api.documents import router as documents_router
from backend.app.api.eval import router as eval_router
from backend.app.api.health import router as health_router

app = FastAPI(
    title="Enterprise Knowledge Intelligence Platform API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Standard Response Envelope for HTTPExceptions
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    code_map = {
        400: "BAD_REQUEST",
        401: "UNAUTHORIZED",
        403: "FORBIDDEN",
        404: "NOT_FOUND",
        409: "CONFLICT",
        500: "INTERNAL_SERVER_ERROR",
        503: "SERVICE_UNAVAILABLE"
    }
    error_code = code_map.get(exc.status_code, "HTTP_ERROR")
    headers = dict(getattr(exc, "headers", None) or {})
    headers["X-Request-ID"] = request_id
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "data": None,
            "error": {
                "code": error_code,
                "message": exc.detail
            },
            "request_id": request_id
        },
        headers=headers
    )


# Response envelope middleware per Section 5 ({ data, error, request_id })
@app.middleware("http")

async def response_envelope_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
    except NotImplementedError as exc:
        return JSONResponse(
            status_code=501,
            content={
                "data": None,
                "error": {
                    "code": "NOT_IMPLEMENTED",
                    "message": str(exc)
                },
                "request_id": request_id
            },
            headers={"X-Request-ID": request_id}
        )
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={
                "data": None,
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": str(exc)
                },
                "request_id": request_id
            },
            headers={"X-Request-ID": request_id}
        )


from fastapi.middleware.cors import CORSMiddleware

allowed_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8080",
    "http://localhost:8000",
    "http://localhost",
    "https://enterprise-rag-platform-blond.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=r"https?://.*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# Mount routers matching Section 5 contracts
app.include_router(health_router)
app.include_router(auth_router)
app.include_router(documents_router)
app.include_router(chat_router)
app.include_router(eval_router)
app.include_router(admin_router)
