"""FastAPI application for HF Agent web interface."""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.staticfiles import StaticFiles
from routes.agent import router as agent_router
from routes.auth import router as auth_router
from routes.platform import router as platform_router
from session_manager import session_manager

# Load .env — search from this file's location upward so it works regardless of CWD
for _candidate in (
    Path(__file__).parent / ".env",
    Path(__file__).parent.parent / ".env",
):
    if _candidate.exists():
        load_dotenv(_candidate, override=True)
        break

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("Starting HF Agent backend...")
    await session_manager.start()
    try:
        yield
    finally:
        logger.info("Shutting down HF Agent backend...")
        await session_manager.close()


app = FastAPI(
    title="HF Agent",
    description="ML Engineering Assistant API",
    version="1.0.0",
    lifespan=lifespan,
)

_DEFAULT_CORS_ORIGINS = [
    "http://localhost:5173",  # Vite dev server
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
]
_configured_origins = [
    origin.strip()
    for origin in os.environ.get("CORS_ALLOW_ORIGINS", "").split(",")
    if origin.strip()
]
_allowed_hosts = [
    host.strip()
    for host in os.environ.get("ALLOWED_HOSTS", "").split(",")
    if host.strip()
]

if _allowed_hosts:
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=_allowed_hosts)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    """Attach conservative security headers to every response."""
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault(
        "Permissions-Policy",
        "camera=(), microphone=(), geolocation=()",
    )
    return response


# CORS middleware defaults to local dev; production can set CORS_ALLOW_ORIGINS.
app.add_middleware(
    CORSMiddleware,
    allow_origins=_configured_origins or _DEFAULT_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(agent_router)
app.include_router(auth_router)
app.include_router(platform_router)


@app.get("/api")
async def api_root():
    """API root endpoint."""
    return {
        "name": "HF Agent API",
        "version": "1.0.0",
        "docs": "/docs",
    }


# Serve static files — check backend/static/ first (local dev + Docker), then /app/static/
static_path = Path(__file__).parent / "static"
if not static_path.exists():
    static_path = Path(__file__).parent.parent / "static"
if static_path.exists():
    app.mount("/", StaticFiles(directory=str(static_path), html=True), name="static")
    logger.info(f"Serving static files from {static_path}")
else:
    logger.info("No static directory found, running in API-only mode")


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
