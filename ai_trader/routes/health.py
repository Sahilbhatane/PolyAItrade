"""Health check endpoints."""

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/")
async def root() -> dict[str, str]:
    """Landing endpoint so opening http://localhost:8000 in a browser does not 404."""
    return {
        "service": "PolyVITrade",
        "status": "running",
        "health": "/health",
        "ready": "/ready",
        "docs": "/docs",
        "openapi": "/openapi.json",
    }


@router.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready")
async def readiness_check() -> dict[str, str]:
    return {"status": "ready"}
