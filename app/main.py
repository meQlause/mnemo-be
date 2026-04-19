import os

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi_csrf_protect import CsrfProtect
from fastapi_csrf_protect.exceptions import CsrfProtectError
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware

from app.api.router import api_router
from app.core.config import settings
from app.core.logger import setup_app_logging

# Initialize logging
setup_app_logging()

app = FastAPI(
    title="RAG Notes API",
    description="Generate structured notes using Retrieval-Augmented Generation.",
    version="1.0.0",
)


class CsrfSettings(BaseModel):
    secret_key: str = settings.SECRET_KEY_CSRF
    cookie_samesite: str = settings.COOKIE_SAMESITE
    cookie_secure: bool = settings.COOKIE_SECURE
    cookie_key: str = settings.CSRF_TOKEN_COOKIE_NAME
    cookie_httponly: bool = False
    header_name: str = "X-CSRF-Token"


@CsrfProtect.load_config
def get_csrf_config():
    return CsrfSettings()


@app.exception_handler(CsrfProtectError)
def csrf_protect_exception_handler(request: Request, exc: CsrfProtectError):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})


class DynamicCORSMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        origin = request.headers.get("origin")
        response = await call_next(request)
        if origin:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Credentials"] = "true"
            response.headers["Access-Control-Allow-Methods"] = (
                "GET, POST, PUT, DELETE, OPTIONS, PATCH"
            )
            response.headers["Access-Control-Allow-Headers"] = (
                "Content-Type, Authorization, X-CSRF-Token"
            )
        return response


app.add_middleware(DynamicCORSMiddleware)

app.include_router(api_router, prefix="/api/v1")

frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(frontend_path):
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_frontend():
        return FileResponse(os.path.join(frontend_path, "index.html"))


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok"}
