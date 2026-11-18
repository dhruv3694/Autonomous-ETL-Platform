"""AI-Powered Autonomous ETL Platform - Backend Application Entrypoint."""
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from backend.api.routes import router as api_router

app = FastAPI(
    title="AI-Powered Autonomous ETL Platform",
    description="Qwen-powered ETL automation platform with human-in-the-loop approval and comprehensive data quality analytics.",
    version="1.0.0"
)

# Enable CORS for local web development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STORAGE_ROOT = Path(__file__).resolve().parent.parent / "storage"
FRONTEND_STATIC = Path(__file__).resolve().parent.parent / "frontend" / "static"

# Mount static files for the React frontend dashboard
if FRONTEND_STATIC.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_STATIC)), name="static")

# Include API routes
app.include_router(api_router)

@app.get("/")
def serve_dashboard():
    index_file = FRONTEND_STATIC / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "AI ETL Platform Backend is running."}

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "AI-Powered Autonomous ETL Platform",
        "version": "1.0.0",
        "storage_initialized": STORAGE_ROOT.exists()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
