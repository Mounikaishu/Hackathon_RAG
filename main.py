import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Deploy-safe imports
from app.agents.router_agent import RouterAgent

# Swagger polish
app = FastAPI(
    title="Placement Intelligence Agentic RAG",
    description="AI-powered placement assistant with multi-agent reasoning",
    version="1.0.0"
)

# Add CORS middleware for frontend (Vercel)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Soft-fail router initialization — backend must start even if model load fails
try:
    print("🚀 Initializing Router Agent...")
    router = RouterAgent()
    print("✅ Router Agent initialized successfully.")
except Exception as e:
    print(f"⚠️ Router Agent initialization warning: {e}")
    router = None

# Mount processed directory to serve static assets (such as charts PNGs)
import os
from fastapi.staticfiles import StaticFiles
processed_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "processed")
os.makedirs(processed_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=processed_dir), name="static")

# Mount React frontend static assets if available
react_assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend", "dist", "assets")
os.makedirs(react_assets_dir, exist_ok=True)
app.mount("/assets", StaticFiles(directory=react_assets_dir), name="assets")

# Include the real API Router from app.api.routes directly at the root
from app.api.routes import router as api_router
app.include_router(api_router)

# 2. Local Server Entrypoint
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
