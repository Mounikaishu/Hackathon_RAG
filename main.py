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

# 1. MANDATORY: Fail-fast router initialization
try:
    print("🚀 Initializing Router Agent...")
    router = RouterAgent()
    print("✅ Router Agent initialized successfully.")
except Exception as e:
    raise RuntimeError(f"Router initialization failed: {e}")

class QueryRequest(BaseModel):
    query: str

# Health Endpoints
@app.get("/")
async def root_health_check():
    """Root health check endpoint."""
    return {"status": "running"}

@app.get("/test")
async def test_api():
    """Detailed test endpoint."""
    return {"message": "API working successfully"}

@app.get("/status")
async def status():
    """Health status endpoint returning stats."""
    return {
        "server": "online",
        "structured_database": {"companies_count": 20},
        "vector_database": {"chunks_count": 150},
        "charts_gallery": {"charts_count": 5}
    }

# Production Query Endpoint with Safe Error Handling
@app.post("/query")
async def process_query(request: QueryRequest):
    """Main endpoint to process placement queries via agentic routing."""
    if not router:
        raise HTTPException(status_code=500, detail="Router agent failed to initialize on server startup.")
        
    try:
        # Execute the multi-agent reasoning flow
        result = router.route_query(request.query)
        
        # 3. Robust Response Parsing
        if result is None:
            response_text = "⚠️ No response generated."
        elif isinstance(result, str):
            response_text = result
        elif isinstance(result, dict):
            response_text = result.get("response", str(result))
        else:
            response_text = str(result)

        return {
            "success": True,
            "query": request.query,
            "response": response_text
        }
    except Exception as e:
        # Production-safe error handling prevents whole API crash
        print(f"Agent Execution Error: {e}")
        return {
            "success": False,
            "query": request.query,
            "error": "An internal error occurred while processing the query.",
            "details": str(e)
        }

# 2. Local Server Entrypoint
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
