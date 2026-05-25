import os
import shutil
from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException
from pydantic import BaseModel
from app.config import settings
from app.embeddings.indexing_pipeline import IndexingPipeline
from app.agents.router_agent import RouterAgent
from app.agents.rag_agent import RagAgent
from app.agents.dataframe_agent import DataframeAgent
from app.agents.vision_agent import VisionAgent
from app.agents.conflict_agent import ConflictAgent
from app.agents.web_search_agent import WebSearchAgent

router = APIRouter()

# Schema models
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    query: str
    routed_agent: str
    routing_reason: str
    response: str

# Instantiate Agents globally for session reuse
router_agent = RouterAgent()
rag_agent = RagAgent()
dataframe_agent = DataframeAgent()
vision_agent = VisionAgent()
conflict_agent = ConflictAgent()
web_search_agent = WebSearchAgent()

def run_indexing_background(pdf_path: str):
    """Triggers the heavy indexing pipeline in a background thread."""
    try:
        pipeline = IndexingPipeline(pdf_path=pdf_path)
        pipeline.run()
    except Exception as e:
        print(f"❌ Background Indexing Error: {e}")

@router.post("/query", response_model=QueryResponse)
async def query_system(request: QueryRequest):
    """
    Main query gateway.
    Routes incoming questions to the optimal agent and returns synthesized trace-backed answers.
    """
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # 1. Route the query using our central router agent
    route_details = router_agent.route_query(query)
    target_agent = route_details["agent"]
    reason = route_details["reason"]
    company = route_details["entities"].get("company")

    # 2. Invoke the target specialized agent
    response_text = ""
    try:
        if target_agent == "dataframe_agent":
            response_text = dataframe_agent.process_query(query)
        elif target_agent == "vision_agent":
            response_text = vision_agent.process_query(query)
        elif target_agent == "conflict_agent":
            response_text = conflict_agent.process_query(query, company=company)
        elif target_agent == "web_search_agent":
            response_text = web_search_agent.process_query(query)
        else:
            # Default fallback: RAG agent
            response_text = rag_agent.process_query(query, company_filter=company)
    except Exception as e:
        response_text = f"❌ Agent Routing Execution Error: {str(e)}"
        target_agent = "error_agent"

    return QueryResponse(
        query=query,
        routed_agent=target_agent,
        routing_reason=reason,
        response=response_text
    )

@router.post("/upload/pdf")
async def upload_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Accepts PDF file uploads, saves them to data/raw/,
    and triggers parsing and indexing in the background.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF file uploads are accepted.")

    settings.ensure_directories()
    target_path = os.path.join(settings.RAW_DATA_DIR, "Placement_RAG_Dataset_Enhanced.pdf")

    try:
        with open(target_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Trigger indexing in the background
        background_tasks.add_task(run_indexing_background, target_path)

        return {
            "status": "success",
            "message": f"Successfully uploaded {file.filename}. Ingestion and indexing pipeline started in the background.",
            "path": target_path
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload PDF file: {str(e)}")

@router.post("/upload/table")
async def upload_table(file: UploadFile = File(...)):
    """
    Direct upload route for structured JSON tabular company records.
    Saves directly to data/processed/eligibility_table.json.
    """
    if not file.filename.endswith(".json"):
        raise HTTPException(status_code=400, detail="Only JSON file uploads are accepted for structured tables.")

    settings.ensure_directories()
    target_path = os.path.join(settings.PROCESSED_DIR, "eligibility_table.json")

    try:
        with open(target_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        # Reload dataframe tool inside the global agent context
        dataframe_agent.pandas_tool = dataframe_agent.pandas_tool.__class__()
        return {
            "status": "success",
            "message": f"Successfully uploaded structured table {file.filename}. Eligibility records are instantly live.",
            "path": target_path
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload structured table: {str(e)}")

@router.post("/upload/image")
async def upload_image(file: UploadFile = File(...)):
    """
    Direct upload route for chart images.
    Saves image directly under data/processed/charts/ for visual agent access.
    """
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".png", ".jpg", ".jpeg"]:
        raise HTTPException(status_code=400, detail="Only standard image files (PNG, JPG, JPEG) are accepted.")

    settings.ensure_directories()
    target_path = os.path.join(settings.CHARTS_DIR, file.filename)

    try:
        with open(target_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return {
            "status": "success",
            "message": f"Successfully uploaded chart image {file.filename} to visual gallery.",
            "path": target_path
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload chart image: {str(e)}")

@router.get("/status")
async def check_status():
    """Returns database status and document count statistics."""
    df = dataframe_agent.pandas_tool.df
    db = rag_agent.db_manager.collection
    
    total_companies = len(df) if not df.empty else 0
    try:
        total_vectors = db.count()
    except Exception:
        total_vectors = 0

    return {
        "status": "active",
        "structured_database": {
            "loaded": not df.empty,
            "companies_count": total_companies
        },
        "vector_database": {
            "loaded": total_vectors > 0,
            "chunks_count": total_vectors
        },
        "charts_gallery": {
            "loaded": os.path.exists(settings.CHARTS_DIR) and len(os.listdir(settings.CHARTS_DIR)) > 0,
            "charts_count": len(os.listdir(settings.CHARTS_DIR)) if os.path.exists(settings.CHARTS_DIR) else 0
        }
    }
