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
from app.agents.multi_hop_agent import MultiHopAgent

router = APIRouter()

from typing import Optional

# Schema models
class QueryRequest(BaseModel):
    query: str
    force_web_search: Optional[bool] = False

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
multi_hop_agent = MultiHopAgent()

def run_indexing_background(pdf_path: str):
    """Triggers the heavy indexing pipeline in a background thread."""
    try:
        pipeline = IndexingPipeline(pdf_path=pdf_path)
        pipeline.run()
    except Exception as e:
        print(f"❌ Background Indexing Error: {e}")

def clean_and_summarize_global(client, query: str, raw_response: str) -> str:
    """Helper to summarize web search responses to 1-2 clean sentences."""
    if not client:
        return raw_response.replace("🌐 **[Real-time Web Search Fallback Agent | Querying Live Databases]**\n\n", "")
    
    prompt = (
        "You are an assistant. Clean up and format the following web search response to be extremely short, direct, and concise (1-2 sentences max).\n"
        "No references, no citations, no introductory filler. Output ONLY the clean answer text.\n\n"
        f"Query: {query}\n"
        f"Raw Response: {raw_response}"
    )
    try:
        completion = client.chat.completions.create(
            model=settings.GROQ_TEXT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        return completion.choices[0].message.content.strip()
    except Exception:
        return raw_response.replace("🌐 **[Real-time Web Search Fallback Agent | Querying Live Databases]**\n\n", "")


def clean_and_format_hybrid(client, query: str, matched_phrase: str, local_response: str, web_response: str) -> str:
    """Helper to format local + global responses into the exact required Hybrid format."""
    web_response_clean = web_response.replace("🌐 **[Real-time Web Search Fallback Agent | Querying Live Databases]**\n\n", "")
    
    if not client:
        return (
            "⚠️ Scope Boundary Notice\n\n"
            f"The phrase \"{matched_phrase}\" is outside the placement dataset scope.\n\n"
            f"📋 Dataset Answer:\n{local_response}\n\n"
            f"🌐 Global Context:\n{web_response_clean}\n\n"
            "This external information is separate from the placement dataset."
        )
    
    prompt = f"""
You are a senior software engineer formatting a RAG response for a hackathon demo.
Format the final answer using the EXACT structure below. Keep it short, clean, and concise. Do not use long paragraphs or citations.

REQUIRED STRUCTURE:
⚠️ Scope Boundary Notice

The phrase "{matched_phrase}" is outside the placement dataset scope.

📋 Dataset Answer:
[Provide an extremely short 1-line summary of the local response, e.g. "Infosys → 42.9 LPA"]

🌐 Global Context:
[Provide a short 1-line real-world context summarizing the web response, e.g. "Nvidia is considered among the highest-paying companies globally."]

This external information is separate from the placement dataset.

INPUTS:
Query: {query}
Local Response: {local_response}
Web Response: {web_response_clean}
"""
    try:
        completion = client.chat.completions.create(
            model=settings.GROQ_TEXT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        return completion.choices[0].message.content.strip()
    except Exception:
        return (
            "⚠️ Scope Boundary Notice\n\n"
            f"The phrase \"{matched_phrase}\" is outside the placement dataset scope.\n\n"
            f"📋 Dataset Answer:\n{local_response}\n\n"
            f"🌐 Global Context:\n{web_response_clean}\n\n"
            "This external information is separate from the placement dataset."
        )


@router.post("/query", response_model=QueryResponse)
async def query_system(request: QueryRequest):
    """
    Main query gateway.
    Routes incoming questions to the optimal agent and returns synthesized trace-backed answers.
    """
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # A. Check Scope Categorization
    query_lower = query.lower()
    
    # Define external keywords (longest to shortest for matching priority)
    external_keywords = ["in the world", "in india", "across india", "outside dataset", "world", "globally", "india", "worldwide", "global", "external"]
    matched_external = None
    for kw in external_keywords:
        if kw in query_lower:
            matched_external = kw
            break
            
    # Define local keywords
    local_keywords = ["this dataset", "dataset", "placement dataset", "local dataset", "our dataset", "in this dataset"]
    has_local_kw = any(lk in query_lower for lk in local_keywords)
    
    # Categorize scope
    is_hybrid = matched_external is not None and has_local_kw
    is_global_only = matched_external is not None and not has_local_kw

    # 1. Route the query using our central router agent
    if request.force_web_search or is_global_only:
        target_agent = "web_search_agent"
        reason = "User explicitly forced web search mode." if request.force_web_search else f"Query is global-only scope (matched external phrase '{matched_external}')."
        company = None
    else:
        route_details = router_agent.route_query(query)
        target_agent = route_details["agent"]
        reason = route_details["reason"]
        company = route_details["entities"].get("company")

    # 2. Invoke the target specialized agent
    response_text = ""
    try:
        if target_agent == "dataframe_agent":
            response_text = dataframe_agent.process_query(query)
        elif target_agent == "multi_hop_agent":
            response_text = multi_hop_agent.process_query(query)
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

    # 3. Post-Process based on Scope Boundary Handling
    if is_hybrid:
        try:
            # Query web agent for global perspective
            web_response = web_search_agent.process_query(query)
            response_text = clean_and_format_hybrid(
                client=router_agent.client,
                query=query,
                matched_phrase=matched_external,
                local_response=response_text,
                web_response=web_response
            )
        except Exception:
            response_text = (
                "⚠️ Scope Boundary Notice\n\n"
                f"The phrase \"{matched_external}\" is outside the placement dataset scope.\n\n"
                f"📋 Dataset Answer:\n{response_text}\n\n"
                "This external information is separate from the placement dataset."
            )
    elif is_global_only and target_agent == "web_search_agent":
        response_text = clean_and_summarize_global(
            client=router_agent.client,
            query=query,
            raw_response=response_text
        )

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


@router.post("/index")
async def trigger_indexing(background_tasks: BackgroundTasks):
    """
    Triggers the full ingestion, parsing, chunking, and database indexing pipeline
    in the background.
    """
    try:
        background_tasks.add_task(run_indexing_background, None)
        return {
            "status": "success",
            "message": "Re-indexing pipeline started in the background."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start indexing: {str(e)}")

