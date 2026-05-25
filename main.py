import os
import sys
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from app.config import settings
from app.api.routes import router, check_status
from app.embeddings.indexing_pipeline import IndexingPipeline

# Initialize FastAPI App
app = FastAPI(
    title="SVECW Placement Intelligence Multimodal RAG",
    description="Groq-Powered Multi-Agent Pipeline for Ingestion, Tabular analysis, Conflict Detection, and Multimodal Vision",
    version="1.0.0"
)

# Enable CORS for frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount processed directory to serve static assets (such as charts PNGs)
import os
processed_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "processed")
os.makedirs(processed_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=processed_dir), name="static")

# Serve Frontend HTML Dashboard
@app.get("/", response_class=HTMLResponse)
async def get_index():
    index_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app", "api", "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h2>SVECW Placement RAG Server is running.</h2><p>Frontend file index.html not found.</p>")

# Include API Router
app.include_router(router, prefix="/api")

# ANSI Color Codes for Premium Terminal Aesthetics
C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_CYAN = "\033[36m"
C_GREEN = "\033[32m"
C_YELLOW = "\033[33m"
C_MAGENTA = "\033[35m"
C_RED = "\033[31m"
C_BLUE = "\033[34m"

def print_welcome_banner():
    """Prints a beautiful colorful hackathon welcome banner to the terminal."""
    banner = f"""
{C_CYAN}{C_BOLD}======================================================================
🎓  SVECW PLACEMENT INTELLIGENCE MULTIMODAL AGENTIC RAG SYSTEM  🎓
======================================================================{C_RESET}
{C_GREEN}⚡ Powered by: Groq (Llama 3.3 Reasoning & Llama 3.2 Vision)
💾 Persistent DB: ChromaDB + Local Sentence-Transformers (Offline)
📊 Data modality: Structured JSON tables, Deduplicated text, PNG charts{C_RESET}
----------------------------------------------------------------------
Type your query and hit Enter. Type {C_YELLOW}/exit{C_RESET} to close.
Built-in Commands:
  {C_CYAN}/status{C_RESET} - View live vector database and structured table statistics
  {C_CYAN}/index{C_RESET}  - Re-run the structural PDF loader & indexing pipeline
----------------------------------------------------------------------
"""
    print(banner)

async def run_cli_session():
    """Starts the interactive CLI session."""
    print_welcome_banner()

    # Check database status
    status = await check_status()
    table_loaded = status["structured_database"]["loaded"]
    vector_loaded = status["vector_database"]["loaded"]

    # Auto-indexing prompt if empty
    if not table_loaded or not vector_loaded:
        print(f"{C_YELLOW}⚠️  Database status: EMPTY. Ingestion indices are not built yet.{C_RESET}")
        choice = input("👉 Would you like to run the Ingestion & Indexing Pipeline now? (y/n): ").strip().lower()
        if choice == 'y':
            try:
                pipeline = IndexingPipeline()
                pipeline.run()
                # Reload status
                status = await check_status()
            except Exception as e:
                print(f"{C_RED}❌ Auto-Indexing failed: {e}{C_RESET}")
        else:
            print(f"{C_YELLOW}⚠️ Proceeding with empty databases. Tabular and Vector queries will fail.{C_RESET}")

    # import route agents directly for CLI loops
    from app.api.routes import query_system, QueryRequest

    while True:
        try:
            print(f"\n{C_BOLD}👤 User Question:{C_RESET}", end=" ")
            user_input = input().strip()
            
            if not user_input:
                continue

            if user_input.lower() == "/exit":
                print(f"\n{C_GREEN}👋 Happy learning! Exiting Placement Intelligence RAG. Good luck in the Hackathon!{C_RESET}\n")
                break

            if user_input.lower() == "/status":
                s = await check_status()
                print(f"\n{C_BOLD}📊 Live Database Statistics:{C_RESET}")
                print(f"  • Structured Companies: {C_GREEN}{s['structured_database']['companies_count']}{C_RESET} profiles loaded")
                print(f"  • Semantic Database:  {C_GREEN}{s['vector_database']['chunks_count']}{C_RESET} text vector chunks loaded")
                print(f"  • Visual Charts:      {C_GREEN}{s['charts_gallery']['charts_count']}{C_RESET} chart images rendered")
                continue

            if user_input.lower() == "/index":
                print(f"\n🔄 Running Ingestion Pipeline...")
                try:
                    pipeline = IndexingPipeline()
                    pipeline.run()
                except Exception as e:
                    print(f"{C_RED}❌ Re-indexing failed: {e}{C_RESET}")
                continue

            # Query system through FastAPI gateway schema
            print(f"{C_MAGENTA}🧠 [Routing Engine] Analyzing query modality...{C_RESET}")
            req = QueryRequest(query=user_input)
            response = await query_system(req)

            # Map color based on routed agent
            agent = response.routed_agent
            agent_color = C_GREEN
            if "dataframe" in agent:
                agent_color = C_YELLOW
            elif "vision" in agent:
                agent_color = C_CYAN
            elif "conflict" in agent:
                agent_color = C_RED
            elif "web" in agent:
                agent_color = C_BLUE

            print(f"🎯 {C_MAGENTA}[Routing Engine] Routed to:{C_RESET} {agent_color}{C_BOLD}{agent}{C_RESET}")
            print(f"💡 {C_MAGENTA}[Routing Engine] Decision Reason:{C_RESET} {C_CYAN}{response.routing_reason}{C_RESET}\n")
            print(f"{C_BOLD}🤖 Agent Response:{C_RESET}\n{response.response}")
            print(f"\n{C_CYAN}----------------------------------------------------------------------{C_RESET}")

        except KeyboardInterrupt:
            print("\n")
            break
        except Exception as e:
            print(f"{C_RED}❌ CLI Query Error: {e}{C_RESET}")

if __name__ == "__main__":
    # If standard run, default to the beautiful Terminal CLI
    if len(sys.argv) == 1:
        import asyncio
        asyncio.run(run_cli_session())
    else:
        # If run with server argument, e.g. 'python main.py server'
        if sys.argv[1] == "server":
            print(f"{C_GREEN}🚀 Starting FastAPI Backend Server on http://{settings.HOST}:{settings.PORT} ...{C_RESET}")
            uvicorn.run("main:app", host=settings.HOST, port=settings.PORT, reload=True)
        else:
            print(f"Unknown arguments. Run 'python main.py' for interactive CLI or 'python main.py server' to start web server.")
