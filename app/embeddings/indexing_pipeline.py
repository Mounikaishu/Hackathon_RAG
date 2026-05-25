import os
import shutil
from app.config import settings
from app.ingestion.pdf_loader import PDFLoader
from app.ingestion.table_extractor import TableExtractor
from app.ingestion.chart_extractor import ChartExtractor
from app.chunking.eligibility_chunker import EligibilityChunker
from app.chunking.interview_chunker import InterviewChunker
from app.chunking.trend_chunker import TrendChunker
from app.embeddings.vector_store import VectorStoreManager

class IndexingPipeline:
    """
    Ties together the entire ingestion, parsing, chunking, and database indexing process.
    Provides a modular, standalone runner to seed the Vector DB with placement intelligence.
    """

    def __init__(self, pdf_path: str = None):
        # Default fallback to the user's Downloads directory where we discovered the file
        default_pdf = os.path.join("c:\\", "Users", "MOUNIKA", "Downloads", "Placement_RAG_Dataset_Enhanced.pdf")
        self.pdf_path = pdf_path or default_pdf
        
        # Verify dataset exists
        if not os.path.exists(self.pdf_path):
            # Check if copied to raw data folder
            raw_path = os.path.join(settings.RAW_DATA_DIR, "Placement_RAG_Dataset_Enhanced.pdf")
            if os.path.exists(raw_path):
                self.pdf_path = raw_path
            else:
                raise FileNotFoundError(
                    f"Dataset PDF not found at {self.pdf_path} or {raw_path}. "
                    "Please upload the file to your Downloads folder first."
                )

    def run(self):
        """Executes the full indexing workflow."""
        print("==================================================")
        print("🚀 STARTING PLACEMENT INTELLIGENCE INDEXING PIPELINE")
        print("==================================================")
        print(f"📄 Processing PDF: {self.pdf_path}")

        # Ensure raw folder exists and copy the file as a best-practice backup
        settings.ensure_directories()
        backup_raw_path = os.path.join(settings.RAW_DATA_DIR, "Placement_RAG_Dataset_Enhanced.pdf")
        if self.pdf_path != backup_raw_path:
            try:
                shutil.copy2(self.pdf_path, backup_raw_path)
                print(f"💾 Copied dataset backup to: {backup_raw_path}")
            except Exception as e:
                print(f"⚠️ Backup copy warning: {e}")

        # 1. Initialize Ingestors & Extractors
        print("\n🔍 Step 1: Running extraction engines...")
        pdf_loader = PDFLoader(self.pdf_path)
        table_extractor = TableExtractor(self.pdf_path)
        chart_extractor = ChartExtractor(self.pdf_path)

        # Parse text structure
        doc_structure = pdf_loader.load_raw_data()
        print(f"✅ Extracted raw text across {doc_structure['total_pages']} pages.")

        # Extract eligibility tables
        eligibility_records = table_extractor.extract_eligibility_table()
        print(f"✅ Extracted {len(eligibility_records)} company structured eligibility records.")

        # Extract charts as high-resolution PNGs
        chart_assets = chart_extractor.extract_chart_pages()
        print(f"✅ Extracted and rendered {len(chart_assets['saved_images'])} chart visual pages.")
        for img in chart_assets["saved_images"]:
            print(f"   🖼️ Saved chart image: {os.path.basename(img)}")

        # 2. Chunking & Structuring
        print("\n🧩 Step 2: Creating smart semantic chunks...")
        all_chunks = []

        # A. Chunk Eligibility Records
        eligibility_chunker = EligibilityChunker()
        elig_chunks = eligibility_chunker.chunk_eligibility_records(eligibility_records)
        all_chunks.extend(elig_chunks)
        print(f"   🔹 Constructed {len(elig_chunks)} semantic eligibility chunks.")

        # B. Chunk Interview Experiences (Section 2)
        # Find interview section in PDF loader output keys
        interview_text = ""
        for key, sec in doc_structure["sections"].items():
            if "interview" in key or "section_2" in key:
                interview_text = sec["content"]
                break
        
        if interview_text:
            interview_chunker = InterviewChunker()
            int_chunks = interview_chunker.chunk_interview_experiences(interview_text)
            all_chunks.extend(int_chunks)
            print(f"   🔹 Constructed {len(int_chunks)} deduplicated interview experience chunks.")
        else:
            print("   ⚠️ Warning: Section 2 (Interview Experiences) text block not found.")

        # C. Chunk Temporal Trends & General Sections (Section 5 / 7 etc.)
        trend_chunker = TrendChunker()
        for key, sec in doc_structure["sections"].items():
            # Process trend or general sections
            if "trend" in key or "statistics" in key or "reasoning" in key:
                trend_chunks = trend_chunker.chunk_trend_data(sec["content"])
                all_chunks.extend(trend_chunks)
                print(f"   🔹 Constructed {len(trend_chunks)} chronological trend chunks from {sec['title']}.")

        # 3. Vector Database Indexing
        print("\n🗄️ Step 3: Indexing vectors in local persistent database...")
        db_manager = VectorStoreManager()
        
        # Reset the database to guarantee clean indexing
        print("   🧹 Re-initializing database collection...")
        db_manager.reset_db()
        
        # Add all chunks
        print(f"   📥 Writing {len(all_chunks)} chunks to ChromaDB...")
        db_manager.add_chunks(all_chunks)
        
        print("\n==================================================")
        print("🎉 PIPELINE SUCCESSFULLY COMPLETED!")
        print(f"✅ Total indexed database chunks: {len(all_chunks)}")
        print("==================================================")
        return len(all_chunks)

if __name__ == "__main__":
    try:
        pipeline = IndexingPipeline()
        pipeline.run()
    except Exception as e:
        print(f"\n❌ Pipeline execution failed: {e}")
