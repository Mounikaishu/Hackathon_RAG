from groq import Groq
from app.config import settings
from app.embeddings.vector_store import VectorStoreManager

class RagAgent:
    """
    Standard Semantic RAG Agent.
    Retrieves relevant text chunks from ChromaDB, filters by metadata (such as targeted companies),
    and synthesizes responses grounded securely on corpus context.
    """

    def __init__(self):
        self.api_key = settings.GROQ_API_KEY
        self.client = None
        if self.api_key:
            self.client = Groq(api_key=self.api_key)
        self.db_manager = VectorStoreManager()

    def process_query(self, query: str, company_filter: str = None) -> str:
        """
        Retrieves matching semantic context and answers the user's question.
        Applies company and section filters to reduce indexing cross-talk.
        """
        query_lower = query.lower()
        interview_keywords = [
            "round",
            "rounds",
            "interview",
            "experience",
            "technical",
            "hr",
            "oa",
            "online assessment",
            "coding round"
        ]
        
        is_interview_query = any(word in query_lower for word in interview_keywords)

        # 1. Retrieve semantic context chunks from database
        if is_interview_query:
            print("🔍 [RAG Pipeline] Searching interview chunks...")
            # Query top 10 matches without initial hard filters to allow manual matching
            raw_results = self.db_manager.query(
                query_text=query,
                n_results=10
            )
            
            results = []
            for r in raw_results:
                text_clean = r["text"].lower()
                meta = r["metadata"]
                section = meta.get("section", "").lower()
                
                # Debug print metadata as requested
                print(f"DEBUG Chunk Metadata: {meta}")
                
                if (
                    "interview" in section
                    or "experience" in text_clean
                    or "round" in text_clean
                    or "technical round" in text_clean
                    or "hr round" in text_clean
                    or "oa" in text_clean
                    or "online assessment" in text_clean
                ):
                    results.append(r)
            results = results[:5]  # Limit to top 5 matches
        else:
            results = self.db_manager.query(
                query_text=query,
                n_results=5,
                company_filter=company_filter
            )

        if not results:
            return (
                "🔍 Semantic Search: No matching interview or placement experiences were found in the database. "
                "Please verify that indexing has run successfully."
            )

        # 2. Extract context passages and track citations
        context_blocks = []
        citations = []
        
        for idx, r in enumerate(results):
            text = r["text"]
            meta = r["metadata"]
            sec = meta.get("section", "general")
            comp = meta.get("company", "Unknown")
            
            # Format text block
            context_blocks.append(f"[Source {idx+1} | Company: {comp} | Section: {sec}]\n{text}")
            
            # Add to citations tracker
            citations.append(f"- **Source {idx+1}**: {comp} placement text (Section: `{sec}`, similarity: {round(r['similarity'] * 100, 1)}%)")

        context_string = "\n\n".join(context_blocks)

        if not self.client:
            # Local fallback: output the retrieved chunks directly
            fallback_res = [
                "⚠️ Groq API Error: GROQ_API_KEY is missing. Showing raw retrieved database matches instead:\n",
                context_string
            ]
            return "\n\n".join(fallback_res)

        # 3. Formulate prompt for Llama 3.3 text model
        system_prompt = (
            "You are a professional SVECW Career Placement Assistant.\n"
            "Your task is to answer the user's question using ONLY the provided placement database context.\n"
            "CRITICAL RULES:\n"
            "1. Base your answer strictly on the supplied Context blocks. Do not assume or extrapolate beyond this data.\n"
            "2. If the context does not contain the answer, state that the information is not present in the document.\n"
            "3. Reference your sources inline using brackets (e.g. [Source 1], [Source 2]).\n"
            "4. Maintain a highly professional, supportive, and formal tone."
        )

        user_content = (
            f"Context Blocks:\n{context_string}\n\n"
            f"User Question: {query}"
        )

        try:
            chat_completion = self.client.chat.completions.create(
                model=settings.GROQ_TEXT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.3
            )
            
            answer = chat_completion.choices[0].message.content.strip()
            
            # Append trace-back citations cleanly at the bottom
            citation_footer = "\n\n---\n### 📄 Database Source Citations:\n" + "\n".join(citations)
            return answer + citation_footer

        except Exception as e:
            return f"❌ RAG Agent Completion Error: {str(e)}\n\nRaw Context Match:\n{context_string}"
