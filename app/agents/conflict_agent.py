from groq import Groq
from app.config import settings
from app.tools.pandas_tool import PandasTool
from app.embeddings.vector_store import VectorStoreManager

class ConflictAgent:
    """
    Inconsistency & Contradiction Detection Agent.
    Cross-checks structured database records against narrative vector chunks,
    identifying conflicting facts (e.g., Amazon CGPA cutoff 6.4 vs 7.0 in tips).
    """

    def __init__(self):
        self.api_key = settings.GROQ_API_KEY
        self.client = None
        if self.api_key:
            self.client = Groq(api_key=self.api_key)
        self.pandas_tool = PandasTool()
        self.db_manager = VectorStoreManager()

    def process_query(self, query: str, company: str = None) -> str:
        """
        Retrieves company data from both Pandas (Structured Table) and ChromaDB (Text)
        and passes them to Llama 3.3 to analyze, detect, and resolve contradictions.
        """
        # Deduce target company if not explicitly passed
        target_company = company
        if not target_company:
            # Simple fallback check
            for comp in ["Amazon", "TCS", "Infosys", "Deloitte", "Google", "Microsoft"]:
                if comp.lower() in query.lower():
                    target_company = comp
                    break
        
        # If no company detected, default to 'Amazon' (since it contains the primary cutoff conflict in the dataset)
        if not target_company:
            target_company = "Amazon"

        # 1. Retrieve Structured Table Facts (Pandas)
        df = self.pandas_tool.df
        table_facts = "No structured record found."
        if not df.empty:
            comp_df = df[df.company.str.lower() == target_company.lower()]
            if not comp_df.empty:
                table_facts = comp_df.to_markdown(index=False)

        # 2. Retrieve Unstructured Text Chunks (ChromaDB)
        text_results = self.db_manager.query(
            query_text=f"{target_company} requirements cutoff experience tip",
            n_results=4,
            company_filter=target_company
        )
        
        text_facts = "No narrative records found."
        if text_results:
            text_facts = "\n\n".join([f"Source {i+1}:\n{r['text']}" for i, r in enumerate(text_results)])

        if not self.client:
            # Fallback local output showing both datasets side-by-side
            return (
                "⚠️ Groq API Error: GROQ_API_KEY is missing. Showing raw comparison side-by-side:\n\n"
                f"📊 **Structured Eligibility Database Claim ({target_company}):**\n{table_facts}\n\n"
                f"📝 **Unstructured Interview Text Database Claim ({target_company}):**\n{text_facts}"
            )

        # 3. Formulate conflict analyzer system prompt
        system_prompt = (
            "You are an Inconsistency Detector and Fact-Checking Agent.\n"
            f"You are investigating contradictory requirements for the company: '{target_company}'.\n"
            "Your task is to carefully review both the Structured Table Data and the Unstructured Text Narratives, "
            "identify any conflicts, discrepancies, or conflicting cutoffs, and explain them clearly to the user.\n\n"
            "INSTRUCTIONS:\n"
            "1. State clearly if there is a conflict (e.g. 'CONFLICT DETECTED: Amazon CGPA cutoff is listed as 6.4 in the table but the interview tips text claims 7.0').\n"
            "2. Present both sides clearly under separate headers:\n"
            "   - '📊 Structured Database Profile'\n"
            "   - '📝 Unstructured Narrative Experience'\n"
            "3. Provide a helpful warning explaining how the user should prepare (e.g., preparing for the higher 7.0 standard as a safety precaution while the table threshold represents the formal registry).\n"
            "4. Keep your answer highly analytical, objective, and precise."
        )

        user_content = (
            f"📊 STRUCTURED DATABASE FACT (Pandas):\n{table_facts}\n\n"
            f"📝 UNSTRUCTURED NARRATIVE FACT (ChromaDB Vector):\n{text_facts}\n\n"
            f"User Question: {query}"
        )

        try:
            chat_completion = self.client.chat.completions.create(
                model=settings.GROQ_TEXT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.1
            )
            
            response = chat_completion.choices[0].message.content.strip()
            header = f"⚠️ **[Conflict Verification Agent | Active Check for Inconsistencies on {target_company}]**\n\n"
            return header + response

        except Exception as e:
            return f"❌ Conflict Agent Error: {str(e)}\n\nStructured facts:\n{table_facts}\n\nText facts:\n{text_facts}"
