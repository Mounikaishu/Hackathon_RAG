from groq import Groq
from app.config import settings
from app.tools.search_tool import SearchTool

class WebSearchAgent:
    """
    Real-time Web Search Fallback Agent.
    Handles out-of-corpus adversarial queries by executing web searches
    and compiling current facts into trace-backed answers.
    """

    def __init__(self):
        self.api_key = settings.GROQ_API_KEY
        self.client = None
        if self.api_key:
            self.client = Groq(api_key=self.api_key)
        self.search_tool = SearchTool()

    def process_query(self, query: str) -> str:
        """
        Executes a web search, aggregates search snippets, and generates
        a grounded answer with actual web reference citations.
        """
        print(f"🌐 [Web Pipeline] Querying search engines for: '{query}'")
        
        # 1. Run web search
        search_context = self.search_tool.search(query, max_results=3)

        if not self.client:
            # Fallback: return raw search results directly if no API key is set
            return (
                "⚠️ Groq API Error: GROQ_API_KEY is missing. Showing raw parsed search engine snippets instead:\n\n"
                f"{search_context}"
            )

        # 2. Formulate synthesis prompt for Llama 3.3
        system_prompt = (
            "You are a real-time Web Intelligence and Search Assistant.\n"
            "Your task is to answer the user's question using the provided real-time Web Search Results.\n\n"
            "CRITICAL RULES:\n"
            "1. Answer the question comprehensively using the web search snippets.\n"
            "2. Cite your sources using the bracketed indices (e.g. [1], [2]).\n"
            "3. List the URLs referenced clearly in a 'References' list at the bottom.\n"
            "4. If no information is found in the search results, state that you cannot verify the fact at this moment.\n"
            "5. Maintain an informative, accurate, and professional tone."
        )

        user_content = (
            f"🔍 WEB SEARCH RESULTS:\n{search_context}\n\n"
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
            header = "🌐 **[Real-time Web Search Fallback Agent | Querying Live Databases]**\n\n"
            return header + answer

        except Exception as e:
            return (
                f"❌ Web Agent Completion Error: {str(e)}\n\n"
                f"Raw search engine matches retrieved:\n{search_context}"
            )
