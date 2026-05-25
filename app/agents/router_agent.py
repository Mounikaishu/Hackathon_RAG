import json
import re
from groq import Groq
from app.config import settings

class RouterAgent:
    """
    The central brain of the system.
    Analyzes user queries and routes them to the correct specialized agent.
    Generates a structured routing JSON payload containing entities and routing decisions.
    """

    def __init__(self):
        self.api_key = settings.GROQ_API_KEY
        self.client = None
        if self.api_key:
            self.client = Groq(api_key=self.api_key)

    def route_query(self, query: str) -> dict:
        """
        Classifies the query and extracts entities.
        Returns a dict: {
            "agent": str,
            "entities": dict,
            "reason": str,
            "cleaned_query": str
        }
        """
        query_lower = query.lower()

        # 1. Fast path for Multi-hop queries
        if "cgpa" in query_lower and ("backlog" in query_lower or "backlogs" in query_lower) and "highest" in query_lower:
            return {
                "agent": "multi_hop_agent",
                "entities": {},
                "reason": "Multi-hop reasoning request detected.",
                "cleaned_query": query
            }

        # 2. Fast path for Vision / Chart queries
        vision_keywords = ["chart", "graph", "bar", "hiring distribution", "role", "analyst", "intern", "officer", "sde"]
        if any(word in query_lower for word in vision_keywords):
            return {
                "agent": "vision_agent",
                "entities": {},
                "reason": "Visual chart query detected.",
                "cleaned_query": query
            }

        # 3. Fast path for Temporal/trend queries
        trend_keywords = ["increase", "trend", "growth", "2021", "2022", "2023", "2024", "package increase"]
        if any(word in query_lower for word in trend_keywords):
            return {
                "agent": "dataframe_agent",
                "entities": {},
                "reason": "Temporal/trend analysis detected.",
                "cleaned_query": query
            }

        # 4. Fast path for Structured Database Lookups (Bonds, Backlogs, Tech Focus)
        bond_keywords = ["bond", "bond period", "bond years"]
        if any(k in query_lower for k in bond_keywords):
            return {
                "agent": "dataframe_agent",
                "entities": {},
                "reason": "Bond query is a structured tabular lookup.",
                "cleaned_query": query
            }

        backlog_keywords = ["backlog", "backlogs", "allow backlogs", "arrears"]
        if any(k in query_lower for k in backlog_keywords):
            return {
                "agent": "dataframe_agent",
                "entities": {},
                "reason": "Backlog query is a structured tabular lookup.",
                "cleaned_query": query
            }

        tech_keywords = ["technology", "tech focus", "focus on", "interviews use", "stack", "language"]
        if any(k in query_lower for k in tech_keywords):
            return {
                "agent": "dataframe_agent",
                "entities": {},
                "reason": "Technology focus is a structured tabular lookup.",
                "cleaned_query": query
            }

        # 5. Fast path for Conflict Agent
        conflict_keywords = ["conflict", "discrepancy", "contradict", "difference", "contradiction", "inconsistency"]
        if any(k in query_lower for k in conflict_keywords) or (("amazon" in query_lower or "google" in query_lower) and ("cgpa" in query_lower or "cutoff" in query_lower)):
            return {
                "agent": "conflict_agent",
                "entities": {},
                "reason": "Conflict discrepancy request detected.",
                "cleaned_query": query
            }

        # 6. Fast path for RAG Agent (Interview preparation)
        interview_keywords = ["round", "rounds", "interview", "experience", "technical", "hr", "oa", "online assessment", "coding round"]
        if any(k in query_lower for k in interview_keywords):
            return {
                "agent": "rag_agent",
                "entities": {},
                "reason": "Interview preparation or text query.",
                "cleaned_query": query
            }

        if not self.client:
            # Standalone fallback route if no API key is present
            return self._fallback_keyword_routing(query)

        system_prompt = (
            "You are the central Router Agent for a Placement Intelligence Multimodal System.\n"
            "Analyze the user's query and output a strictly valid JSON object matching the schema below.\n\n"
            "SCHEMA DECISION RULE:\n"
            "Choose EXACTLY one of these target agents:\n"
            "1. 'dataframe_agent': For structured queries involving filtering, sorting, or math comparisons of cutoffs, packages, backlogs, or bonds (e.g. 'companies requiring CGPA < 7.5', 'best package').\n"
            "2. 'vision_agent': For visual queries asking about charts, hiring role distributions, bar graphs, or analyst/SDE roles ratios (e.g. 'Which company hires the most analysts according to the chart?').\n"
            "3. 'conflict_agent': For queries addressing inconsistencies, discrepancies, conflicts, or asking for cutoffs/packages of companies with known contradictions like Amazon or Google (e.g. 'What is Amazon's CGPA cutoff?', 'What is Google's package?', 'Amazon cutoff discrepancy').\n"
            "4. 'web_search_agent': For out-of-corpus queries asking about real-time, external, or stock price information (e.g. 'current Infosys stock price', 'Infosys CEO').\n"
            "5. 'multi_hop_agent': For complex multi-document synthesis or multi-hop filtering queries that combine multiple criteria across tables (e.g. 'A student with CGPA 7.6 and 1 backlog wants the highest-paying job...').\n"
            "6. 'rag_agent': For semantic text-based searches regarding interview rounds, tips, preparation strategies, technical focus, or general narratives.\n\n"
            "OUTPUT FORMAT:\n"
            "Return ONLY a raw JSON block. Do not include any conversational introduction, explanation, or markdown fences.\n"
            "{\n"
            "  \"agent\": \"[agent_name]\",\n"
            "  \"extracted_entities\": {\n"
            "    \"company\": \"[company name mentioned or null]\",\n"
            "    \"cgpa\": [numeric CGPA mentioned or null],\n"
            "    \"package\": [numeric package in LPA mentioned or null]\n"
            "  },\n"
            "  \"reason\": \"[brief explanation of routing]\",\n"
            "  \"cleaned_query\": \"[refined search query]\"\n"
            "}"
        )

        try:
            chat_completion = self.client.chat.completions.create(
                model=settings.GROQ_TEXT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0.0,  # Highly deterministic
                response_format={"type": "json_object"}
            )
            
            response_text = chat_completion.choices[0].message.content.strip()
            parsed_json = json.loads(response_text)
            
            # Map the parsed JSON structure to standard return dict
            return {
                "agent": parsed_json.get("agent", "rag_agent"),
                "entities": parsed_json.get("extracted_entities", {}),
                "reason": parsed_json.get("reason", "Inference routing"),
                "cleaned_query": parsed_json.get("cleaned_query", query)
            }

        except Exception as e:
            # Fail gracefully to keyword classifier
            return self._fallback_keyword_routing(query, warning=f"Groq API Error: {str(e)}")

    def _fallback_keyword_routing(self, query: str, warning: str = None) -> dict:
        """Rule-based keyword-matching routing fallback."""
        query_lower = query.lower()
        
        # A. Check Multi-hop triggers (cgpa + backlog + highest)
        if "cgpa" in query_lower and ("backlog" in query_lower or "backlogs" in query_lower) and "highest" in query_lower:
            return {
                "agent": "multi_hop_agent",
                "entities": {},
                "reason": f"Fallback: Multi-hop reasoning request detected. {warning or ''}",
                "cleaned_query": query
            }

        # B. Check temporal growth / trend triggers (route to dataframe_agent)
        trend_keywords = ["increase", "trend", "2021", "2022", "2023", "2024", "growth", "rose", "grew"]
        if any(kw in query_lower for kw in trend_keywords) and any(kw in query_lower for kw in ["company", "pkg", "package", "increase", "grew", "highest", "difference"]):
            return {
                "agent": "dataframe_agent",
                "entities": {},
                "reason": f"Fallback: Temporal trend query detected. {warning or ''}",
                "cleaned_query": query
            }

        # 1. Check web search / adversarial triggers
        search_keywords = ["stock", "price", "ceo", "current", "founder", "market capitalization", "news today"]
        if any(kw in query_lower for kw in search_keywords):
            return {
                "agent": "web_search_agent",
                "entities": {},
                "reason": f"Fallback: Adversarial keywords detected. {warning or ''}",
                "cleaned_query": query
            }

        # 2. Check Vision / Chart triggers
        vision_keywords = ["chart", "graph", "bar", "hiring distribution", "role", "analyst", "intern", "officer", "sde"]
        if any(kw in query_lower for kw in vision_keywords):
            return {
                "agent": "vision_agent",
                "entities": {},
                "reason": f"Fallback: Visual chart keywords detected. {warning or ''}",
                "cleaned_query": query
            }

        # 3. Check conflict triggers
        conflict_keywords = [
            "conflict", "discrepancy", "contradict", "difference", 
            "backlog discrepancy", "cgpa discrepancy", "portal", "official", 
            "contradiction", "inconsistency", "amazon cgpa", "google package", 
            "amazon cutoff", "google cutoff"
        ]
        if any(kw in query_lower for kw in conflict_keywords):
            return {
                "agent": "conflict_agent",
                "entities": {},
                "reason": f"Fallback: Inconsistent data keywords detected. {warning or ''}",
                "cleaned_query": query
            }

        # 4. Check DataFrame / filtering cutoffs
        df_keywords = ["cgpa", "package", "backlog", "bond", "lpa", "highest", "average", "cutoff", "filter", "sort", "<", ">", "="]
        if any(kw in query_lower for kw in df_keywords):
            # Extract basic numbers if possible
            cgpa_match = re.search(r"cgpa\s*(?:of|<=|>=|<|>|=)?\s*(\d+\.\d+|\d+)", query_lower)
            package_match = re.search(r"(?:package|lpa)\s*(?:of|<=|>=|<|>|=)?\s*(\d+\.\d+|\d+)", query_lower)
            
            entities = {
                "company": None,
                "cgpa": float(cgpa_match.group(1)) if cgpa_match else None,
                "package": float(package_match.group(1)) if package_match else None
            }
            
            return {
                "agent": "dataframe_agent",
                "entities": entities,
                "reason": f"Fallback: Tabular/numerical keywords detected. {warning or ''}",
                "cleaned_query": query
            }

        # 5. Default to RAG
        return {
            "agent": "rag_agent",
            "entities": {},
            "reason": f"Fallback: Standard semantic text query. {warning or ''}",
            "cleaned_query": query
        }
    
if __name__ == "__main__":
    # Small developer visual validation test
    router = RouterAgent()
    print(router.route_query("What is the average package of companies requiring CGPA < 7.5?"))
    print(router.route_query("What is the current stock price of Google?"))
