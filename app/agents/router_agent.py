import json
import re
import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer
from app.config import settings

class RouterAgent:
    """
    The central brain of the system.
    Analyzes user queries and routes them to the correct specialized agent
    using Semantic Intent Routing via SentenceTransformers and fallback rule-matching.
    """

    def __init__(self, threshold: float = 0.35):
        self.api_key = settings.GROQ_API_KEY
        self.client = None
        if self.api_key:
            self.client = Groq(api_key=self.api_key)
            
        # Initialize SentenceTransformer local embedding model
        self.model = SentenceTransformer(settings.LOCAL_EMBEDDING_MODEL)
        self.threshold = threshold
        
        # Semantic agent profiles
        self.agents_descriptions = {
            "dataframe_agent": "structured tabular queries such as CGPA, package, salary, eligibility, bond period, backlogs, filtering, comparisons",
            "rag_agent": "interview experiences, placement rounds, preparation, company interview narratives",
            "vision_agent": "charts, graphs, hiring distributions, visual analysis, multimodal reasoning",
            "multi_hop_agent": "multi-step reasoning using multiple constraints like CGPA, backlog, package filtering",
            "conflict_agent": "contradictory values, mismatched cutoffs, official vs portal conflicts",
            "web_search_agent": "real-time information, stock prices, global knowledge, current external data"
        }
        
        self.agent_names = list(self.agents_descriptions.keys())
        # Precompute description embeddings
        self.agent_embeddings = self.model.encode(
            [self.agents_descriptions[name] for name in self.agent_names],
            convert_to_numpy=True
        )

    def _extract_entities_local(self, query: str) -> dict:
        """Helper to extract company, cgpa, and package from query using regex/keywords."""
        query_lower = query.lower()
        
        # 1. Company extraction
        matched_company = None
        companies = ["Amazon", "Google", "TCS", "Infosys", "Deloitte", "Microsoft", "Intel", "Qualcomm", "Cognizant", "Wipro", "Oracle", "IBM"]
        for comp in companies:
            if comp.lower() in query_lower:
                matched_company = comp
                break
                
        # 2. CGPA extraction
        cgpa_val = None
        cgpa_match = re.search(r"cgpa\s*(?:of|cutoff|requirement|<=|>=|<|>|=)?\s*(\d+\.\d+|\d+)", query_lower)
        if cgpa_match:
            try:
                cgpa_val = float(cgpa_match.group(1))
            except ValueError:
                pass
        else:
            # Try plain decimal number matching if it looks like a CGPA
            dec_match = re.findall(r"\b([56789]\.\d+)\b", query_lower)
            if dec_match:
                cgpa_val = float(dec_match[0])

        # 3. Package extraction
        package_val = None
        package_match = re.search(r"(?:package|salary|lpa|pay)\s*(?:of|cutoff|<=|>=|<|>|=)?\s*(\d+\.\d+|\d+)", query_lower)
        if package_match:
            try:
                package_val = float(package_match.group(1))
            except ValueError:
                pass
        else:
            # Check for generic number followed by lpa/package
            num_lpa_match = re.search(r"\b(\d+\.\d+|\d+)\s*(?:lpa|lakhs|package)\b", query_lower)
            if num_lpa_match:
                package_val = float(num_lpa_match.group(1))
                
        return {
            "company": matched_company,
            "cgpa": cgpa_val,
            "package": package_val
        }

    def route_query(self, query: str) -> dict:
        """
        Classifies the query and extracts entities.
        Uses Semantic Cosine Similarity scoring first, falling back to rule-based classification if low confidence.
        """
        query_lower = query.lower()

        # Compute query embedding
        q_emb = self.model.encode(query, convert_to_numpy=True)
        
        # Calculate cosine similarities: dot product of normalized vectors
        q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-9)
        agents_norms = self.agent_embeddings / (np.linalg.norm(self.agent_embeddings, axis=1, keepdims=True) + 1e-9)
        similarities = np.dot(agents_norms, q_norm)
        
        # Find highest similarity match
        best_idx = np.argmax(similarities)
        best_score = float(similarities[best_idx])
        best_agent = self.agent_names[best_idx]
        
        # Format debug trace logs
        scores_str = ", ".join(f"{name}: {similarities[idx]:.3f}" for idx, name in enumerate(self.agent_names))
        trace_reason = f"Semantic match ({best_agent}) with similarity {best_score:.3f}. Scores: [{scores_str}]."

        # Threshold Decision
        if best_score >= self.threshold:
            entities = self._extract_entities_local(query)
            return {
                "agent": best_agent,
                "entities": entities,
                "reason": trace_reason,
                "cleaned_query": query
            }
            
        # Fallback path if semantic classification is below threshold
        fallback_warning = f"Low semantic confidence ({best_score:.3f}). Scores: [{scores_str}]."
        return self._fallback_keyword_routing(query, warning=fallback_warning)

    def _fallback_keyword_routing(self, query: str, warning: str = None) -> dict:
        """Rule-based keyword-matching routing fallback containing all system constraints."""
        query_lower = query.lower()
        warning_suffix = f" {warning}" if warning else ""

        # 1. Multi-hop queries check
        if "cgpa" in query_lower and ("backlog" in query_lower or "backlogs" in query_lower) and "highest" in query_lower:
            return {
                "agent": "multi_hop_agent",
                "entities": self._extract_entities_local(query),
                "reason": f"Fallback: Multi-hop reasoning request detected.{warning_suffix}",
                "cleaned_query": query
            }

        # 2. Package/Salary queries check
        salary_keywords = [
            "highest package", "highest paying", "best package", "highest salary",
            "pays the highest", "top paying", "highest pay", "pays highest", "top salary"
        ]
        if any(k in query_lower for k in salary_keywords):
            return {
                "agent": "dataframe_agent",
                "entities": self._extract_entities_local(query),
                "reason": f"Fallback: Salary comparison query detected.{warning_suffix}",
                "cleaned_query": query
            }

        # 3. Vision / Chart queries check
        vision_keywords = ["chart", "graph", "bar", "hiring distribution", "role", "analyst", "intern", "officer", "sde"]
        if any(word in query_lower for word in vision_keywords):
            return {
                "agent": "vision_agent",
                "entities": self._extract_entities_local(query),
                "reason": f"Fallback: Visual chart query detected.{warning_suffix}",
                "cleaned_query": query
            }

        # 4. Temporal/trend queries check
        trend_keywords = ["increase", "trend", "growth", "2021", "2022", "2023", "2024", "package increase"]
        if any(word in query_lower for word in trend_keywords):
            return {
                "agent": "dataframe_agent",
                "entities": self._extract_entities_local(query),
                "reason": f"Fallback: Temporal/trend analysis detected.{warning_suffix}",
                "cleaned_query": query
            }

        # 5. Bonds check
        bond_keywords = ["bond", "bond period", "bond years"]
        if any(k in query_lower for k in bond_keywords):
            return {
                "agent": "dataframe_agent",
                "entities": self._extract_entities_local(query),
                "reason": f"Fallback: Bond query detected.{warning_suffix}",
                "cleaned_query": query
            }

        # 6. Backlogs check
        backlog_keywords = ["backlog", "backlogs", "allow backlogs", "arrears"]
        if any(k in query_lower for k in backlog_keywords):
            return {
                "agent": "dataframe_agent",
                "entities": self._extract_entities_local(query),
                "reason": f"Fallback: Backlog query detected.{warning_suffix}",
                "cleaned_query": query
            }

        # 7. Technology focus check
        tech_keywords = ["technology", "tech focus", "focus on", "interviews use", "stack", "language"]
        if any(k in query_lower for k in tech_keywords):
            return {
                "agent": "dataframe_agent",
                "entities": self._extract_entities_local(query),
                "reason": f"Fallback: Technology focus query detected.{warning_suffix}",
                "cleaned_query": query
            }

        # 8. Conflict agent check
        conflict_keywords = ["conflict", "discrepancy", "contradict", "difference", "contradiction", "inconsistency", "backlog discrepancy", "cgpa discrepancy"]
        if any(k in query_lower for k in conflict_keywords) or (("amazon" in query_lower or "google" in query_lower) and ("cgpa" in query_lower or "cutoff" in query_lower)):
            return {
                "agent": "conflict_agent",
                "entities": self._extract_entities_local(query),
                "reason": f"Fallback: Conflict discrepancy request detected.{warning_suffix}",
                "cleaned_query": query
            }

        # 9. Interview experience check
        interview_keywords = ["round", "rounds", "interview", "experience", "technical", "hr", "oa", "online assessment", "coding round"]
        if any(k in query_lower for k in interview_keywords):
            return {
                "agent": "rag_agent",
                "entities": self._extract_entities_local(query),
                "reason": f"Fallback: Interview experience query detected.{warning_suffix}",
                "cleaned_query": query
            }

        # 10. Web search/adversarial fallback
        search_keywords = ["stock", "price", "ceo", "current", "founder", "market capitalization", "news today"]
        if any(kw in query_lower for kw in search_keywords):
            return {
                "agent": "web_search_agent",
                "entities": self._extract_entities_local(query),
                "reason": f"Fallback: Web search/adversarial query detected.{warning_suffix}",
                "cleaned_query": query
            }

        # 10.5 General DataFrame / filtering cutoffs check
        df_keywords = ["cgpa", "package", "backlog", "bond", "lpa", "highest", "average", "cutoff", "filter", "sort", "<", ">", "="]
        if any(kw in query_lower for kw in df_keywords):
            return {
                "agent": "dataframe_agent",
                "entities": self._extract_entities_local(query),
                "reason": f"Fallback: Tabular/numerical keywords detected.{warning_suffix}",
                "cleaned_query": query
            }

        # 11. Final fallback: RAG agent
        return {
            "agent": "rag_agent",
            "entities": self._extract_entities_local(query),
            "reason": f"Fallback: Standard semantic text query.{warning_suffix}",
            "cleaned_query": query
        }

if __name__ == "__main__":
    # Standard testing routine
    router = RouterAgent()
    print(router.route_query("What is the current stock price of Google?"))
    print(router.route_query("Tell me about TCS interview experience"))
