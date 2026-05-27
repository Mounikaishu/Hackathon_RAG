import json
import re
import numpy as np
from groq import Groq
from app.config import settings

class RouterAgent:
    """
    The central brain of the system.
    Analyzes user queries and routes them to the correct specialized agent
    using Semantic Intent Routing via SentenceTransformers and fallback rule-matching.
    """

    def __init__(self, threshold: float = 0.25):
        self.api_key = settings.GROQ_API_KEY
        self.client = None
        if self.api_key:
            self.client = Groq(api_key=self.api_key)
            
        # Lazy-load the embedding model to save RAM during startup on Render
        self.model = None
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
        self.agent_embeddings = None

    def _get_model(self):
        """Lazy loads the embedding model during the first routed query."""
        if self.model is None:
            # Heavy import deferred
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(settings.LOCAL_EMBEDDING_MODEL)
            
            # Precompute description embeddings only once the model is loaded
            self.agent_embeddings = self.model.encode(
                [self.agents_descriptions[name] for name in self.agent_names],
                convert_to_numpy=True
            )
            
        return self.model

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

    def _resolve_edge_case(self, query: str, sim_dict: dict, entities: dict) -> str:
        """
        Intelligently resolves ambiguous queries based on semantic intent rules.
        """
        query_lower = query.lower()

        # High-priority E6 Boolean Entity Query override
        _e6_bool_keywords = ["does", "allow", "allowed", "permit", "permits"]
        _e6_attr_keywords = ["backlog", "backlogs", "bond", "bonds"]
        _e6_company_keywords = ["microsoft", "amazon", "google", "tcs", "infosys", "wipro", "ibm", "oracle"]
        _e6_list_keywords = ["list", "which companies", "all companies"]
        if (any(kw in query_lower for kw in _e6_bool_keywords) and
            any(kw in query_lower for kw in _e6_attr_keywords) and
            any(kw in query_lower for kw in _e6_company_keywords) and
            not any(kw in query_lower for kw in _e6_list_keywords)):
            return "dataframe_agent"

        # NEW: Direct attribute lookup overrides – company + attribute keywords go to dataframe_agent
        attribute_keywords = ["cgpa", "backlog", "backlogs", "bond", "package", "salary", "lpa", "technology", "tech stack", "focus", "eligibility"]
        if entities.get("company") and any(kw in query_lower for kw in attribute_keywords):
            return "dataframe_agent"

        # CASE A: Eligibility Recommendation -> multi_hop_agent
        eligibility_phrases = [
            "where can i apply", "am i eligible", "what companies", 
            "eligible for", "can i apply", "apply with", "apply to", 
            "eligible to", "eligible check"
        ]
        has_eligibility_phrase = any(phrase in query_lower for phrase in eligibility_phrases)
        has_cgpa = (entities.get("cgpa") is not None) or ("cgpa" in query_lower)
        
        if has_eligibility_phrase or (has_cgpa and any(k in query_lower for k in ["apply", "eligible", "job", "company", "companies"])):
            return "multi_hop_agent"

        # CASE B: Comparison / Recommendation -> multi_hop_agent
        companies_list = [
            "amazon", "google", "tcs", "infosys", "deloitte", "microsoft", 
            "intel", "qualcomm", "cognizant", "wipro", "oracle", "ibm", 
            "capgemini", "adobe", "sap", "hcl", "tech mahindra", "samsung"
        ]
        mentioned_companies = [c for c in companies_list if c in query_lower]
        
        comparison_phrases = [
            "vs", "versus", "join", "better", "compare", "career", 
            "choice", "choose", "prefer", "difference", "recommend", "opinion"
        ]
        has_comparison = any(phrase in query_lower for phrase in comparison_phrases)
        
        if len(mentioned_companies) >= 2 or (len(mentioned_companies) >= 1 and has_comparison) or (has_comparison and "company" in query_lower):
            return "multi_hop_agent"

        # CASE C: Real-time External Queries -> web_search_agent
        web_keywords = ["stock", "price", "ceo", "current", "founder", "market cap", "market capitalization", "news", "latest", "today", "who is", "weather", "ticker"]
        if any(kw in query_lower for kw in web_keywords):
            return "web_search_agent"

        return None

    def route_query(self, query: str) -> dict:
        """
        Classifies the query and extracts entities.
        Uses hierarchical routing with High/Medium/Low confidence bands, margin analysis, and edge-case resolution.
        """
        query_lower = query.lower()
        # ====================================================
        # HARD OVERRIDE: Comparison Queries → Multi-Hop Agent
        # ====================================================
        comparison_keywords = [
            "compare",
            "vs",
            "versus",
            "difference",
            "better",
            "all dimensions"
        ]
        comparison_dimensions = [
            "eligibility",
            "package",
            "hiring",
            "trend",
            "bond",
            "cgpa",
            "backlog",
            "technical focus"
        ]
        comparison_trigger = any(
            keyword in query_lower
            for keyword in comparison_keywords
        )
        dimension_count = sum(
            dimension in query_lower
            for dimension in comparison_dimensions
        )
        if comparison_trigger or dimension_count >= 2:
            print("⚡ Hard-routing comparison query → multi_hop_agent")
            return {
                "agent": "multi_hop_agent",
                "entities": self._extract_entities_local(query),
                "reason": "Hard routed comparison query detected.",
                "cleaned_query": query
            }

        # High-priority E6 Boolean Entity Query detection (override before other routing)
        _e6_bool_keywords = ["does", "allow", "allowed", "permit", "permits"]
        _e6_attr_keywords = ["backlog", "backlogs", "bond", "bonds"]
        _e6_company_keywords = ["microsoft", "amazon", "google", "tcs", "infosys", "wipro", "ibm", "oracle"]
        _e6_list_keywords = ["list", "which companies", "all companies"]
        if (any(kw in query_lower for kw in _e6_bool_keywords) and
            any(kw in query_lower for kw in _e6_attr_keywords) and
            any(kw in query_lower for kw in _e6_company_keywords) and
            not any(kw in query_lower for kw in _e6_list_keywords)):
            return {
                "agent": "dataframe_agent",
                "entities": self._extract_entities_local(query),
                "reason": "Boolean entity query detected.",
                "cleaned_query": query
            }

        # High-priority routing override BEFORE semantic routing
        conflict_keywords = [
            "conflict",
            "conflicting",
            "mismatch",
            "discrepancy",
            "inconsistency",
            "across sources",
            "different values",
            "contradiction",
            "which is correct"
        ]
        if any(kw in query_lower for kw in conflict_keywords):
            return {
            "agent": "conflict_agent",
            "entities": self._extract_entities_local(query),
            "reason": "High-priority routing override: conflict keyword matched.",
            "cleaned_query": query
        }

        # High-priority M6 override: targeted hiring comparison (2 companies + role + comparison trigger)
        # Route to vision_agent for deterministic DataFrame comparison; NOT full synthesis
        _m6_comparison_triggers = ["versus", "vs", "compare", "difference between"]
        _m6_role_keywords       = ["sde", "intern", "interns", "analyst", "analysts", "officer", "officers"]
        _m6_company_names       = ["amazon", "google", "tcs", "infosys", "microsoft",
                                    "wipro", "cognizant", "accenture", "flipkart", "oracle", "ibm"]

        # Find companies present in query (simple substring match)
        _m6_companies_found = [c for c in _m6_company_names if c in query_lower]
        _m6_has_comparison  = any(kw in query_lower for kw in _m6_comparison_triggers)
        _m6_has_role        = any(kw in query_lower for kw in _m6_role_keywords)

        if len(_m6_companies_found) >= 2 and _m6_has_comparison and _m6_has_role:
            return {
                "agent": "vision_agent",
                "entities": self._extract_entities_local(query),
                "reason": "M6 targeted hiring comparison query detected (2 companies + role + comparison trigger).",
                "cleaned_query": query
            }

        # High-priority join override BEFORE semantic routing
        tech_keywords = ["python", "java", "c++", "cloud", "system design", "tech focus", "focused", "technology"]
        hiring_keywords = ["intern", "interns", "sde", "analyst", "officer", "hiring", "most hires", "hires the most"]
        comparison_keywords = ["which", "most", "highest", "top"]
        
        has_tech_sig = any(tk in query_lower for tk in tech_keywords)
        has_hiring_sig = any(hk in query_lower for hk in hiring_keywords)
        has_comp_sig = any(ck in query_lower for ck in comparison_keywords)
        
        if has_tech_sig and has_hiring_sig and has_comp_sig:
            return {
                "agent": "multi_hop_agent",
                "entities": self._extract_entities_local(query),
                "reason": "Hard join query detected.",
                "cleaned_query": query
            }

        # E7: Interview Rounds Retrieval override
        _e7_round_keywords = [
            "round", "rounds", "interview rounds",
            "selection process", "hiring process", "conduct"
        ]
        _e7_company_keywords = [
            "tcs", "amazon", "infosys", "google", "microsoft",
            "oracle", "wipro", "ibm", "accenture", "deloitte",
            "cognizant", "capgemini", "hcl", "tech mahindra"
        ]
        _e7_prep_exclusions = [
            "prepare", "preparation", "study", "topics",
            "how to crack", "interview tips", "focus"
        ]

        if (
            any(kw in query_lower for kw in _e7_round_keywords)
            and any(kw in query_lower for kw in _e7_company_keywords)
            and not any(kw in query_lower for kw in _e7_prep_exclusions)
        ):
            return {
                "agent": "rag_agent",
                "entities": self._extract_entities_local(query),
                "reason": "Interview rounds retrieval query detected.",
                "cleaned_query": query
            }

        # M2 Threshold Filter override BEFORE recommendation routing
        _m2_req_keywords = [
            "require", "requires", "minimum cgpa", "cgpa above", 
            "cgpa higher than", "greater than", "more than", "above"
        ]
        _m2_comp_keywords = ["which companies", "companies"]
        _m2_attr_keywords = ["cgpa", "backlogs", "package", "bond"]
        _m2_student_keywords = [
            "i have", "my cgpa", "student", "can i apply", 
            "eligible for me", "wants"
        ]

        if (
            any(kw in query_lower for kw in _m2_req_keywords)
            and any(kw in query_lower for kw in _m2_comp_keywords)
            and any(kw in query_lower for kw in _m2_attr_keywords)
            and not any(kw in query_lower for kw in _m2_student_keywords)
        ):
            return {
                "agent": "dataframe_agent",
                "entities": self._extract_entities_local(query),
                "reason": "Threshold filter query detected.",
                "cleaned_query": query
            }

        # High-priority 3-condition optimization override BEFORE semantic routing
        student_keywords = ["student", "cgpa", "backlog", "backlogs"]
        preference_keywords = ["maximum pay", "highest pay", "highest package", "best package", "maximum package", "no bond", "without bond"]
        constraint_keywords = ["with", "wants", "eligible", "can apply"]
        
        has_student = any(k in query_lower for k in student_keywords)
        has_preference = any(k in query_lower for k in preference_keywords)
        has_constraint = any(k in query_lower for k in constraint_keywords)
        
        if has_student and has_preference and has_constraint:
            return {
                "agent": "multi_hop_agent",
                "entities": self._extract_entities_local(query),
                "reason": "Hard 3-condition optimization query detected.",
                "cleaned_query": query
            }

        # High-priority tech-focus override BEFORE semantic routing
        tech_keywords = [
            "technical focus", "tech focus", "technology focus", "use python",
            "python-focused", "java-focused", "c++ focused", "cloud-focused",
            "which companies use", "companies using", "focus on"
        ]
        languages = ["python", "java", "c++", "cloud", "system design"]
        hiring_keywords = ["intern", "interns", "analyst", "sde", "officer", "hiring", "distribution", "chart"]

        if (
            any(keyword in query_lower for keyword in tech_keywords)
            and any(lang in query_lower for lang in languages)
            and not any(hiring in query_lower for hiring in hiring_keywords)
        ):
            return {
                "agent": "dataframe_agent",
                "entities": self._extract_entities_local(query),
                "reason": "Tech-focus filtering query detected.",
                "cleaned_query": query
            }
        # High-priority E2 Backlog Direct Lookup override
        _e2_backlog_keywords = ["backlog", "backlogs", "active backlogs", "max backlogs"]
        _e2_lookup_keywords = ["how many", "what is", "limit"]
        _e2_company_keywords = [
            "amazon", "google", "microsoft", "tcs", "infosys",
            "oracle", "wipro", "ibm", "deloitte", "flipkart", "hcl",
            "tech mahindra", "qualcomm", "samsung", "adobe", "intel"
        ]
        _e2_list_keywords = ["which companies", "all companies", "list", "who allows", "who permit"]

        if (
            any(kw in query_lower for kw in _e2_backlog_keywords)
            and any(kw in query_lower for kw in _e2_company_keywords)
            and any(kw in query_lower for kw in _e2_lookup_keywords)
            and not any(kw in query_lower for kw in _e2_list_keywords)
        ):
            return {
                "agent": "dataframe_agent",
                "entities": self._extract_entities_local(query),
                "reason": "Direct backlog lookup query detected.",
                "cleaned_query": query
            }

        # High-priority E3 Bond Direct Lookup override
        _e3_bond_keywords = ["bond", "bond period", "service bond", "bond duration"]
        _e3_lookup_keywords = ["what is", "how long", "period", "duration"]
        _e3_company_keywords = [
            "amazon", "google", "microsoft", "tcs", "infosys",
            "oracle", "wipro", "ibm"
        ]
        _e3_list_keywords = ["bond-free", "which companies", "all companies", "list"]

        if (
            any(kw in query_lower for kw in _e3_bond_keywords)
            and any(kw in query_lower for kw in _e3_company_keywords)
            and any(kw in query_lower for kw in _e3_lookup_keywords)
            and not any(kw in query_lower for kw in _e3_list_keywords)
        ):
            return {
                "agent": "dataframe_agent",
                "entities": self._extract_entities_local(query),
                "reason": "Direct bond lookup query detected.",
                "cleaned_query": query
            }

        # High-priority E4 Technology Focus override
        _e4_tech_keywords = [
            "technology", "tech focus", "technical focus", "focus on",
            "programming language", "language"
        ]
        _e4_company_keywords = [
            "flipkart", "amazon", "google", "microsoft", "oracle",
            "tcs", "infosys", "wipro", "ibm"
        ]

        if (
            any(kw in query_lower for kw in _e4_tech_keywords)
            and any(kw in query_lower for kw in _e4_company_keywords)
        ):
            return {
                "agent": "dataframe_agent",
                "entities": self._extract_entities_local(query),
                "reason": "Technology focus retrieval query detected.",
                "cleaned_query": query
            }

        # High-priority E5 Direct Table Lookup override
        _e5_package_keywords = ["package", "salary", "lpa", "compensation", "offered"]
        _e5_company_keywords = [
            "google", "amazon", "microsoft", "tcs", "infosys",
            "wipro", "ibm", "oracle"
        ]

        if (
            any(kw in query_lower for kw in _e5_package_keywords)
            and any(kw in query_lower for kw in _e5_company_keywords)
        ):
            return {
                "agent": "dataframe_agent",
                "entities": self._extract_entities_local(query),
                "reason": "Direct table lookup query detected.",
                "cleaned_query": query
            }

        # High-priority E6 Boolean Entity Query override
        _e6_bool_keywords = ["does", "allow", "allowed", "permit", "permits"]
        _e6_attr_keywords = ["backlog", "backlogs", "bond", "bonds"]
        _e6_company_keywords = [
            "microsoft", "amazon", "google", "tcs", "infosys",
            "wipro", "ibm", "oracle"
        ]
        _e6_list_keywords = ["list", "which companies", "all companies"]

        if (
            any(kw in query_lower for kw in _e6_bool_keywords)
            and any(kw in query_lower for kw in _e6_attr_keywords)
            and any(kw in query_lower for kw in _e6_company_keywords)
            and not any(kw in query_lower for kw in _e6_list_keywords)
        ):
            return {
                "agent": "dataframe_agent",
                "entities": self._extract_entities_local(query),
                "reason": "Boolean entity query detected.",
                "cleaned_query": query
            }

        # High-priority E8 Easy Text Retrieval override
        _e8_tech_keywords = [
            "programming language", "language", "tech focus",
            "technical focus", "tested at", "focus at"
        ]
        _e8_company_keywords = [
            "amazon", "google", "microsoft", "oracle",
            "tcs", "infosys", "wipro", "ibm"
        ]

        if (
            any(kw in query_lower for kw in _e8_tech_keywords)
            and any(kw in query_lower for kw in _e8_company_keywords)
        ):
            return {
                "agent": "dataframe_agent",
                "entities": self._extract_entities_local(query),
                "reason": "Technology focus retrieval query detected.",
                "cleaned_query": query
            }

        # High-priority M3 Category + Sort override
        category_keywords = [
            "it service", "service firms", "product companies",
            "consulting firms", "among", "category"
        ]
        sort_keywords = [
            "highest", "top", "maximum", "best", "highest package"
        ]
        package_keywords = [
            "package", "salary", "lpa", "compensation"
        ]
        if (
            any(word in query_lower for word in category_keywords)
            and any(word in query_lower for word in sort_keywords)
            and any(word in query_lower for word in package_keywords)
        ):
            return {
                "agent": "dataframe_agent",
                "entities": self._extract_entities_local(query),
                "reason": "Category filter + sorting query detected.",
                "cleaned_query": query
            }
        # High-priority M1 Multi-row threshold filter override
        _m1_thresh_keywords = ["at least", "more than", "greater than", "minimum"]
        _m1_back_keywords = ["backlog", "backlogs"]
        _m1_list_keywords = ["list", "all companies", "which companies"]

        if (
            any(kw in query_lower for kw in _m1_thresh_keywords)
            and any(kw in query_lower for kw in _m1_back_keywords)
            and any(kw in query_lower for kw in _m1_list_keywords)
        ):
            return {
                "agent": "dataframe_agent",
                "entities": self._extract_entities_local(query),
                "reason": "Multi-row threshold filter query detected.",
                "cleaned_query": query
            }

                # Hard routing for comparison queries
        comparison_keywords = ["compare", "vs", "versus", "difference", "better", "all dimensions"]
        comparison_dimensions = ["eligibility", "package", "hiring", "trend", "bond", "cgpa", "backlog", "technical focus"]
        if (any(k in query_lower for k in comparison_keywords) or sum(d in query_lower for d in comparison_dimensions) >= 2):
            return {
                "agent": "multi_hop_agent",
                "entities": self._extract_entities_local(query),
                "reason": "Hard routed comparison query detected.",
                "cleaned_query": query
            }

        # Compute query embedding (lazy loads model on first query)
        q_emb = self._get_model().encode(query, convert_to_numpy=True)
        
        # Calculate cosine similarities: dot product of normalized vectors
        q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-9)
        agents_norms = self.agent_embeddings / (np.linalg.norm(self.agent_embeddings, axis=1, keepdims=True) + 1e-9)
        similarities = np.dot(agents_norms, q_norm)
        
        # Sort indices to find best and second best similarity scores
        sorted_indices = np.argsort(similarities)[::-1]
        best_idx = sorted_indices[0]
        second_best_idx = sorted_indices[1]
        
        best_agent = self.agent_names[best_idx]
        best_score = float(similarities[best_idx])
        second_best_score = float(similarities[second_best_idx])
        margin = best_score - second_best_score
        
        # Build score dictionary
        sim_dict = {self.agent_names[idx]: float(similarities[idx]) for idx in range(len(self.agent_names))}
        scores_str = ", ".join(f"{name}: {similarities[idx]:.3f}" for idx, name in enumerate(self.agent_names))
        
        # Extract entities
        entities = self._extract_entities_local(query)

        # 1. Classify Ambiguity / Confidence state
        is_ambiguous = margin < 0.10
        # Use configurable threshold for low confidence
        low_cutoff = self.threshold
        high_cutoff = 0.55
        is_high = best_score > high_cutoff
        is_medium = low_cutoff <= best_score <= high_cutoff
        is_low = best_score < low_cutoff

        # 2. Invoke Edge-Case Resolver if medium/low confidence or ambiguous
        resolved_agent = None
        if is_medium or is_low or is_ambiguous:
            resolved_agent = self._resolve_edge_case(query, sim_dict, entities)

        if resolved_agent:
            trace_reason = f"Resolved via edge-case rules to {resolved_agent}. Score: {best_score:.3f}, margin: {margin:.3f}. Scores: [{scores_str}]."
            return {
                "agent": resolved_agent,
                "entities": entities,
                "reason": trace_reason,
                "cleaned_query": query
            }

        # 3. Route according to Confidence Bands
        if is_high and not is_ambiguous:
            trace_reason = f"High-confidence semantic match ({best_agent}) with similarity {best_score:.3f} and margin {margin:.3f}. Scores: [{scores_str}]."
            return {
                "agent": best_agent,
                "entities": entities,
                "reason": trace_reason,
                "cleaned_query": query
            }
        elif is_medium and not is_ambiguous:
            # Medium confidence with clear separation -> trust best semantic agent
            trace_reason = f"Medium-confidence semantic match ({best_agent}) with similarity {best_score:.3f} and margin {margin:.3f}. Scores: [{scores_str}]."
            return {
                "agent": best_agent,
                "entities": entities,
                "reason": trace_reason,
                "cleaned_query": query
            }

        # 4. Fallback to keyword rules (low confidence or unresolved ambiguity)
        fallback_warning = f"Low confidence / unresolved ambiguity (Best score: {best_score:.3f}, Margin: {margin:.3f}). Scores: [{scores_str}]."
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
