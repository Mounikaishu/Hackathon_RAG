from groq import Groq
from app.config import settings
from app.tools.pandas_tool import PandasTool
import re
import json

class MultiHopAgent:
    """
    Multi-Hop Reasoning Agent.
    Resolves queries that require synthesizing facts across multiple tables
    (e.g., combining eligibility filters with hiring count statistics).
    Uses deterministic retrieval first, followed by strict LLM explanation generation.
    """

    def __init__(self):
        self.api_key = settings.GROQ_API_KEY
        self.client = None
        if self.api_key:
            self.client = Groq(api_key=self.api_key)
        self.pandas_tool = PandasTool()

    def process_query(self, query: str, df = None) -> str:
        """
        Processes multi-hop/multi-document synthesis placement queries.
        Retrieves database facts deterministically via pandas first, then summarizes/reasons using Groq.
        """
        query_lower = query.lower().strip()
        active_df = df if df is not None else self.pandas_tool.df

        # 1. Parsing & Entity Extraction
        companies_list = [
            "google", "microsoft", "amazon", "tcs", "infosys", "deloitte", "accenture", 
            "flipkart", "wipro", "cognizant", "capgemini", "ibm", "adobe", "oracle", 
            "sap", "hcl", "tech mahindra", "qualcomm", "intel", "samsung r&d", "samsung"
        ]
        
        # Identify unique mentioned companies in order
        mentioned_companies = []
        query_temp = query_lower
        
        # Check for Samsung R&D first to avoid substring matching conflicts
        if "samsung r&d" in query_temp or "samsung r&d;" in query_temp:
            mentioned_companies.append("Samsung R&D")
            query_temp = query_temp.replace("samsung r&d", "").replace("samsung r&d;", "")
            
        for comp in ["google", "microsoft", "amazon", "tcs", "infosys", "deloitte", "accenture", "flipkart", "wipro", "cognizant", "capgemini", "ibm", "adobe", "oracle", "sap", "hcl", "tech mahindra", "qualcomm", "intel", "samsung"]:
            if comp in query_temp:
                disp_name = {
                    "google": "Google", "microsoft": "Microsoft", "amazon": "Amazon", "tcs": "TCS",
                    "infosys": "Infosys", "deloitte": "Deloitte", "accenture": "Accenture", "flipkart": "Flipkart",
                    "wipro": "Wipro", "cognizant": "Cognizant", "capgemini": "Capgemini", "ibm": "IBM",
                    "adobe": "Adobe", "oracle": "Oracle", "sap": "SAP", "hcl": "HCL", "tech mahindra": "Tech Mahindra",
                    "qualcomm": "Qualcomm", "intel": "Intel", "samsung": "Samsung R&D"
                }[comp]
                if disp_name not in mentioned_companies:
                    mentioned_companies.append(disp_name)

        # CGPA value
        cgpa_val = None
        cgpa_match = re.search(r"cgpa\s*(?:of|cutoff|requirement|<=|>=|<|>|=)?\s*(\d+\.\d+|\d+)", query_lower)
        if cgpa_match:
            try:
                cgpa_val = float(cgpa_match.group(1))
            except ValueError:
                pass
        else:
            dec_matches = re.findall(r"\b([4-9]\.\d+|10\.0)\b", query_lower)
            if dec_matches:
                cgpa_val = float(dec_matches[0])

        # Backlogs count
        backlog_val = 0
        backlog_match = re.search(r"(\d+)\s*(?:backlog|backlogs|arrear|arrears)", query_lower)
        if backlog_match:
            try:
                backlog_val = int(backlog_match.group(1))
            except ValueError:
                pass

        # Static fallbacks for high availability and zero vision hallucinations
        fallback_trends = {
            "tcs": {"pkg_2021": 3.6, "pkg_2022": 3.8, "pkg_2023": 4.0, "pkg_2024": 4.1, "trend": "↑ Steady growth"},
            "infosys": {"pkg_2021": 36.0, "pkg_2022": 39.0, "pkg_2023": 41.5, "pkg_2024": 42.9, "trend": "↑ Strong growth"},
            "amazon": {"pkg_2021": 22.0, "pkg_2022": 25.0, "pkg_2023": 27.0, "pkg_2024": 28.6, "trend": "↑ Consistent rise"},
            "google": {"pkg_2021": 38.0, "pkg_2022": 40.0, "pkg_2023": 41.0, "pkg_2024": 42.0, "trend": "↑ Marginal growth"},
            "deloitte": {"pkg_2021": 7.0, "pkg_2022": 8.2, "pkg_2023": 9.0, "pkg_2024": 9.6, "trend": "↑ Steady growth"},
            "microsoft": {"pkg_2021": 19.0, "pkg_2022": 20.0, "pkg_2023": 21.0, "pkg_2024": 21.4, "trend": "↑ Slow growth"},
            "wipro": {"pkg_2021": 24.0, "pkg_2022": 25.0, "pkg_2023": 25.8, "pkg_2024": 26.1, "trend": "↑ Slow growth"},
            "cognizant": {"pkg_2021": 38.0, "pkg_2022": 40.0, "pkg_2023": 41.5, "pkg_2024": 42.3, "trend": "↑ Strong growth"},
            "accenture": {"pkg_2021": 14.0, "pkg_2022": 15.0, "pkg_2023": 16.5, "pkg_2024": 17.3, "trend": "↑ Moderate growth"},
            "flipkart": {"pkg_2021": 22.0, "pkg_2022": 23.0, "pkg_2023": 24.5, "pkg_2024": 25.3, "trend": "↑ Moderate growth"}
        }

        fallback_hiring = {
            "tcs": {"SDE": 88, "Analyst": 42, "Officer": 70, "Intern": 44, "Total": 244},
            "infosys": {"SDE": 30, "Analyst": 68, "Officer": 62, "Intern": 22, "Total": 182},
            "deloitte": {"SDE": 42, "Analyst": 85, "Officer": 62, "Intern": 44, "Total": 233},
            "accenture": {"SDE": 25, "Analyst": 22, "Officer": 52, "Intern": 68, "Total": 167},
            "amazon": {"SDE": 42, "Analyst": 36, "Officer": 40, "Intern": 82, "Total": 200},
            "flipkart": {"SDE": 58, "Analyst": 55, "Officer": 50, "Intern": 32, "Total": 195},
            "google": {"SDE": 30, "Analyst": 92, "Officer": 46, "Intern": 30, "Total": 198},
            "microsoft": {"SDE": 58, "Analyst": 58, "Officer": 36, "Intern": 68, "Total": 220},
            "wipro": {"SDE": 42, "Analyst": 92, "Officer": 40, "Intern": 82, "Total": 256},
            "cognizant": {"SDE": 48, "Analyst": 28, "Officer": 82, "Intern": 34, "Total": 192},
            "capgemini": {"SDE": 68, "Analyst": 38, "Officer": 50, "Intern": 58, "Total": 214},
            "ibm": {"SDE": 58, "Analyst": 38, "Officer": 78, "Intern": 68, "Total": 242},
            "adobe": {"SDE": 42, "Analyst": 80, "Officer": 62, "Intern": 48, "Total": 232},
            "oracle": {"SDE": 35, "Analyst": 92, "Officer": 62, "Intern": 95, "Total": 284},
            "sap": {"SDE": 48, "Analyst": 42, "Officer": 28, "Intern": 38, "Total": 156},
            "hcl": {"SDE": 48, "Analyst": 42, "Officer": 38, "Intern": 32, "Total": 160},
            "tech mahindra": {"SDE": 58, "Analyst": 28, "Officer": 58, "Intern": 30, "Total": 174},
            "qualcomm": {"SDE": 25, "Analyst": 38, "Officer": 82, "Intern": 78, "Total": 223},
            "intel": {"SDE": 48, "Analyst": 48, "Officer": 42, "Intern": 48, "Total": 186},
            "samsung r&d": {"SDE": 42, "Analyst": 80, "Officer": 42, "Intern": 38, "Total": 202}
        }

        # 2. INTENT SELECTION & DISPATCH
        
        # AA. HARD COMPUTED AGGREGATION TRIGGER
        is_computed_aggregation = any(
            kw in query_lower for kw in [
                "ratio", "efficiency", "best value", "package per cgpa", 
                "package-to-cgpa", "relative to eligibility", "package efficiency", 
                "package/cgpa", "salary relative to"
            ]
        )
        
        if is_computed_aggregation:
            import pandas as pd
            comp_df = active_df.copy()
            comp_df["company"] = comp_df["company"].str.replace(";", "").str.strip()
            comp_df["min_cgpa"] = pd.to_numeric(comp_df["min_cgpa"], errors="coerce")
            comp_df["package_lpa"] = pd.to_numeric(comp_df["package_lpa"], errors="coerce")
            
            # Avoid division by zero
            comp_df = comp_df[comp_df["min_cgpa"] > 0]
            comp_df = comp_df[comp_df["package_lpa"] > 0]
            
            comp_df["ratio"] = comp_df["package_lpa"] / comp_df["min_cgpa"]
            top_5 = comp_df.sort_values(by="ratio", ascending=False).head(5)
            
            top_5_list = []
            for idx, (_, row) in enumerate(top_5.iterrows(), 1):
                top_5_list.append({
                    "rank": idx,
                    "company": row["company"],
                    "ratio": round(float(row["ratio"]), 2),
                    "package_lpa": float(row["package_lpa"]),
                    "min_cgpa": float(row["min_cgpa"])
                })
                
            if not self.client:
                return self._format_computed_aggregation_local(top_5_list)
                
            system_prompt = (
                "You are a SVECW Placement Intelligence Assistant. Your task is to summarize the computed package-to-CGPA efficiency ratios of the top companies based ONLY on the provided calculations.\n"
                "CRITICAL GUARDRAILS:\n"
                "- Do NOT perform any math calculations yourself.\n"
                "- Use ONLY the ratios and company values explicitly provided in the user content.\n"
                "- Keep the response extremely short, structured, and matching the requested template.\n"
                "- Do NOT include any external company details, work culture opinions, or hallucinations."
            )
            
            top_5_json = json.dumps(top_5_list, indent=2)
            
            user_content = f"""
Formula Used: Package (LPA) ÷ Minimum CGPA
Top 5 Computed Ratios:
{top_5_json}

User Query: {query}

Please generate the response using the EXACT template below, inserting the computed values. Do not write any paragraphs or intro/outro text.

🎯 Package-to-CGPA Ratio Analysis

Formula Used:
Package (LPA) ÷ Minimum CGPA

🏆 Top Companies by Ratio

1. [Company 1] → [Ratio 1]
   ([Package 1] LPA ÷ [CGPA 1] CGPA)

2. [Company 2] → [Ratio 2]
   ([Package 2] LPA ÷ [CGPA 2] CGPA)

3. [Company 3] → [Ratio 3]
   ([Package 3] LPA ÷ [CGPA 3] CGPA)

4. [Company 4] → [Ratio 4]
   ([Package 4] LPA ÷ [CGPA 4] CGPA)

5. [Company 5] → [Ratio 5]
   ([Package 5] LPA ÷ [CGPA 5] CGPA)

📌 Best Overall:
[1-sentence explanation matching the best company's ratio, e.g. "Intel offers the strongest package-to-CGPA ratio in the dataset, meaning it provides the highest compensation relative to its eligibility requirement."]
"""
            try:
                chat_completion = self.client.chat.completions.create(
                    model=settings.GROQ_TEXT_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ],
                    temperature=0.0
                )
                return chat_completion.choices[0].message.content.strip()
            except Exception:
                return self._format_computed_aggregation_local(top_5_list)

        # A. HARD FULL SYNTHESIS TRIGGER
        comparison_keywords = ["compare", "all dimensions", "eligibility", "package", "hiring", "trend", "full comparison", "complete comparison"]
        is_full_synthesis = (
            any(kw in query_lower for kw in comparison_keywords)
            and len(mentioned_companies) >= 2
        ) or len(mentioned_companies) >= 2

        if is_full_synthesis:
            # 4-HOP Deterministic facts collection
            # HOP 1: Eligibility
            elig_data = {}
            for comp in mentioned_companies:
                row = active_df[active_df.company.str.lower().str.replace(";", "").str.strip() == comp.lower()]
                if not row.empty:
                    elig_data[comp] = {
                        "min_cgpa": float(row.iloc[0]["min_cgpa"]),
                        "max_backlogs": int(row.iloc[0]["max_backlogs"]),
                        "bond_years": int(row.iloc[0]["bond_years"]),
                        "tech_focus": str(row.iloc[0]["tech_focus"]),
                        "key_topics": str(row.iloc[0]["key_topics"])
                    }
                else:
                    elig_data[comp] = {"min_cgpa": 0.0, "max_backlogs": 0, "bond_years": 0, "tech_focus": "N/A", "key_topics": "N/A"}

            # HOP 2: Package
            pkg_data = {}
            for comp in mentioned_companies:
                row = active_df[active_df.company.str.lower().str.replace(";", "").str.strip() == comp.lower()]
                if not row.empty:
                    pkg_data[comp] = float(row.iloc[0]["package_lpa"])
                else:
                    pkg_data[comp] = 0.0

            # HOP 3: Hiring Distribution
            hiring_data_retrieved = {}
            for comp in mentioned_companies:
                c_key = comp.lower().replace(" r&d", "").replace(" samsung", "samsung").strip()
                found = False
                for k, v in fallback_hiring.items():
                    if k in c_key or c_key in k:
                        hiring_data_retrieved[comp] = v
                        found = True
                        break
                if not found:
                    hiring_data_retrieved[comp] = {"SDE": 0, "Analyst": 0, "Officer": 0, "Intern": 0, "Total": 0}

            # HOP 4: Placement Trend
            trends_data_retrieved = {}
            for comp in mentioned_companies:
                c_match = self.pandas_tool.df_trends[self.pandas_tool.df_trends.company.str.lower() == comp.lower()]
                if not c_match.empty:
                    trends_data_retrieved[comp] = {
                        "2021": float(c_match.iloc[0]["pkg_2021"]),
                        "2022": float(c_match.iloc[0]["pkg_2022"]),
                        "2023": float(c_match.iloc[0]["pkg_2023"]),
                        "2024": float(c_match.iloc[0]["pkg_2024"]),
                        "trend": str(c_match.iloc[0]["trend"]).replace("\u00e2\u2020\u2018", "↑").replace("â†‘", "↑")
                    }
                else:
                    c_key = comp.lower().strip()
                    found = False
                    for k, v in fallback_trends.items():
                        if k in c_key or c_key in k:
                            trends_data_retrieved[comp] = {
                                "2021": v["pkg_2021"],
                                "2022": v["pkg_2022"],
                                "2023": v["pkg_2023"],
                                "2024": v["pkg_2024"],
                                "trend": v["trend"].replace("\u00e2\u2020\u2018", "↑").replace("â†‘", "↑")
                            }
                            found = True
                            break
                    if not found:
                        trends_data_retrieved[comp] = {"2021": 0.0, "2022": 0.0, "2023": 0.0, "2024": 0.0, "trend": "Stable"}

            # Compile merged facts
            retrieved_facts = {}
            for comp in mentioned_companies:
                retrieved_facts[comp] = {
                    "Eligibility": elig_data[comp],
                    "Package": pkg_data[comp],
                    "Hiring": hiring_data_retrieved[comp],
                    "Trends": trends_data_retrieved[comp]
                }
            
            comp1 = mentioned_companies[0]
            comp2 = mentioned_companies[1] if len(mentioned_companies) > 1 else "Google"

            # Execute local fallback directly if Groq client is missing
            if not self.client:
                return self._format_full_synthesis_local(comp1, comp2, retrieved_facts[comp1], retrieved_facts.get(comp2, retrieved_facts[comp1]))

            # Structured Groq Prompt
            system_prompt = (
                "You are a SVECW Placement Intelligence Assistant. Your task is to perform a detailed comparison of companies on all dimensions based ONLY on the provided retrieved facts.\n"
                "CRITICAL GUARDRAILS:\n"
                "- Do NOT invent, assume, or extrapolate any salaries, bonds, tech focus, hiring counts, or trends.\n"
                "- Do NOT include any external knowledge, work culture opinions, or general company reputation.\n"
                "- Use ONLY the facts explicitly provided in the retrieved facts JSON.\n"
                "- If bond_years is 0, format it as 'No bond'. If it is 1, format it as '1 year'. Otherwise, format it as 'X years'.\n"
                "- For trend values, use the exact packages and trend text provided. Clean up trend arrows if needed.\n"
                "- Keep the response concise, structured, and hackathon-demo friendly."
            )

            facts_str = json.dumps(retrieved_facts, indent=2)

            user_content = f"""
Retrieved Facts (JSON):
{facts_str}

User Query: {query}

Please generate the response using the EXACT template below, replacing brackets with the compiled facts and comparisons. Do not write any paragraphs or intro/outro text.

🎯 Full Company Comparison: {comp1} vs {comp2}

1️⃣ Eligibility

{comp1}
• Minimum CGPA: [Min CGPA A]
• Backlogs Allowed: [Max Backlogs A]
• Bond: [Bond A]

{comp2}
• Minimum CGPA: [Min CGPA B]
• Backlogs Allowed: [Max Backlogs B]
• Bond: [Bond B]

📌 [1-sentence eligibility comparison, e.g. "Amazon is easier to qualify for due to lower CGPA requirements."]

---

2️⃣ Package

{comp1} → [Package A] LPA
{comp2} → [Package B] LPA

📌 [1-sentence package comparison, e.g. "Google offers higher compensation."]

---

3️⃣ Hiring Distribution

{comp1}: SDE: [SDE A], Analyst: [Analyst A], Officer: [Officer A], Intern: [Intern A] (Total: [Total A])
{comp2}: SDE: [SDE B], Analyst: [Analyst B], Officer: [Officer B], Intern: [Intern B] (Total: [Total B])

📌 [1-sentence hiring comparison, e.g. "Google hires more Analysts, while Amazon dominates Intern hiring."]

---

4️⃣ Placement Trend (2021–2024)

{comp1}: 2021: [pkg 2021 A], 2022: [pkg 2022 A], 2023: [pkg 2023 A], 2024: [pkg 2024 A] ([Trend A])
{comp2}: 2021: [pkg 2021 B], 2022: [pkg 2022 B], 2023: [pkg 2023 B], 2024: [pkg 2024 B] ([Trend B])

📌 [1-sentence trend comparison, e.g. "Amazon showed a more consistent rise, while Google remained relatively flat."]

---

🏆 Overall Recommendation

Choose {comp1} if:
• [bullet point fact-based advantage of {comp1}, e.g. Higher compensation matters / Less bond period]

Choose {comp2} if:
• [bullet point fact-based advantage of {comp2}, e.g. Easier eligibility matters / More intern positions]

Final Verdict:
[A grounded 1-2 sentence final verdict summarizing the choice based strictly on the facts.]
"""
            try:
                chat_completion = self.client.chat.completions.create(
                    model=settings.GROQ_TEXT_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ],
                    temperature=0.0
                )
                return chat_completion.choices[0].message.content.strip()
            except Exception:
                return self._format_full_synthesis_local(comp1, comp2, retrieved_facts[comp1], retrieved_facts.get(comp2, retrieved_facts[comp1]))

        # B. CASE 1: Standard Career Comparison / Recommendation (1 company comparison)
        if len(mentioned_companies) == 1 and any(kw in query_lower for kw in ["compare", "better", "versus", "vs", "career", "join"]):
            comp1 = mentioned_companies[0]
            comp2 = "Google" if comp1 != "Google" else "Microsoft"
            
            matched_rows = active_df[active_df.company.str.lower().isin([comp1.lower(), comp2.lower()])]
            if not matched_rows.empty:
                # Clean names
                matched_rows = matched_rows.copy()
                matched_rows["company"] = matched_rows["company"].str.replace(";", "")
                facts_str = matched_rows.to_json(orient="records", indent=2)

                if not self.client:
                    return self._format_comparison_local(matched_rows)

                system_prompt = (
                    "You are a SVECW Placement Intelligence Assistant.\n"
                    "Format a comparison and career recommendation based ONLY on the retrieved company facts. "
                    "DO NOT invent, assume, or generalize any facts about salaries, bonds, tech focus, benefits, or work culture. "
                    "If a fact is not present in the inputs, do not mention it. Keep the response short, clean, and concise."
                )

                user_content = f"""
Retrieved Company Facts:
{facts_str}

User Query: {query}

Format the final response using this EXACT layout:
🎯 Career Comparison: {comp1} vs {comp2}

{comp1}
• Package: [Package] LPA
• Tech Focus: [Tech Focus]
• Bond: [Bond Years] years (or "No bond")

{comp2}
• Package: [Package] LPA
• Tech Focus: [Tech Focus]
• Bond: [Bond Years] years (or "No bond")

🏆 Recommendation

Choose {comp1} if you prioritize:
• [Reason based ONLY on facts]
• [Reason based ONLY on facts]

Choose {comp2} if you prioritize:
• [Reason based ONLY on facts]
• [Reason based ONLY on facts]

📌 Overall:
[1-sentence summary based ONLY on package, bond, or tech focus comparison]
"""
                try:
                    chat_completion = self.client.chat.completions.create(
                        model=settings.GROQ_TEXT_MODEL,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_content}
                        ],
                        temperature=0.0
                    )
                    return chat_completion.choices[0].message.content.strip()
                except Exception:
                    return self._format_comparison_local(matched_rows)

        # C. CASE 3: Highest Paying Eligible Company (e.g. CGPA 7.6 & 1 backlog)
        if cgpa_val is not None and ("backlog" in query_lower or "backlogs" in query_lower):
            eligible = active_df[(active_df["min_cgpa"] <= cgpa_val) & (active_df["max_backlogs"] >= backlog_val)]
            
            if not eligible.empty:
                best_company = eligible.sort_values("package_lpa", ascending=False).iloc[0]
                comp_name = best_company["company"].replace(";", "").strip()

                if not self.client:
                    return self._format_highest_paying_local(cgpa_val, backlog_val, best_company)

                system_prompt = (
                    "You are a SVECW Placement Eligibility Recommendation Agent.\n"
                    "Format a response using ONLY the provided eligible company details. Do not invent facts."
                )
                
                user_content = f"""
Criteria: CGPA = {cgpa_val}, Backlogs = {backlog_val}
Best Eligible Company:
{{
  "company": "{comp_name}",
  "package_lpa": {best_company["package_lpa"]},
  "min_cgpa": {best_company["min_cgpa"]},
  "max_backlogs": {best_company["max_backlogs"]}
}}

User Query: {query}

Format the final response using this EXACT layout:
🎯 Eligibility Recommendation

Based on:
• CGPA = {cgpa_val}
• Backlogs = {backlog_val}

🏆 Best Eligible Company:
{comp_name}

Package:
{best_company["package_lpa"]} LPA

Reason:
Highest package among eligible companies with CGPA <= {cgpa_val} and backlogs allowed >= {backlog_val}.
"""
                try:
                    chat_completion = self.client.chat.completions.create(
                        model=settings.GROQ_TEXT_MODEL,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_content}
                        ],
                        temperature=0.0
                    )
                    return chat_completion.choices[0].message.content.strip()
                except Exception:
                    return self._format_highest_paying_local(cgpa_val, backlog_val, best_company)

        # D. CASE 2: Placement Eligibility Analysis (e.g. CGPA 5.0)
        if cgpa_val is not None:
            eligible = active_df[active_df["min_cgpa"] <= cgpa_val]
            
            if eligible.empty:
                nearest = active_df.sort_values("min_cgpa").head(3)
                nearest = nearest.copy()
                nearest["company"] = nearest["company"].str.replace(";", "")
                facts_str = nearest.to_json(orient="records", indent=2)

                if not self.client:
                    return self._format_eligibility_local(cgpa_val, eligible, nearest)

                system_prompt = (
                    "You are a SVECW Placement Eligibility Analysis Agent.\n"
                    "Format a response using ONLY the provided nearest company facts. Do not invent facts."
                )
                
                user_content = f"""
Current CGPA: {cgpa_val}
Nearest Companies Facts:
{facts_str}

User Query: {query}

Format the final response using this EXACT layout:
🎯 Placement Eligibility Analysis

Current CGPA: {cgpa_val}

⚠️ No eligible companies found.

Closest opportunities:
• [Company A] → [Min CGPA A]
• [Company B] → [Min CGPA B]
• [Company C] → [Min CGPA C]

💡 Recommendation:
Improve CGPA above 6.0 to unlock more opportunities.
"""
                try:
                    chat_completion = self.client.chat.completions.create(
                        model=settings.GROQ_TEXT_MODEL,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_content}
                        ],
                        temperature=0.0
                    )
                    return chat_completion.choices[0].message.content.strip()
                except Exception:
                    return self._format_eligibility_local(cgpa_val, eligible, nearest)
            else:
                eligible = eligible.copy()
                eligible["company"] = eligible["company"].str.replace(";", "")
                facts_str = eligible.to_json(orient="records", indent=2)

                if not self.client:
                    return self._format_eligibility_local(cgpa_val, eligible, None)

                system_prompt = (
                    "You are a SVECW Placement Eligibility Analysis Agent.\n"
                    "Format a response using ONLY the provided eligible company facts. Do not invent facts."
                )
                
                user_content = f"""
Current CGPA: {cgpa_val}
Eligible Companies Facts:
{facts_str}

User Query: {query}

Format the final response using this EXACT layout:
🎯 Placement Eligibility Analysis

Current CGPA: {cgpa_val}

✅ Based on your CGPA, you are eligible to apply for the following companies:
• [Company Name] (Package: [Package] LPA) — Min CGPA: [Min CGPA]

💡 Recommendation:
Review interview experience details for these companies to prepare.
"""
                try:
                    chat_completion = self.client.chat.completions.create(
                        model=settings.GROQ_TEXT_MODEL,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_content}
                        ],
                        temperature=0.0
                    )
                    return chat_completion.choices[0].message.content.strip()
                except Exception:
                    return self._format_eligibility_local(cgpa_val, eligible, None)

        # General LLM Fallback (synthesize full context)
        if not self.client:
            return "⚠️ Multi-Hop Agent Error: Groq API Key is not configured."

        system_prompt = (
            "You are a Multi-Hop Placement Analytics Agent. Your goal is to synthesize structured query data "
            "across different tables (eligibility, hiring distributions, and placement trends) to solve multi-step reasoning questions.\n"
            "CRITICAL GUARDRAILS:\n"
            "- Do NOT invent, assume, or extrapolate any facts not explicitly present in the data.\n"
            "- Do NOT hallucinate packages, CGPA requirements, backlogs, or bonds.\n"
            "Here is the database context:\n"
            f"Eligibility Table (df):\n{active_df.to_markdown(index=False) if not active_df.empty else 'No data'}\n\n"
            "Use this information to answer the user's query."
        )

        try:
            chat_completion = self.client.chat.completions.create(
                model=settings.GROQ_TEXT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0.0
            )
            response = chat_completion.choices[0].message.content.strip()
            header = f"🧠 **[Multi-Hop Reasoning Agent]**\n\n"
            return header + response
        except Exception as e:
            return f"❌ Multi-Hop Agent execution error: {str(e)}"

    # LOCAL FORMATTING HELPERS
    def _format_comparison_local(self, matched_rows) -> str:
        rows = list(matched_rows.iterrows())
        if len(rows) < 2:
            row1 = rows[0][1]
            comp1 = row1['company'].replace(';', '').strip()
            comp2 = "Google" if comp1 != "Google" else "Microsoft"
            return f"🎯 Career Comparison: {comp1} vs {comp2}\n\n{comp1}\n• Package: {row1['package_lpa']} LPA\n• Tech Focus: {row1['tech_focus']}\n• Bond: {row1['bond_years']} years\n"
        
        row1 = rows[0][1]
        row2 = rows[1][1]
        
        comp1 = row1['company'].replace(';', '').strip()
        comp2 = row2['company'].replace(';', '').strip()
        
        pkg1 = f"{row1['package_lpa']:.1f} LPA"
        pkg2 = f"{row2['package_lpa']:.1f} LPA"
        
        tech1 = f"{row1['tech_focus']}, {row1['key_topics']}" if row1['tech_focus'] else row1['key_topics']
        tech1_parts = [p.strip() for p in tech1.split(",") if p.strip()]
        seen = set()
        tech1_clean = ", ".join([x for x in tech1_parts if not (x.lower() in seen or seen.add(x.lower()))])
        
        tech2 = f"{row2['tech_focus']}, {row2['key_topics']}" if row2['tech_focus'] else row2['key_topics']
        tech2_parts = [p.strip() for p in tech2.split(",") if p.strip()]
        seen2 = set()
        tech2_clean = ", ".join([x for x in tech2_parts if not (x.lower() in seen2 or seen2.add(x.lower()))])
        
        bond1 = "No bond" if int(row1['bond_years']) == 0 else f"{int(row1['bond_years'])} year" if int(row1['bond_years']) == 1 else f"{int(row1['bond_years'])} years"
        bond2 = "No bond" if int(row2['bond_years']) == 0 else f"{int(row2['bond_years'])} year" if int(row2['bond_years']) == 1 else f"{int(row2['bond_years'])} years"
        
        rec_bullets_1 = []
        rec_bullets_2 = []
        
        val_pkg1 = float(row1['package_lpa'])
        val_pkg2 = float(row2['package_lpa'])
        
        if val_pkg1 > val_pkg2:
            rec_bullets_1.append("Higher compensation")
            rec_bullets_1.append("Stronger compensation package")
        elif val_pkg2 > val_pkg1:
            rec_bullets_2.append("Higher compensation")
            rec_bullets_2.append("Stronger compensation package")
            
        val_cgpa1 = float(row1['min_cgpa'])
        val_cgpa2 = float(row2['min_cgpa'])
        if val_cgpa1 < val_cgpa2:
            rec_bullets_1.append("Lower eligibility barrier")
        elif val_cgpa2 < val_cgpa1:
            rec_bullets_2.append("Lower eligibility barrier")
            
        val_bond1 = int(row1['bond_years'])
        val_bond2 = int(row2['bond_years'])
        if val_bond1 == 0 and val_bond2 > 0:
            rec_bullets_1.append("No bond")
        elif val_bond2 == 0 and val_bond1 > 0:
            rec_bullets_2.append("No bond")
            
        if not rec_bullets_1:
            rec_bullets_1.append(f"Focus on {tech1_clean}")
            rec_bullets_1.append("Different growth trajectory")
        if not rec_bullets_2:
            rec_bullets_2.append(f"Focus on {tech2_clean}")
            rec_bullets_2.append("Different growth trajectory")
            
        if len(rec_bullets_1) < 2:
            rec_bullets_1.append(f"Focus on {row1['tech_focus']}")
        if len(rec_bullets_2) < 2:
            rec_bullets_2.append(f"Focus on {row2['tech_focus']}")
            
        if val_pkg1 > val_pkg2 and val_bond2 == 0 and val_bond1 > 0:
            overall = f"{comp1} offers stronger compensation, while {comp2} provides more flexibility."
        elif val_pkg2 > val_pkg1 and val_bond1 == 0 and val_bond2 > 0:
            overall = f"{comp2} offers stronger compensation, while {comp1} provides more flexibility."
        else:
            overall = f"{comp1} offers {pkg1} package with {bond1}, while {comp2} offers {pkg2} package with {bond2}."
            
        return f"""🎯 Career Comparison: {comp1} vs {comp2}

{comp1}
• Package: {pkg1}
• Tech Focus: {tech1_clean}
• Bond: {bond1}

{comp2}
• Package: {pkg2}
• Tech Focus: {tech2_clean}
• Bond: {bond2}

🏆 Recommendation

Choose {comp1} if you prioritize:
{chr(10).join([f"• {b}" for b in rec_bullets_1[:2]])}

Choose {comp2} if you prioritize:
{chr(10).join([f"• {b}" for b in rec_bullets_2[:2]])}

📌 Overall:
{overall}"""

    def _format_eligibility_local(self, cgpa_val, eligible, nearest) -> str:
        if eligible.empty:
            lines = [
                "🎯 Placement Eligibility Analysis",
                "",
                f"Current CGPA: {cgpa_val}",
                "",
                "⚠️ No eligible companies found.",
                "",
                "Closest opportunities:"
            ]
            for _, row in nearest.head(3).iterrows():
                comp_name = row['company'].replace(';', '').strip()
                lines.append(f"• {comp_name} → {row['min_cgpa']:.1f}")
            lines.append("")
            lines.append("💡 Recommendation:")
            lines.append("Improve CGPA above 6.0 to unlock more opportunities.")
            return "\n".join(lines)
        else:
            lines = [
                "🎯 Placement Eligibility Analysis",
                "",
                f"Current CGPA: {cgpa_val}",
                "",
                "✅ Based on your CGPA, you are eligible to apply for the following companies:"
            ]
            for _, row in eligible.sort_values("package_lpa", ascending=False).iterrows():
                comp_name = row['company'].replace(';', '').strip()
                lines.append(f"• {comp_name} (Package: {row['package_lpa']:.1f} LPA) — Min CGPA: {row['min_cgpa']:.1f}")
            lines.append("")
            lines.append("💡 Recommendation:")
            lines.append("Review interview experience details for these companies to prepare.")
            return "\n".join(lines)

    def _format_highest_paying_local(self, cgpa_val, backlog_val, best_company) -> str:
        comp_name = best_company["company"].replace(";", "").strip()
        pkg = best_company["package_lpa"]
        return f"""🎯 Eligibility Recommendation

Based on:
• CGPA = {cgpa_val}
• Backlogs = {backlog_val}

🏆 Best Eligible Company:
{comp_name}

Package:
{pkg:.1f} LPA

Reason:
Highest package among eligible companies."""

    def _format_full_synthesis_local(self, comp1, comp2, facts1, facts2) -> str:
        elig1 = facts1["Eligibility"]
        elig2 = facts2["Eligibility"]
        
        bond1 = "No bond" if elig1["bond_years"] == 0 else f"{elig1['bond_years']} year" if elig1["bond_years"] == 1 else f"{elig1['bond_years']} years"
        bond2 = "No bond" if elig2["bond_years"] == 0 else f"{elig2['bond_years']} year" if elig2["bond_years"] == 1 else f"{elig2['bond_years']} years"
        
        if elig1["min_cgpa"] < elig2["min_cgpa"]:
            elig_comp = f"{comp1} is easier to qualify for due to lower CGPA requirements."
        elif elig2["min_cgpa"] < elig1["min_cgpa"]:
            elig_comp = f"{comp2} is easier to qualify for due to lower CGPA requirements."
        else:
            elig_comp = f"Both have identical CGPA cutoffs ({elig1['min_cgpa']})."
            
        pkg1_val = float(facts1["Package"])
        pkg2_val = float(facts2["Package"])
        if pkg1_val > pkg2_val:
            pkg_comp = f"{comp1} offers higher compensation."
        elif pkg2_val > pkg1_val:
            pkg_comp = f"{comp2} offers higher compensation."
        else:
            pkg_comp = f"Both offer identical packages of {pkg1_val} LPA."
            
        h1 = facts1["Hiring"]
        h2 = facts2["Hiring"]
        
        roles_comp = []
        if h1["SDE"] > h2["SDE"]:
            roles_comp.append(f"{comp1} hires more SDEs ({h1['SDE']} vs {h2['SDE']})")
        elif h2["SDE"] > h1["SDE"]:
            roles_comp.append(f"{comp2} hires more SDEs ({h2['SDE']} vs {h1['SDE']})")
            
        if h1["Analyst"] > h2["Analyst"]:
            roles_comp.append(f"{comp1} hires more Analysts ({h1['Analyst']} vs {h2['Analyst']})")
        elif h2["Analyst"] > h1["Analyst"]:
            roles_comp.append(f"{comp2} hires more Analysts ({h2['Analyst']} vs {h1['Analyst']})")
            
        if h1["Intern"] > h2["Intern"]:
            roles_comp.append(f"{comp1} hires more Interns ({h1['Intern']} vs {h2['Intern']})")
        elif h2["Intern"] > h1["Intern"]:
            roles_comp.append(f"{comp2} hires more Interns ({h2['Intern']} vs {h1['Intern']})")
            
        hiring_comp = ", while ".join(roles_comp[:2]) + "." if roles_comp else "Both have similar hiring profiles."
        
        t1 = facts1["Trends"]
        t2 = facts2["Trends"]
        
        growth1 = t1["2024"] - t1["2021"]
        growth2 = t2["2024"] - t2["2021"]
        
        clean_trend1 = t1["trend"].replace("↑", "").strip()
        clean_trend2 = t2["trend"].replace("↑", "").strip()
        
        if growth1 > growth2:
            trend_comp = f"{comp1} showed higher package growth from 2021 to 2024 (+{growth1:.1f} LPA vs +{growth2:.1f} LPA)."
        elif growth2 > growth1:
            trend_comp = f"{comp2} showed higher package growth from 2021 to 2024 (+{growth2:.1f} LPA vs +{growth1:.1f} LPA)."
        else:
            trend_comp = f"Both showed similar package growth trajectories."
            
        bullets1 = []
        bullets2 = []
        
        if pkg1_val > pkg2_val:
            bullets1.append("Higher package compensation matters")
        if elig1["min_cgpa"] < elig2["min_cgpa"]:
            bullets1.append("Easier eligibility matters")
        if elig1["bond_years"] < elig2["bond_years"]:
            bullets1.append("Shorter bond duration matters")
            
        if pkg2_val > pkg1_val:
            bullets2.append("Higher package compensation matters")
        if elig2["min_cgpa"] < elig1["min_cgpa"]:
            bullets2.append("Easier eligibility matters")
        if elig2["bond_years"] < elig1["bond_years"]:
            bullets2.append("Shorter bond duration matters")
            
        if not bullets1:
            bullets1.append(f"Preference for {comp1}'s focus")
        if not bullets2:
            bullets2.append(f"Preference for {comp2}'s focus")
            
        verdict = f"{comp1} offers stronger compensation, while {comp2} provides more flexibility."

        return f"""🎯 Full Company Comparison: {comp1} vs {comp2}

1️⃣ Eligibility

{comp1}
• Minimum CGPA: {elig1['min_cgpa']}
• Backlogs Allowed: {elig1['max_backlogs']}
• Bond: {bond1}

{comp2}
• Minimum CGPA: {elig2['min_cgpa']}
• Backlogs Allowed: {elig2['max_backlogs']}
• Bond: {bond2}

📌 {elig_comp}

---

2️⃣ Package

{comp1} → {pkg1_val:.1f} LPA
{comp2} → {pkg2_val:.1f} LPA

📌 {pkg_comp}

---

3️⃣ Hiring Distribution

{comp1}: SDE: {h1['SDE']}, Analyst: {h1['Analyst']}, Officer: {h1['Officer']}, Intern: {h1['Intern']} (Total: {h1['Total']})
{comp2}: SDE: {h2['SDE']}, Analyst: {h2['Analyst']}, Officer: {h2['Officer']}, Intern: {h2['Intern']} (Total: {h2['Total']})

📌 {hiring_comp}

---

4️⃣ Placement Trend (2021–2024)

{comp1}: 2021: {t1['2021']:.1f}, 2022: {t1['2022']:.1f}, 2023: {t1['2023']:.1f}, 2024: {t1['2024']:.1f} ({t1['trend']})
{comp2}: 2021: {t2['2021']:.1f}, 2022: {t2['2022']:.1f}, 2023: {t2['2023']:.1f}, 2024: {t2['2024']:.1f} ({t2['trend']})

📌 {trend_comp}

---

🏆 Overall Recommendation

Choose {comp1} if:
• {bullets1[0]}

Choose {comp2} if:
• {bullets2[0]}

Final Verdict:
{verdict}"""

    def _format_computed_aggregation_local(self, top_5_list: list) -> str:
        best_comp = top_5_list[0]["company"]
        lines = [
            "🎯 Package-to-CGPA Ratio Analysis",
            "",
            "Formula Used:",
            "Package (LPA) ÷ Minimum CGPA",
            "",
            "🏆 Top Companies by Ratio",
            ""
        ]
        for item in top_5_list:
            lines.append(f"{item['rank']}. {item['company']} → {item['ratio']:.2f}")
            lines.append(f"   ({item['package_lpa']:.1f} LPA ÷ {item['min_cgpa']:.1f} CGPA)")
            lines.append("")
        
        lines.append(f"📌 Best Overall:")
        lines.append(f"{best_comp} offers the strongest package-to-CGPA ratio in the dataset, meaning it provides the highest compensation relative to its eligibility requirement.")
        
        return "\n".join(lines)
