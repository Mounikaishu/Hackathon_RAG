import unittest
import pandas as pd
from app.agents.router_agent import RouterAgent
from app.tools.pandas_tool import PandasTool
from app.tools.calculator_tool import CalculatorTool

class TestAgentsAndTools(unittest.TestCase):
    """
    Unit tests to verify routing decision trees, sandboxed DataFrame queries,
    and calculator mathematical parsing limits.
    """

    def setUp(self):
        self.router = RouterAgent()
        self.pandas_tool = PandasTool()
        self.calculator = CalculatorTool()

    def test_fallback_routing_web_search(self):
        """Checks if adversarial out-of-corpus queries route to web_search_agent."""
        query = "What is the current Infosys stock price today?"
        route = self.router._fallback_keyword_routing(query)
        self.assertEqual(route["agent"], "web_search_agent")

    def test_fallback_routing_vision(self):
        """Checks if chart-based queries route to vision_agent."""
        query = "What company hires the most analyst roles according to the bar chart?"
        route = self.router._fallback_keyword_routing(query)
        self.assertEqual(route["agent"], "vision_agent")

    def test_fallback_routing_conflict(self):
        """Checks if conflict-oriented queries route to conflict_agent."""
        query = "Is there a discrepancy or contradiction in Amazon's cutoff criteria?"
        route = self.router._fallback_keyword_routing(query)
        self.assertEqual(route["agent"], "conflict_agent")

    def test_fallback_routing_dataframe(self):
        """Checks if numeric cgpa / package threshold queries route to dataframe_agent."""
        query = "Companies offering package > 20 LPA with CGPA of 7.5"
        route = self.router._fallback_keyword_routing(query)
        self.assertEqual(route["agent"], "dataframe_agent")

    def test_eligibility_recommendation_routing(self):
        """Checks if eligibility queries with CGPA route to multi_hop_agent override."""
        query = "I have CGPA 5.0. Where can I apply?"
        route = self.router.route_query(query)
        self.assertEqual(route["agent"], "multi_hop_agent")

    def test_comparison_recommendation_routing(self):
        """Checks if comparison queries route to multi_hop_agent resolver."""
        query = "Should I join Google or Microsoft? Which is better for my career?"
        route = self.router.route_query(query)
        self.assertEqual(route["agent"], "multi_hop_agent")

    def test_calculator_valid(self):
        """Verifies correct execution of valid algebraic expressions."""
        expr = "(28.6 + 42.0) / 2"
        res = self.calculator.calculate(expr)
        self.assertIn("35.3", res)

    def test_calculator_invalid_alphabetic(self):
        """Verifies that calculator blocks arbitrary python commands/alphabeticals."""
        unsafe_expr = "import os; os.system('echo hack')"
        res = self.calculator.calculate(unsafe_expr)
        self.assertIn("unsafe", res.lower())

    def test_calculator_division_by_zero(self):
        """Verifies that calculator handles division by zero safely."""
        expr = "10 / 0"
        res = self.calculator.calculate(expr)
        self.assertIn("zero", res.lower())

    def test_pandas_sandbox_execution(self):
        """Verifies that pandas tool evaluates query strings safely on a mock df."""
        # Create a mock df inside the pandas tool
        test_df = pd.DataFrame([
            {"company": "Google", "min_cgpa": 7.4, "package_lpa": 42.0},
            {"company": "TCS", "min_cgpa": 7.5, "package_lpa": 4.1}
        ])
        
        self.pandas_tool.df = test_df
        
        # Test basic retrieval pandas code
        query_code = "df[df['min_cgpa'] <= 7.4]"
        response = self.pandas_tool.execute_query(query_code)
        
        self.assertTrue(response["success"])
        self.assertIn("Google", response["result"])
        self.assertNotIn("TCS", response["result"])

    def test_multi_hop_agent_career_comparison(self):
        """Verifies Case 1: Standard Career Comparison response structure."""
        from app.agents.multi_hop_agent import MultiHopAgent
        agent = MultiHopAgent()
        agent.client = None
        
        query = "Should I join TCS? Which is better for my career?"
        response = agent.process_query(query)
        
        self.assertIn("Career Comparison: TCS vs Google", response)
        self.assertIn("• Package:", response)
        self.assertIn("• Tech Focus:", response)
        self.assertIn("• Bond:", response)
        self.assertIn("Recommendation", response)

    def test_multi_hop_agent_eligibility_analysis_empty(self):
        """Verifies Case 2: Placement Eligibility Analysis when no companies are eligible."""
        from app.agents.multi_hop_agent import MultiHopAgent
        agent = MultiHopAgent()
        agent.client = None
        
        query = "I have CGPA 5.0. Where can I apply?"
        response = agent.process_query(query)
        
        self.assertIn("Placement Eligibility Analysis", response)
        self.assertIn("No eligible companies found", response)
        self.assertIn("Closest opportunities", response)
        self.assertIn("💡 Recommendation:", response)

    def test_multi_hop_agent_eligibility_analysis_eligible(self):
        """Verifies Case 2: Placement Eligibility Analysis when eligible companies are found."""
        from app.agents.multi_hop_agent import MultiHopAgent
        agent = MultiHopAgent()
        agent.client = None
        
        query = "I have CGPA 8.5. Where can I apply?"
        response = agent.process_query(query)
        
        self.assertIn("Placement Eligibility Analysis", response)
        self.assertIn("eligible to apply for the following companies", response)

    def test_multi_hop_agent_highest_paying(self):
        """Verifies Case 3: Highest Paying Eligible Company selection and reasoning."""
        from app.agents.multi_hop_agent import MultiHopAgent
        agent = MultiHopAgent()
        agent.client = None
        
        query = "A student with CGPA 7.6 and 1 backlog wants the highest-paying company"
        response = agent.process_query(query)
        
        self.assertIn("Eligibility Recommendation", response)
        self.assertIn("Best Eligible Company:", response)
        self.assertIn("Package:", response)

    def test_multi_hop_agent_hard_full_synthesis(self):
        """Verifies Hard Full Synthesis mode execution and headings presence."""
        from app.agents.multi_hop_agent import MultiHopAgent
        agent = MultiHopAgent()
        agent.client = None
        
        query = "H7 Compare Google and Amazon on all dimensions: eligibility, package, hiring, trend."
        response = agent.process_query(query)
        
        self.assertIn("Full Company Comparison: Google vs Amazon", response)
        self.assertIn("1️⃣ Eligibility", response)
        self.assertIn("2️⃣ Package", response)
        self.assertIn("3️⃣ Hiring Distribution", response)
        self.assertIn("4️⃣ Placement Trend (2021–2024)", response)
        self.assertIn("🏆 Overall Recommendation", response)
 
    def test_multi_hop_agent_computed_ratio_aggregation(self):
        """Verifies computed package-to-CGPA ratio aggregation execution."""
        from app.agents.multi_hop_agent import MultiHopAgent
        agent = MultiHopAgent()
        agent.client = None
        
        query = "Which company offers the best package-to-CGPA ratio?"
        response = agent.process_query(query)
        
        self.assertIn("Package-to-CGPA Ratio Analysis", response)
        self.assertIn("Formula Used:", response)
        self.assertIn("🏆 Top Companies by Ratio", response)
        self.assertIn("Intel → 5.91", response)
        self.assertIn("Qualcomm → 5.74", response)
        self.assertIn("Google → 5.68", response)
        self.assertIn("📌 Best Overall:", response)

    def test_conflict_agent_amazon_cgpa(self):
        """Verifies Case 1: Amazon CGPA conflict explanation and formatting."""
        from app.agents.conflict_agent import ConflictAgent
        agent = ConflictAgent()
        agent.client = None
        
        query = "Is the Amazon CGPA cutoff 6.4 or 7.0? Explain."
        response = agent.process_query(query)
        
        self.assertIn("⚠️ Conflict Detected", response)
        self.assertIn("Two different CGPA cutoff values were found for Amazon.", response)
        self.assertIn("• Official Placement Source → 6.4 CGPA", response)
        self.assertIn("• Placement Portal Source → 7.0 CGPA", response)
        self.assertIn("🧠 Explanation", response)
        self.assertIn("Since official placement records are considered more reliable than secondary portal information, the system prioritizes the official source.", response)
        self.assertIn("✅ Recommended Value", response)
        self.assertIn("Amazon CGPA cutoff = 6.4", response)

    def test_conflict_agent_google_package(self):
        """Verifies Case 2: Google package conflict explanation and formatting."""
        from app.agents.conflict_agent import ConflictAgent
        agent = ConflictAgent()
        agent.client = None
        
        query = "What is Google's package? 42 or 45 LPA?"
        response = agent.process_query(query)
        
        self.assertIn("⚠️ Conflict Detected", response)
        self.assertIn("Two package values were found for Google.", response)
        self.assertIn("• Official Placement Source → 42.0 LPA", response)
        self.assertIn("• Secondary Portal Source → 45.0 LPA", response)
        self.assertIn("🧠 Explanation", response)
        self.assertIn("Official placement records are prioritized because they are institution-verified.", response)
        self.assertIn("✅ Recommended Value", response)
        self.assertIn("Google package = 42.0 LPA", response)

    def test_conflict_agent_no_conflict(self):
        """Verifies Case 3: Consistent sources response layout (No Conflict)."""
        from app.agents.conflict_agent import ConflictAgent
        agent = ConflictAgent()
        agent.client = None
        
        query = "What is Amazon CGPA cutoff?"
        response = agent.process_query(query)
        
        self.assertIn("✅ No Conflict Detected", response)
        self.assertIn("The retrieved sources consistently report Amazon's minimum CGPA cutoff as 6.4.", response)

    def test_router_override_conflict(self):
        """Verifies that conflict keywords trigger immediate override routing to conflict_agent."""
        from app.agents.router_agent import RouterAgent
        router = RouterAgent()
        
        queries = [
            "Which company had conflicting CGPA data across sources?",
            "What companies have mismatch in package?",
            "Is there a discrepancy in cutoff?",
            "Tell me if there is inconsistency in bond details"
        ]
        for q in queries:
            route = router.route_query(q)
            self.assertEqual(route["agent"], "conflict_agent", f"Failed to route query: '{q}'")

    def test_conflict_agent_global_scan_cgpa(self):
        """Verifies Example 1: Global CGPA conflict scan and formatting."""
        from app.agents.conflict_agent import ConflictAgent
        agent = ConflictAgent()
        agent.client = None
        
        query = "Which company had conflicting CGPA data across sources?"
        response = agent.process_query(query)
        
        self.assertIn("⚠️ Conflict Detection Analysis", response)
        self.assertIn("The following companies contain conflicting CGPA cutoff values across retrieved sources:", response)
        self.assertIn("• Amazon\nOfficial → 6.4\nPortal → 7.0", response)
        self.assertIn("• Google\nOfficial → 7.4\nPortal → 7.8", response)
        self.assertIn("📌 Summary", response)
        self.assertIn("2 companies were found with conflicting CGPA information.", response)
        self.assertIn("Official placement records are prioritized during conflict resolution.", response)

    def test_conflict_agent_global_scan_bond(self):
        """Verifies Example 2: Global Bond scan returns 'No Conflicts Detected'."""
        from app.agents.conflict_agent import ConflictAgent
        agent = ConflictAgent()
        agent.client = None
        
        query = "Which companies had conflicting bond information?"
        response = agent.process_query(query)
        
        self.assertIn("✅ No Conflicts Detected", response)
        self.assertIn("No companies were found with conflicting bond information across retrieved sources.", response)

    def test_multi_hop_hard_filter_sort(self):
        """Verifies Example 1: Hard Filter + Sort Mode for the benchmark query."""
        from app.agents.multi_hop_agent import MultiHopAgent
        agent = MultiHopAgent()
        agent.client = None
        
        query = "H3 For CGPA 8.0+, zero backlog students, rank companies by package."
        response = agent.process_query(query)
        
        self.assertIn("🎯 Company Ranking for CGPA 8.0+ and Zero Backlogs", response)
        self.assertIn("Eligibility Criteria:", response)
        self.assertIn("• CGPA ≥ 8.0", response)
        self.assertIn("• Backlogs = 0", response)
        self.assertIn("🏆 Ranked by Package", response)
        self.assertIn("1. Infosys → 42.9 LPA", response)
        self.assertIn("2. Cognizant → 42.3 LPA", response)
        self.assertIn("3. Intel → 41.4 LPA", response)
        self.assertIn("4. Qualcomm → 41.3 LPA", response)
        self.assertIn("5. Capgemini → 38.3 LPA", response)
        self.assertIn("6. Tech Mahindra → 35.9 LPA", response)
        self.assertIn("7. SAP → 20.7 LPA", response)
        self.assertIn("8. Accenture → 17.3 LPA", response)
        self.assertIn("📌 Best Option:", response)
        self.assertIn("Infosys offers the highest package among companies matching the eligibility criteria.", response)

    def test_multi_hop_hard_filter_sort_general(self):
        """Verifies dynamic sorting and filtering on a general query."""
        from app.agents.multi_hop_agent import MultiHopAgent
        agent = MultiHopAgent()
        agent.client = None
        
        query = "Top companies for CGPA 7.5 and no backlogs"
        response = agent.process_query(query)
        
        self.assertIn("🎯 Company Ranking for CGPA 7.5+ and Zero Backlogs", response)
        self.assertIn("• CGPA ≥ 7.5", response)
        self.assertIn("• Backlogs = 0", response)
        self.assertIn("1. Google → 42.0 LPA", response)
        self.assertIn("2. Intel → 41.4 LPA", response)
        self.assertIn("Google offers the highest package among companies matching the eligibility criteria.", response)

    def test_router_override_join(self):
        """Verifies that hard join queries route immediately to multi_hop_agent."""
        from app.agents.router_agent import RouterAgent
        router = RouterAgent()
        
        queries = [
            "Which Python-focused company hires the most Interns?",
            "Which Java-focused company hires the most SDEs?",
            "Which Cloud-focused company hires the most analysts?"
        ]
        for q in queries:
            route = router.route_query(q)
            self.assertEqual(route["agent"], "multi_hop_agent", f"Failed to route query: '{q}'")

    def test_multi_hop_hard_join_benchmark(self):
        """Verifies Example 3: Hard Join Mode for the benchmark query."""
        from app.agents.multi_hop_agent import MultiHopAgent
        agent = MultiHopAgent()
        agent.client = None
        
        query = "H2 Which Python-focused company hires the most Interns?"
        response = agent.process_query(query)
        
        self.assertIn("🎯 Python-Focused Internship Hiring Analysis", response)
        self.assertIn("Python-focused companies found:", response)
        self.assertIn("• Google", response)
        self.assertIn("• Oracle", response)
        self.assertIn("📊 Intern Hiring Comparison", response)
        self.assertIn("Google → 30 Interns", response)
        self.assertIn("Oracle → 92 Interns", response)
        self.assertIn("🏆 Best Match:", response)
        self.assertIn("Oracle hires the highest number of interns among Python-focused companies.", response)

    def test_multi_hop_hard_join_general(self):
        """Verifies dynamic joining and filtering on a general query."""
        from app.agents.multi_hop_agent import MultiHopAgent
        agent = MultiHopAgent()
        agent.client = None
        
        query = "Which Java-focused company hires the most SDEs?"
        response = agent.process_query(query)
        
        self.assertIn("🎯 Java-Focused SDE Hiring Analysis", response)
        self.assertIn("Java-focused companies found:", response)
        self.assertIn("• Infosys", response)
        self.assertIn("• Cognizant", response)
        self.assertIn("• Samsung R&D", response)
        self.assertIn("📊 SDE Hiring Comparison", response)
        self.assertIn("Infosys → 30 SDEs", response)
        self.assertIn("Cognizant → 48 SDEs", response)
        self.assertIn("Samsung R&D → 42 SDEs", response)
        self.assertIn("🏆 Best Match:", response)
        self.assertIn("Cognizant hires the highest number of sdes among Java-focused companies.", response)

    def test_router_override_optimization(self):
        """Verifies that 3-condition optimization queries route immediately to multi_hop_agent."""
        from app.agents.router_agent import RouterAgent
        router = RouterAgent()
        
        query = "A student with CGPA 7.0, 1 backlog wants maximum pay with no bond"
        route_result = router.route_query(query)
        self.assertEqual(route_result["agent"], "multi_hop_agent")
        self.assertIn("Hard 3-condition optimization query detected.", route_result["reason"])

    def test_multi_hop_optimization_benchmark(self):
        """Verifies Example 1: Hard 3-Condition Filter Mode for the benchmark query."""
        from app.agents.multi_hop_agent import MultiHopAgent
        agent = MultiHopAgent()
        agent.client = None
        
        query = "H1 A student with CGPA 7.0, 1 backlog wants maximum pay with no bond"
        response = agent.process_query(query)
        
        self.assertIn("🎯 Placement Optimization Analysis", response)
        self.assertIn("Student Profile:", response)
        self.assertIn("• CGPA: 7.0", response)
        self.assertIn("• Backlogs: 1", response)
        self.assertIn("• Bond Preference: No bond", response)
        self.assertIn("🏆 Best Eligible Company:", response)
        self.assertIn("Microsoft", response)
        self.assertIn("📋 Eligibility Match", response)
        self.assertIn("• Minimum CGPA → 6.1 ✅", response)
        self.assertIn("• Allows 1 backlog ✅", response)
        self.assertIn("• Bond Requirement → 0 years ✅", response)
        self.assertIn("💰 Package Offered:", response)
        self.assertIn("21.4 LPA", response)
        self.assertIn("📌 Why this company?", response)
        self.assertIn("Microsoft offers the highest package among companies satisfying all three constraints.", response)

    def test_multi_hop_optimization_general(self):
        """Verifies dynamic 3-condition optimization on a different profile where bond is allowed."""
        from app.agents.multi_hop_agent import MultiHopAgent
        agent = MultiHopAgent()
        agent.client = None
        
        query = "A student with CGPA 6.4, 1 backlog wants maximum pay"
        response = agent.process_query(query)
        
        self.assertIn("🎯 Placement Optimization Analysis", response)
        self.assertIn("Student Profile:", response)
        self.assertIn("• CGPA: 6.4", response)
        self.assertIn("• Backlogs: 1", response)
        self.assertIn("• Bond Preference: Allowed", response)
        self.assertIn("🏆 Best Eligible Company:", response)
        self.assertIn("Amazon", response)
        self.assertIn("📋 Eligibility Match", response)
        self.assertIn("• Minimum CGPA → 6.4 ✅", response)
        self.assertIn("• Allows 1 backlog ✅", response)
        self.assertIn("• Bond Requirement → 2 years ✅", response)
        self.assertIn("💰 Package Offered:", response)
        self.assertIn("28.6 LPA", response)

    def test_router_override_tech_focus(self):
        """Verifies that tech-focus queries route immediately to dataframe_agent."""
        from app.agents.router_agent import RouterAgent
        router = RouterAgent()
        
        query = "Which companies use Python as the technical focus?"
        route_result = router.route_query(query)
        self.assertEqual(route_result["agent"], "dataframe_agent")
        self.assertIn("Tech-focus filtering query detected.", route_result["reason"])

    def test_dataframe_tech_focus_benchmark(self):
        """Verifies Example: Tech Focus Filtering Mode for the benchmark query."""
        from app.agents.dataframe_agent import DataframeAgent
        agent = DataframeAgent()
        agent.client = None
        
        query = "Which companies use Python as the technical focus?"
        response = agent.process_query(query)
        
        self.assertIn("🎯 Python-Focused Companies", response)
        self.assertIn("The following companies use Python as their technical focus:", response)
        self.assertIn("• Google", response)
        self.assertIn("• Oracle", response)
        self.assertIn("📌 Summary:", response)
        self.assertIn("2 companies in the placement dataset primarily focus on Python for technical interviews.", response)

    def test_dataframe_tech_focus_general(self):
        """Verifies dynamic tech focus filtering for a general query."""
        from app.agents.dataframe_agent import DataframeAgent
        agent = DataframeAgent()
        agent.client = None
        
        query = "Which companies focus on Java?"
        response = agent.process_query(query)
        
        self.assertIn("🎯 Java-Focused Companies", response)
        self.assertIn("The following companies use Java as their technical focus:", response)
        self.assertIn("• Infosys", response)
        self.assertIn("• Cognizant", response)
        self.assertIn("• Samsung R&D", response)
        self.assertIn("📌 Summary:", response)
        self.assertIn("3 companies in the placement dataset primarily focus on Java for technical interviews.", response)

if __name__ == "__main__":
    unittest.main()
