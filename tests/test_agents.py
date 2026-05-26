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

if __name__ == "__main__":
    unittest.main()


