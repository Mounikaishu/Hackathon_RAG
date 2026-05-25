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

if __name__ == "__main__":
    unittest.main()
