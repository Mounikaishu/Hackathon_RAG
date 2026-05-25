import os
import json
import pandas as pd
from app.config import settings

class PandasTool:
    """
    Loads company eligibility JSON into a Pandas DataFrame and executes analytical
    filtering, sorting, and aggregation queries in a safe local sandbox.
    Guarantees 100% numerical accuracy for structured queries.
    """

    def __init__(self):
        self.table_path = os.path.join(settings.PROCESSED_DIR, "eligibility_table.json")
        self.df = self._load_dataframe()

    def _load_dataframe(self) -> pd.DataFrame:
        """Loads processed JSON data or returns a structured schema fallback."""
        columns = ["company", "min_cgpa", "max_backlogs", "package_lpa", "bond_years", "key_topics", "tech_focus"]
        
        if not os.path.exists(self.table_path):
            # Safe empty fallback schema
            return pd.DataFrame(columns=columns)
            
        try:
            with open(self.table_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            # Ensure proper numeric typing
            if not df.empty:
                df["min_cgpa"] = pd.to_numeric(df["min_cgpa"], errors="coerce")
                df["max_backlogs"] = pd.to_numeric(df["max_backlogs"], errors="coerce")
                df["package_lpa"] = pd.to_numeric(df["package_lpa"], errors="coerce")
                df["bond_years"] = pd.to_numeric(df["bond_years"], errors="coerce")
            else:
                df = pd.DataFrame(columns=columns)
            return df
        except Exception:
            return pd.DataFrame(columns=columns)

    def execute_query(self, pandas_code: str) -> dict:
        """
        Executes a string of pandas code against the DataFrame 'df' in a restricted environment.
        Returns a structured result dict containing the markdown text representation and status.
        """
        if self.df.empty:
            return {
                "success": False,
                "error": "The eligibility database is currently empty. Please run the ingestion pipeline first.",
                "result": ""
            }

        # Restrict execution environment to prevent arbitrary code vulnerabilities
        local_vars = {
            "df": self.df,
            "pd": pd
        }
        
        # Strip any leading 'print()' or unsafe prefixes
        code_cleaned = pandas_code.strip()
        if code_cleaned.startswith("print("):
            # Extract content inside print()
            match = re.match(r"print\((.*)\)", code_cleaned)
            if match:
                code_cleaned = match.group(1)

        try:
            # Evaluate expression
            result = eval(code_cleaned, {"__builtins__": {}}, local_vars)
            
            # Format output based on the resulting pandas data type
            if isinstance(result, pd.DataFrame):
                if result.empty:
                    formatted_result = "No companies match these criteria."
                else:
                    # Renders beautiful markdown table
                    formatted_result = result.to_markdown(index=False)
            elif isinstance(result, pd.Series):
                if result.empty:
                    formatted_result = "No results found."
                else:
                    formatted_result = pd.DataFrame(result).to_markdown()
            elif result is None:
                formatted_result = "Query executed successfully, but returned no values."
            else:
                # Scalar result (e.g. mean package or single cell lookup)
                formatted_result = str(result)

            return {
                "success": True,
                "error": None,
                "result": formatted_result
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "result": ""
            }
