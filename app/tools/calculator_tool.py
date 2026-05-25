import re

class CalculatorTool:
    """
    Safely evaluates basic algebraic mathematical expressions.
    Filters input to ensure only numbers and standard mathematical operators are parsed.
    """

    def calculate(self, expression: str) -> str:
        """
        Validates the math expression and evaluates it.
        Returns the computed result as a rounded string or a safe error message.
        """
        # Remove whitespace
        cleaned = expression.replace(" ", "")

        # Allow ONLY numbers, decimal points, parentheses, and standard operators
        # +, -, *, /, %, **, (,)
        safe_pattern = r"^[0-9\+\-\*\/\%\.\(\)\*]+$"

        if not re.match(safe_pattern, cleaned):
            return "⚠️ Math Error: Input contains invalid or unsafe alphabetical/special characters."

        # Double check to prevent double-asterisk issues (e.g. infinite exponent loops)
        if "**" in cleaned and len(cleaned) > 15:
             return "⚠️ Math Error: Power calculations are limited to 15 characters to prevent timeouts."

        try:
            # Evaluate within a zero-builtin environment
            result = eval(cleaned, {"__builtins__": {}}, {})
            
            # Format float nicely
            if isinstance(result, float):
                # Round to 4 decimal points
                result = round(result, 4)
                
            return f"Result: {result}"
        except ZeroDivisionError:
            return "⚠️ Math Error: Division by zero is undefined."
        except Exception as e:
            return f"⚠️ Math Error: Invalid mathematical syntax: {str(e)}"
