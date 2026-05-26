import os
import json
import re
import fitz  # PyMuPDF
from app.config import settings

class TableExtractor:
    """
    Parses and extracts structured tables from the PDF document.
    Saves tables as structured JSON datasets for clean, 100% accurate tabular agent operations.
    """

    def __init__(self, pdf_path: str):
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at: {pdf_path}")
        self.pdf_path = pdf_path
        self.companies_list = [
            "TCS", "Infosys", "Deloitte", "Accenture", "Amazon", "Flipkart", 
            "Google", "Microsoft", "Wipro", "Cognizant", "Capgemini", "IBM", 
            "Adobe", "Oracle", "SAP", "HCL", "Tech Mahindra", "Qualcomm", 
            "Intel", "Samsung R&D", "Samsung"
        ]

    def extract_eligibility_table(self) -> list:
        """
        Extracts the Section 1 company eligibility table from the PDF.
        Tries native PyMuPDF table finder, and falls back to regex-based line scanning.
        """
        doc = fitz.open(self.pdf_path)
        extracted_rows = []

        # Step 1: Attempt native PyMuPDF table search
        for page_num in range(len(doc)):
            page = doc[page_num]
            try:
                tables = page.find_tables()
                if tables and len(tables.tables) > 0:
                    table = tables[0]
                    headers = [str(h).strip() for h in table.header.names]
                    
                    # Verify this is indeed the eligibility table (checking for 'CGPA' or 'Package')
                    if any("cgpa" in h.lower() or "package" in h.lower() for h in headers):
                        table_data = table.extract()
                        
                        # Process table rows (skipping header)
                        for r_idx, row in enumerate(table_data):
                            if r_idx == 0:
                                continue # Skip header
                            
                            # Clean up cells
                            cleaned_row = [str(cell).strip() if cell else "" for cell in row]
                            if not cleaned_row or not cleaned_row[0]:
                                continue
                            
                            # Map row to dictionary structure
                            entry = self._map_row_to_dict(cleaned_row)
                            if entry:
                                extracted_rows.append(entry)
                        
                        if len(extracted_rows) > 0:
                            break  # Found and processed table!
            except Exception as e:
                # Log table extraction warning and proceed to fallback
                pass

        # Step 2: Fallback to high-precision Regex Row Scanner
        if not extracted_rows:
            full_text = ""
            for page in doc:
                full_text += page.get_text("text") + "\n"
            
            extracted_rows = self._regex_scan_text(full_text)

        doc.close()

        # Step 3: Save to JSON
        os.makedirs(settings.PROCESSED_DIR, exist_ok=True)
        output_path = os.path.join(settings.PROCESSED_DIR, "eligibility_table.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(extracted_rows, f, indent=2, ensure_ascii=False)

        return extracted_rows

    def _map_row_to_dict(self, row: list) -> dict:
        """Helper to convert list of cell strings to typed dictionary."""
        if len(row) < 4:
            return None
            
        company = row[0]
        # Skip header or noise lines
        if "company" in company.lower():
            return None
            
        try:
            # Parse floats and ints safely, removing non-numeric characters (like LPA, Yrs)
            cgpa = float(re.findall(r"\d+\.?\d*", row[1])[0])
            backlogs = int(re.findall(r"\d+", row[2])[0])
            package = float(re.findall(r"\d+\.?\d*", row[3])[0])
            bond = int(re.findall(r"\d+", row[4])[0]) if len(row) > 4 and re.findall(r"\d+", row[4]) else 0
            
            key_topics = row[5] if len(row) > 5 else ""
            tech_focus = row[6] if len(row) > 6 else ""

            return {
                "company": company,
                "min_cgpa": cgpa,
                "max_backlogs": backlogs,
                "package_lpa": package,
                "bond_years": bond,
                "key_topics": key_topics.strip(),
                "tech_focus": tech_focus.strip()
            }
        except Exception:
            return None

    def _regex_scan_text(self, text: str) -> list:
        """Regex-based line scanner fallback for unstructured layouts."""
        rows = []
        lines = text.split("\n")
        
        # Regex pattern matching: Company, Float (CGPA), Int (Backlogs), Float (Package), Int (Bond)
        # Followed by remaining texts
        row_regex = r"^([A-Za-z0-9&\s\.\;\-\_]+?)\s+(\d+\.\d+|\d+)\s+(\d+)\s+(\d+\.\d+|\d+)\s+(\d+)\s+(.+)$"

        for line in lines:
            line_cleaned = line.strip()
            # If line starts with a known company or resembles eligibility criteria
            if any(line_cleaned.startswith(c) for c in self.companies_list):
                match = re.match(row_regex, line_cleaned)
                if match:
                    comp = match.group(1).strip()
                    cgpa = float(match.group(2))
                    backlogs = int(match.group(3))
                    package = float(match.group(4))
                    bond = int(match.group(5))
                    
                    remaining = match.group(6).strip()
                    # Splitting topics and tech focus (usually delimited or end words)
                    # Let's check if there's a comma or split it by common tech words
                    key_topics = remaining
                    tech_focus = ""
                    
                    # Split topics and focus intelligently
                    if "DSA" in remaining:
                        parts = remaining.split("DSA")
                        key_topics = "DSA" + parts[1] if len(parts) > 1 else "DSA"
                        tech_focus = parts[0].strip() if parts[0].strip() else "DSA"
                        
                        # Swap if needed
                        if any(focus in key_topics for focus in ["Java", "Python", "C++", "System Design"]):
                            # Re-arrange
                            for f in ["Java", "Python", "C++", "System Design", "Cloud"]:
                                if f in key_topics:
                                    tech_focus = f
                                    key_topics = key_topics.replace(f, "").strip(", ").strip()
                                    break
                    
                    # Tidy up company name if extra noise was matched
                    for known in self.companies_list:
                        if comp.lower().replace(" ", "") == known.lower().replace(" ", ""):
                            comp = known
                            break

                    rows.append({
                        "company": comp,
                        "min_cgpa": cgpa,
                        "max_backlogs": backlogs,
                        "package_lpa": package,
                        "bond_years": bond,
                        "key_topics": key_topics.strip(", "),
                        "tech_focus": tech_focus.strip(", ")
                    })
        return rows

    def extract_trends_table(self) -> list:
        """
        Extracts Section 5 Placement Trend data from the PDF.
        Saves as trends_table.json.
        """
        doc = fitz.open(self.pdf_path)
        extracted_rows = []
        
        # Only parse page containing Section 5 trends table
        trends_text = ""
        for page in doc:
            p_text = page.get_text("text")
            if "Section 5:" in p_text and ("Placement Trend" in p_text or "Temporal Trends" in p_text):
                trends_text += p_text + "\n"
                
        lines = [line.strip() for line in trends_text.split("\n") if line.strip()]
        
        # Parse vertical layout
        i = 0
        while i < len(lines):
            line = lines[i]
            matched_company = None
            for comp in self.companies_list:
                if line.lower() == comp.lower():
                    matched_company = comp
                    break
            
            if matched_company and i + 5 < len(lines):
                try:
                    pkg_2021 = float(lines[i+1])
                    pkg_2022 = float(lines[i+2])
                    pkg_2023 = float(lines[i+3])
                    pkg_2024 = float(lines[i+4])
                    trend_text = lines[i+5]
                    
                    extracted_rows.append({
                        "company": matched_company,
                        "pkg_2021": pkg_2021,
                        "pkg_2022": pkg_2022,
                        "pkg_2023": pkg_2023,
                        "pkg_2024": pkg_2024,
                        "trend": trend_text
                    })
                    i += 6
                    continue
                except ValueError:
                    pass
            i += 1
            
        # Fallback to regex horizontal scanner if no vertical match
        if not extracted_rows:
            for line in lines:
                for comp in self.companies_list:
                    if line.startswith(comp):
                        match = re.match(
                            r"^([A-Za-z0-9&\s\.\;\-\_]+?)\s+(\d+\.\d+|\d+)\s+(\d+\.\d+|\d+)\s+(\d+\.\d+|\d+)\s+(\d+\.\d+|\d+)\s+(.+)$",
                            line
                        )
                        if match:
                            extracted_rows.append({
                                "company": comp,
                                "pkg_2021": float(match.group(2)),
                                "pkg_2022": float(match.group(3)),
                                "pkg_2023": float(match.group(4)),
                                "pkg_2024": float(match.group(5)),
                                "trend": match.group(6).strip()
                            })
                            break
        
        doc.close()
        
        # Save to JSON
        os.makedirs(settings.PROCESSED_DIR, exist_ok=True)
        output_path = os.path.join(settings.PROCESSED_DIR, "trends_table.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(extracted_rows, f, indent=2, ensure_ascii=False)
            
        return extracted_rows

