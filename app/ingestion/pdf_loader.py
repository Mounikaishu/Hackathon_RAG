import os
import fitz  # PyMuPDF
import re

class PDFLoader:
    """
    Utility class to load and parse text from a PDF file using PyMuPDF.
    Partitions the text by page and automatically splits it into major semantic sections.
    """

    def __init__(self, pdf_path: str):
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at: {pdf_path}")
        self.pdf_path = pdf_path

    def load_raw_data(self) -> dict:
        """
        Loads the PDF and returns a structured dictionary of page texts and sections.
        """
        doc = fitz.open(self.pdf_path)
        pages_text = []
        full_text = ""

        # Extract text page by page
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            pages_text.append(text)
            full_text += f"\n--- Page {page_num + 1} ---\n{text}"

        doc.close()

        # Parse sections based on header patterns
        sections = self._parse_sections(full_text)

        return {
            "total_pages": len(pages_text),
            "pages": pages_text,
            "full_text": full_text,
            "sections": sections
        }

    def _parse_sections(self, text: str) -> dict:
        """
        Splits the text into logical sections based on the SVECW dataset headers.
        """
        sections = {}
        
        # Regex to match Section headings (e.g., Section 1: Company Eligibility Profiles)
        section_pattern = r"(Section\s+\d+:\s+[^\n]+)"
        
        # Find all section headers and their text spans
        matches = list(re.finditer(section_pattern, text))
        
        if not matches:
            # Fallback if no specific sections are identified, save under 'general'
            sections["general"] = text
            return sections

        for i, match in enumerate(matches):
            section_title = match.group(1).strip()
            # Normalize title to a simpler key (e.g. 'section_1')
            section_key = section_title.lower()
            section_key = re.sub(r'[^a-z0-9_:\s]', '', section_key).replace(' ', '_')

            # Extract text from current match to the next match
            start_idx = match.end()
            end_idx = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            section_content = text[start_idx:end_idx].strip()
            
            sections[section_key] = {
                "title": section_title,
                "content": section_content
            }
            
        return sections
