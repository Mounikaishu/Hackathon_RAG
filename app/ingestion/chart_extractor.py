import os
import fitz  # PyMuPDF
from app.config import settings

class ChartExtractor:
    """
    Renders PDF pages containing visual charts or images into high-resolution PNG files.
    Allows vision agents to reason directly on the actual image assets.
    """

    def __init__(self, pdf_path: str):
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at: {pdf_path}")
        self.pdf_path = pdf_path

    def extract_chart_pages(self) -> dict:
        """
        Scans the PDF to find the page containing 'Section 3' or hiring charts,
        renders that page as a PNG image, and returns the path to the saved asset.
        """
        doc = fitz.open(self.pdf_path)
        chart_pages = []
        saved_paths = []

        # Keywords that indicate the page contains charts/graphs
        chart_keywords = ["section 3", "hiring distribution", "bar chart", "image / chart content"]

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text").lower()

            # If the page contains any chart indicators, mark it for extraction
            if any(keyword in text for keyword in chart_keywords):
                chart_pages.append(page_num)

        # Ensure directory exists
        os.makedirs(settings.CHARTS_DIR, exist_ok=True)

        for page_num in chart_pages:
            page = doc[page_num]
            
            # Double the resolution (2.0x zoom factor) for crisp text/borders
            zoom = 2.0
            mat = fitz.Matrix(zoom, zoom)
            
            # Render page to a pixmap (pixel map)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            
            # Define output filename
            output_filename = f"hiring_charts_page_{page_num + 1}.png"
            output_path = os.path.join(settings.CHARTS_DIR, output_filename)
            
            # Save the pixmap as PNG
            pix.save(output_path)
            saved_paths.append(output_path)

        doc.close()

        # If no page matched keyword search, fallback: render page 3 (standard Section 3 page)
        if not saved_paths and len(doc) >= 3:
            doc = fitz.open(self.pdf_path)
            page = doc[2]  # 0-indexed Page 3 (usually the third physical page)
            zoom = 2.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            output_path = os.path.join(settings.CHARTS_DIR, "hiring_charts_page_3.png")
            pix.save(output_path)
            saved_paths.append(output_path)
            doc.close()

        return {
            "matched_pages": chart_pages,
            "saved_images": saved_paths
        }
