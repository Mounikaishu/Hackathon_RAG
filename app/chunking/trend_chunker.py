import re

class TrendChunker:
    """
    Chunks and structures time-series / chronological placement narrative data.
    Extracts chronological year tokens and saves them in chunk metadata to support high-performance temporal reasoning queries.
    """

    def chunk_trend_data(self, text: str) -> list:
        """
        Splits trend narrative text by paragraphs, extracts mentioned years,
        and creates metadata-enriched chunks.
        """
        chunks = []
        paragraphs = text.split("\n\n")

        for para in paragraphs:
            para_cleaned = para.strip()
            if len(para_cleaned) < 25:
                continue

            # Regex to match years (e.g., 2023, 2024, 2025, 2026)
            year_matches = re.findall(r"\b(20\d{2})\b", para_cleaned)
            # Deduplicate list of years
            years = sorted(list(set(int(y) for y in year_matches)))

            # Deduce company mentions
            companies_mentioned = []
            for word in para_cleaned.split():
                # Clean up word punctuation
                word_clean = re.sub(r"[^\w]", "", word)
                if word_clean in ["TCS", "Infosys", "Deloitte", "Accenture", "Amazon", "Flipkart", "Google", "Microsoft", "Wipro", "Cognizant"]:
                    companies_mentioned.append(word_clean)
            companies_mentioned = list(set(companies_mentioned))

            # Structure text chunk
            metadata = {
                "section": "trends",
                "years": years,
                "companies": companies_mentioned,
                "temporal": len(years) > 0
            }

            chunks.append({
                "text": para_cleaned,
                "metadata": metadata
            })

        return chunks
