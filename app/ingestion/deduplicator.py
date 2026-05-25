import re
from difflib import SequenceMatcher

class Deduplicator:
    """
    Cleanses and deduplicates textual data.
    Identifies repeated near-identical chunks (e.g. interview experiences repeated 3x)
    using high-performance local SequenceMatcher scoring to prevent vector database pollution.
    """

    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold

    def calculate_similarity(self, a: str, b: str) -> float:
        """Computes a normalized similarity score [0.0 - 1.0] between two strings."""
        return SequenceMatcher(None, a, b).ratio()

    def clean_text(self, text: str) -> str:
        """Sanitizes text by removing layout line-breaks, extra spacing, and formatting noise."""
        if not text:
            return ""
        # Remove consecutive newlines/whitespaces
        cleaned = re.sub(r'\s+', ' ', text)
        # Remove markdown grid/table noise if any
        cleaned = re.sub(r'\|+', ' ', cleaned)
        return cleaned.strip()

    def deduplicate_experiences(self, raw_chunks: list) -> list:
        """
        Takes a list of raw text chunks (e.g. interview rounds, tips) and
        returns a deduplicated list containing only distinct, clean chunks.
        """
        deduplicated = []

        for raw_chunk in raw_chunks:
            chunk_cleaned = self.clean_text(raw_chunk)
            if not chunk_cleaned or len(chunk_cleaned) < 15:
                continue # Skip empty or noise lines

            # Check if this chunk is highly similar to any chunk already accepted
            is_duplicate = False
            for accepted in deduplicated:
                score = self.calculate_similarity(chunk_cleaned, accepted)
                if score >= self.similarity_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                deduplicated.append(chunk_cleaned)

        return deduplicated

    def group_and_deduplicate(self, company_experience_map: dict) -> dict:
        """
        Takes a dictionary of {company: [list_of_experiences]} and
        deduplicates experiences for each company separately.
        """
        cleaned_map = {}
        for company, experiences in company_experience_map.items():
            cleaned_list = self.deduplicate_experiences(experiences)
            cleaned_map[company] = cleaned_list
        return cleaned_map
