import re
from app.ingestion.deduplicator import Deduplicator

class InterviewChunker:
    """
    Parses, cleans, deduplicates, and chunks the interview experiences section.
    Yields highly focused rounds and tips chunks with precise metadata.
    """

    def __init__(self, similarity_threshold: float = 0.85):
        self.deduplicator = Deduplicator(similarity_threshold)
        self.companies_list = ["TCS", "Infosys", "Deloitte", "Accenture", "Amazon", "Flipkart", "Google", "Microsoft", "Wipro", "Cognizant", "Capgemini", "IBM", "Adobe", "Oracle", "SAP", "HCL", "Tech Mahindra", "Qualcomm", "Intel", "Samsung R&D"]

    def chunk_interview_experiences(self, section_text: str) -> list:
        """
        Parses the raw Section 2 text, groups by company, deduplicates content,
        and constructs granular chunks.
        """
        chunks = []
        
        # Split text by company headers
        # e.g., '■ TCS | Technical Focus: System Design'
        header_regex = r"(■\s+([A-Za-z0-9&\s\.\;\-\_]+?)\s*\|\s*Technical Focus:\s*([^\n]+))"
        matches = list(re.finditer(header_regex, section_text))
        
        if not matches:
            # If no matches, fall back to basic line-by-line semantic chunks
            return self._basic_chunking(section_text)

        for i, match in enumerate(matches):
            header_text = match.group(1)
            company_name = match.group(2).strip()
            tech_focus = match.group(3).strip()
            
            # Extract content text block for current company
            start_idx = match.end()
            end_idx = matches[i + 1].start() if i + 1 < len(matches) else len(section_text)
            company_block = section_text[start_idx:end_idx].strip()
            
            # Deduplicate repeated text patterns inside the block
            # In this dataset, experiences are repeated 3x.
            # We split by 'Round' or 'Tip' to isolate the individual sections.
            raw_segments = []
            
            # Find all Round details (e.g. 'Round 1 Details...')
            round_pattern = r"(Round\s+\d+\s*.*?)(?=Round\s+\d+|Tip:|■|$)"
            rounds_found = re.findall(round_pattern, company_block, re.DOTALL)
            raw_segments.extend(rounds_found)
            
            # Find all Tips (e.g. 'Tip: Strong DSA...')
            tip_pattern = r"(Tip:\s*.*?)(?=Round\s+\d+|Tip:|■|$)"
            tips_found = re.findall(tip_pattern, company_block, re.DOTALL)
            raw_segments.extend(tips_found)

            # Clean and deduplicate the raw segments using the Deduplicator
            deduplicated_segments = self.deduplicator.deduplicate_experiences(raw_segments)
            
            # Convert deduplicated segments into structured chunks
            for segment in deduplicated_segments:
                segment_cleaned = segment.strip()
                is_tip = segment_cleaned.lower().startswith("tip")
                
                # Determine round number if applicable
                round_num = None
                if not is_tip:
                    round_match = re.search(r"Round\s+(\d+)", segment_cleaned, re.IGNORECASE)
                    if round_match:
                        round_num = int(round_match.group(1))

                # Structure semantic text representation
                chunk_text = (
                    f"Company: {company_name}\n"
                    f"Interview Technical Focus: {tech_focus}\n"
                    f"Content Details: {segment_cleaned}"
                )

                metadata = {
                    "section": "interview",
                    "company": company_name,
                    "tech_focus": tech_focus,
                    "is_tip": is_tip,
                    "round_number": round_num if round_num else -1
                }

                chunks.append({
                    "text": chunk_text,
                    "metadata": metadata
                })

        return chunks

    def _basic_chunking(self, text: str) -> list:
        """Fallback chunker if structured regex split fails."""
        chunks = []
        paragraphs = text.split("\n\n")
        
        for para in paragraphs:
            para_cleaned = para.strip()
            if len(para_cleaned) > 20:
                chunks.append({
                    "text": para_cleaned,
                    "metadata": {"section": "interview_general"}
                })
        return chunks
