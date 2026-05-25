class EligibilityChunker:
    """
    Transforms structured company eligibility records into natural language chunks
    with rich search-ready metadata.
    """

    def chunk_eligibility_records(self, records: list) -> list:
        """
        Processes a list of structured company profiles and yields a list of chunk dictionaries
        containing 'text' and 'metadata' keys.
        """
        chunks = []
        for record in records:
            company = record.get("company", "Unknown")
            cgpa = record.get("min_cgpa", 0.0)
            backlogs = record.get("max_backlogs", 0)
            package = record.get("package_lpa", 0.0)
            bond = record.get("bond_years", 0)
            topics = record.get("key_topics", "")
            focus = record.get("tech_focus", "")

            # Create standard descriptive narrative text
            text = (
                f"Company Eligibility Profile: {company}.\n"
                f"Minimum CGPA Cutoff Requirement: {cgpa}.\n"
                f"Maximum Active Backlogs Allowed: {backlogs}.\n"
                f"Offered Placement Package: {package} LPA (Lakhs Per Annum).\n"
                f"Service Agreement Bond: {bond} years.\n"
                f"Primary Technical Interview Topics: {topics}.\n"
                f"Core Technology Stack Focus: {focus}."
            )

            # Metadata for semantic filtering
            metadata = {
                "section": "eligibility",
                "company": company,
                "min_cgpa": float(cgpa),
                "max_backlogs": int(backlogs),
                "package_lpa": float(package),
                "bond_years": int(bond),
                "tech_focus": focus
            }

            chunks.append({
                "text": text,
                "metadata": metadata
            })

        return chunks
