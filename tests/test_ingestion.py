import unittest
from app.ingestion.table_extractor import TableExtractor
from app.ingestion.deduplicator import Deduplicator

class TestIngestion(unittest.TestCase):
    """
    Unit tests to verify data loading, table parsing helpers,
    and fuzzy deduplication algorithms under ingestion modules.
    """

    def setUp(self):
        # Instantiate with a dummy path (no file is read during functional mock checks)
        self.table_extractor = TableExtractor("dummy.pdf")
        self.deduplicator = Deduplicator(similarity_threshold=0.85)

    def test_map_row_to_dict_valid(self):
        """Verifies that eligibility row mapper converts string cells into properly typed fields."""
        raw_row = ["Google", "7.4", "0", "42.0 LPA", "1 Yrs", "DSA, Algorithms", "Python"]
        result = self.table_extractor._map_row_to_dict(raw_row)
        
        self.assertIsNotNone(result)
        self.assertEqual(result["company"], "Google")
        self.assertEqual(result["min_cgpa"], 7.4)
        self.assertEqual(result["max_backlogs"], 0)
        self.assertEqual(result["package_lpa"], 42.0)
        self.assertEqual(result["bond_years"], 1)
        self.assertEqual(result["key_topics"], "DSA, Algorithms")
        self.assertEqual(result["tech_focus"], "Python")

    def test_map_row_to_dict_invalid(self):
        """Verifies that invalid rows (like header rows) return None instead of crashing."""
        header_row = ["Company", "Min CGPA", "Max Backlogs", "Package (LPA)"]
        result = self.table_extractor._map_row_to_dict(header_row)
        self.assertNil = self.assertIsNone(result)

    def test_deduplicator_similarity(self):
        """Verifies that similar text blocks get mapped to high similarity scores."""
        text_a = "Round 1 Online Assessment: Aptitude + Logical Reasoning + DSA coding (HackerRank, 90 min)"
        text_b = "Round 1 Online Assessment: Aptitude + Logical Reasoning + DSA coding (HackerRank, 90 min) " # trailing space
        text_c = "Round 3 Managerial/HR: Behavioural questions, teamwork scenarios, Why TCS?"
        
        score_similar = self.deduplicator.calculate_similarity(text_a, text_b)
        score_different = self.deduplicator.calculate_similarity(text_a, text_c)
        
        self.assertGreaterEqual(score_similar, 0.95)
        self.assertLess(score_different, 0.50)

    def test_deduplicate_experiences_list(self):
        """Asserts that identical repeated text segments are scrubbed down to a single instance."""
        repeated_list = [
            "Round 1: DSA coding problem",
            "Round 1: DSA coding problem",
            "Round 1: DSA coding problem",
            "Round 2: System Design interview",
            "Round 2: System Design interview"
        ]
        
        cleaned = self.deduplicator.deduplicate_experiences(repeated_list)
        self.assertEqual(len(cleaned), 2)
        self.assertEqual(cleaned[0], "Round 1: DSA coding problem")
        self.assertEqual(cleaned[1], "Round 2: System Design interview")

if __name__ == "__main__":
    unittest.main()
