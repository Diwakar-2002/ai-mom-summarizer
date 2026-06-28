import unittest
from metrics import (
    compute_rouge,
    compute_bleu,
    compute_meteor,
    compute_bertscore,
    compute_semantic_cosine,
    compute_factual_consistency,
    compute_compression_ratio,
    compute_action_item_metrics
)

class TestMetrics(unittest.TestCase):
    def test_compute_rouge(self):
        ref = "The quick brown fox jumps over the lazy dog."
        cand = "The quick brown fox jumps over the lazy dog."
        scores = compute_rouge(ref, cand)
        self.assertAlmostEqual(scores["rouge1"], 1.0)
        self.assertAlmostEqual(scores["rouge2"], 1.0)
        self.assertAlmostEqual(scores["rougeL"], 1.0)
        
        empty_scores = compute_rouge("", "")
        self.assertEqual(empty_scores["rouge1"], 0.0)

    def test_compute_bleu(self):
        ref = "The quick brown fox jumps over the lazy dog."
        cand = "The quick brown fox jumps over the lazy dog."
        score = compute_bleu(ref, cand)
        self.assertGreater(score, 0.9)
        
        score_empty = compute_bleu("", "")
        self.assertEqual(score_empty, 0.0)

    def test_compute_meteor(self):
        ref = "The quick brown fox jumps over the lazy dog."
        cand = "The quick brown fox jumps over the lazy dog."
        score = compute_meteor(ref, cand)
        self.assertGreater(score, 0.9)
        
        score_empty = compute_meteor("", "")
        self.assertEqual(score_empty, 0.0)

    def test_compute_bertscore(self):
        ref = "This is a test summary."
        cand = "This is a test summary."
        scores = compute_bertscore(ref, cand)
        self.assertGreater(scores["bertscore_f1"], 0.8)

    def test_compute_semantic_cosine(self):
        ref = "We discussed the budget of Springfield Road."
        cand = "The budget for Springfield Road was discussed."
        score = compute_semantic_cosine(ref, cand)
        self.assertGreater(score, 0.7)

    def test_compute_factual_consistency(self):
        transcript = "The council approved a budget of $2500 for the election day official rates."
        consistent_summary = "The council approved election rates."
        inconsistent_summary = "The council approved a $5000 budget for the election."
        
        score_consistent = compute_factual_consistency(transcript, consistent_summary)
        score_inconsistent = compute_factual_consistency(transcript, inconsistent_summary)
        
        print(f"Consistent score: {score_consistent}")
        print(f"Inconsistent score: {score_inconsistent}")
        
        self.assertIn("factual_consistency", score_consistent)
        self.assertIn("method", score_consistent)
        self.assertGreater(score_consistent["factual_consistency"], score_inconsistent["factual_consistency"])

    def test_compute_compression_ratio(self):
        trans = "This is a very long transcript that goes on and on."
        summ = "Short summary."
        ratio = compute_compression_ratio(trans, summ)
        self.assertEqual(ratio, len(trans) / len(summ))

    def test_compute_action_item_metrics(self):
        self.assertEqual(compute_action_item_metrics([], [])["action_item_f1"], 1.0)
        
        ref = [
            {"action": "Review the long term debt reports", "owner": "CFO", "deadline": "June 16"},
            {"action": "Prepare print copy of the revised zoning bylaw", "owner": "Office", "deadline": "None"}
        ]
        
        self.assertEqual(compute_action_item_metrics(ref, ref)["action_item_f1"], 1.0)
        
        cand = [
            {"action": "Examine the reports for long term debt", "owner": "CFO", "deadline": "June 16"},
            {"action": "Make print copy of new bylaw", "owner": "Office", "deadline": "None"}
        ]
        
        scores = compute_action_item_metrics(ref, cand, threshold=0.6)
        self.assertGreaterEqual(scores["action_item_f1"], 0.8)

if __name__ == "__main__":
    unittest.main()
