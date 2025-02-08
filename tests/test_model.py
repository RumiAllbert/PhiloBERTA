import unittest

from philoberta.data import CorpusLoader
from philoberta.evaluation import Evaluator
from philoberta.model import SPHILBERTA


class TestPHILOBERTA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = SPHILBERTA()
        cls.data_loader = CorpusLoader()
        cls.evaluator = Evaluator(cls.model, cls.data_loader)

    def test_embedding_dimensions(self):
        """Test if embeddings have correct dimensions"""
        test_texts = ["Test text in Greek λόγος", "Another test ψυχή"]
        embeddings = self.model.get_embeddings(test_texts)
        self.assertEqual(embeddings.shape, (2, 768))

    def test_cross_similarity_range(self):
        """Test if similarity scores are in valid range [0,1]"""
        greek_contexts = ["λόγος is important", "studying λόγος"]
        latin_contexts = ["ratio in text", "understanding ratio"]

        similarity = self.model.cross_similarity(
            "λόγος", "ratio", greek_contexts, latin_contexts
        )
        self.assertTrue(0 <= similarity <= 1)

    def test_term_pairs_validity(self):
        """Test if term pairs are properly formatted"""
        term_pairs = self.data_loader.get_term_pairs()
        for greek, latin in term_pairs:
            self.assertTrue(isinstance(greek, str))
            self.assertTrue(isinstance(latin, str))

    def test_genre_labels_coverage(self):
        """Test if genre labels cover major authors"""
        genre_labels = self.data_loader.get_genre_labels()
        essential_authors = {"plato", "aristotle", "homer", "thucydides"}
        self.assertTrue(essential_authors.issubset(genre_labels.keys()))


if __name__ == "__main__":
    unittest.main()
