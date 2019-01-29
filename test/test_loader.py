import pandas as pd

from unittest import TestCase

from core.loader import DataLoader
from core.preprocessing import get_stopwords, tokenizer


class TestDataLoader(TestCase):
    def setUp(self):
        self.loader = DataLoader()

    def test_load_data(self):
        self.assertEqual(9, len(self.loader.classes))
        self.assertEqual(len(self.loader.classes), len(self.loader.dir))
        self.assertEqual(
            len(self.loader.classes), len(self.loader.encode_dict))
        self.assertEqual(
            len(self.loader.encode_dict), len(self.loader.decode_dict))

        self.assertEqual(len(self.loader.title), len(self.loader.url))
        self.assertEqual(len(self.loader.url), len(self.loader.text))
        self.assertEqual(len(self.loader.text), len(self.loader.datetime))
        self.assertEqual(len(self.loader.datetime), len(self.loader.label))

    def test_tokenize_and_tokenize(self):
        stopwords = get_stopwords()
        self.loader.tokenize(tokenizer, {
            "stopwords": stopwords,
            "include_verb": True
        })
        self.assertIn("tokenized", self.loader.data.columns)
        self.assertTrue(self.loader.tokenized)

        train, test = self.loader.load()
        self.assertIsInstance(train, pd.DataFrame)
        self.assertIsInstance(test, pd.DataFrame)
