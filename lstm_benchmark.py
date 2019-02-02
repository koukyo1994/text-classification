import argparse

from core.loader import DataLoader
from core.util import timer, get_logger
from core.preprocessing import get_stopwords, tokenizer
from core.nn.util import to_sequence, load_w2v
from core.nn.model import train_and_validate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="exp")
    parser.add_argument("--embedding")

    args = parser.parse_args()
    assert args.embedding

    logger = get_logger(exp=args.exp)
    with timer("Load Data", logger):
        loader = DataLoader()

    with timer("tokenize", logger):
        loader.tokenize(tokenizer, {
            "stopwords": get_stopwords(),
            "include_verb": True
        })

    train, test = loader.load()
    X = train["tokenized"]
    X_test = test["tokenized"]
    y = train["label"]
    y_test = test["label"]

    with timer("Convert to sequence", logger):
        X, X_test, word_index = to_sequence(X, X_test)

    with timer("Load embedding", logger):
        embedding_matrix = load_w2v(word_index, args.embedding, 95000)
    train_and_validate(X, y, X_test, y_test, embedding_matrix, logger)
