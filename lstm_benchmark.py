import argparse

from core.loader import DataLoader
from core.util import timer, get_logger
from core.nn.preprocessing import tokenizer
from core.nn.util import to_sequence, load_w2v
from core.nn.model import train_and_validate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="exp")
    parser.add_argument("--embedding")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n_epochs", default=10, type=int)

    args = parser.parse_args()
    assert args.embedding

    logger = get_logger(exp=args.exp)
    with timer("Load Data", logger):
        loader = DataLoader()

    with timer("tokenize", logger):
        loader.tokenize(tokenizer)

    train, test = loader.load()
    X = train["tokenized"]
    X_test = test["tokenized"]
    y = train["label"]
    y_test = test["label"]

    with timer("Convert to sequence", logger):
        X, X_test, word_index = to_sequence(X, X_test, max_features=80000)

    with timer("Load embedding", logger):
        embedding_matrix = load_w2v(word_index, args.embedding, 80000)
    train_and_validate(
        X,
        y,
        X_test,
        y_test,
        embedding_matrix,
        logger,
        args.n_epochs,
        device=args.device)
