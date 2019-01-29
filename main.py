import argparse

from sklearn.metrics import f1_score

from core.loader import DataLoader
from core.util import timer, get_logger
from core.model import lr_model
from core.preprocessing import get_stopwords, tokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="exp")

    args = parser.parse_args()
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

    model = lr_model()
    with timer("fitting", logger):
        model.fit(X, y)

    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average="macro")
    logger.info(f"F1: {f1:.3f}")
