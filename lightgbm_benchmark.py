import argparse

import numpy as np
import lightgbm as lgb

from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer as Tfidf

from core.loader import DataLoader
from core.preprocessing import get_stopwords, tokenizer
from core.util import timer, get_logger

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
    X = train["tokenized"].fillna("")
    X_test = test["tokenized"].fillna("")
    y = train["label"].values
    y_test = test["label"].values

    with timer("vectorize", logger):
        tv = Tfidf(max_features=20000, ngram_range=(1, 3))
        X = tv.fit_transform(X)
        X_test = tv.transform(X_test)

    params = {
        "objective": "multiclass",
        "boosting_type": "gbdt",
        "n_estimators": 2000,
        "learning_rate": 0.023354457787945405,
        "num_leaves": 580,
        "subsample_for_bin": 181455,
        "min_split_gain": 0.04339612875775048,
        "min_child_samples": 9,
        "min_child_weight": 0.1769382985126983,
        "subsample": 0.5469632221611453,
        "subsample_freq": 3,
        "colsample_bytree": 0.5248533394278168,
        "reg_alpha": 0.16194939488762344,
        "reg_lambda": 0.11920816987519009,
        "max_bin": 656,
        "n_jobs": -1
    }

    with timer("fitting", logger):
        models = []
        fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for (trn_index, val_index) in fold.split(X, y):
            X_train, X_val = X[trn_index], X[val_index]
            y_train, y_val = y[trn_index], y[val_index]
            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_train,
                y_train,
                eval_set=(X_val, y_val),
                early_stopping_rounds=50,
                verbose=100)
            models.append(model)
    y_pred = np.zeros((X_test.shape[0], len(loader.classes)))
    for m in models:
        y_pred += m.predict_proba(X_test) / 5
    f1 = f1_score(y_test, np.argmax(y_pred, axis=1), average="macro")
    logger.info(f"F1: {f1:.3f}")
