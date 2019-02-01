from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression as LR
from sklearn.feature_extraction.text import TfidfTransformer as Tfidf
from sklearn.feature_extraction.text import CountVectorizer as Count


def lr_model():
    steps = [("vec", Count(max_features=20000, ngram_range=(1, 3))),
             ("tfidf", Tfidf()),
             ("clf",
              LR(penalty="l2",
                 solver="saga",
                 multi_class="auto",
                 random_state=42,
                 max_iter=1e4,
                 n_jobs=-1,
                 C=9.930234540337622,
                 tol=0.0007738614855921237))]
    return Pipeline(steps=steps)
