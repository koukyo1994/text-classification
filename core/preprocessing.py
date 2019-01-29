import re
import urllib.request

import MeCab
import mojimoji

from pathlib import Path

tagger = MeCab.Tagger(
    "-Ochasen -d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd")


def get_stopwords(save_dir="input/") -> set:
    path = Path(save_dir) / "stopwords.txt"
    if path.exists():
        with open(path) as f:
            stopwords = f.read().split("\n")
        return set(stopwords)
    url = "http://svn.sourceforge.jp/svnroot/slothlib/CSharp/Version1/SlothLib/NLP/Filter/StopWord/word/Japanese.txt"
    stopwords = urllib.request.urlopen(url).read().decode("utf8")
    with open(path, "w") as f:
        f.write(stopwords)
    return set(stopwords.split("\n"))


def tokenizer(x: str, stopwords: set, include_verb=True) -> str:
    text = mojimoji.zen_to_han(x.replace("\n", ""), kana=False)
    parsed = tagger.parse(text).split("\n")
    parsed = [t.split("\t") for t in parsed]
    parsed = list(
        filter(
            lambda x: '助詞' not in x[3] and '記号' not in x[3] and '助動詞' not in x[3],
            filter(lambda x: x[0] != '' and x[0] != 'EOS', parsed)))
    if not include_verb:
        parsed = list(filter(lambda x: "動詞" not in x[3], parsed))
    parsed = [p[2] for p in parsed if not re.match("\d", p[2])]
    return " ".join(list(filter(lambda x: x not in stopwords, parsed)))
