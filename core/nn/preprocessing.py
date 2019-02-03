import MeCab
import mojimoji

tagger = MeCab.Tagger(
    "-Ochasen -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd")


def tokenizer(text):
    text = mojimoji.zen_to_han(text.replace("\n", ""), kana=False)
    parsed = tagger.parse(text).split("\n")
    parsed = [t.split("\t") for t in parsed]
    parsed = list(filter(lambda x: x[0] != "" and x[0] != "EOS", parsed))
    parsed = [p[2] for p in parsed]
    return parsed
