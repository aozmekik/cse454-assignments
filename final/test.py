import re
from trstop import trstop
import string
from typing import List
from jpype import JClass, JString, getDefaultJVMPath, shutdownJVM, startJVM, java
from examples import DATA_PATH, ZEMBEREK_PATH
from pathlib import Path

startJVM(getDefaultJVMPath(), '-ea',
         '-Djava.class.path=%s' % (ZEMBEREK_PATH))

TurkishMorphology: JClass = JClass('zemberek.morphology.TurkishMorphology')
TurkishSentenceNormalizer: JClass = JClass(
    'zemberek.normalization.TurkishSentenceNormalizer'
)


Paths: JClass = JClass('java.nio.file.Paths')


def stem(text: str) -> str:
    morphology = TurkishMorphology.createWithDefaults()

    analysis: java.util.ArrayList = (
        morphology.analyzeAndDisambiguate(text).bestAnalysis()
    )

    pos: List[str] = []
    for i, analysis in enumerate(analysis, start=1):
        pos.append(
            f'{str(analysis.getLemmas()[0])}'
        )
    return ' '.join(pos)


def normalize(text: str) -> str:

    normalizer = TurkishSentenceNormalizer(
        TurkishMorphology.createWithDefaults(),
        Paths.get(str(DATA_PATH.joinpath('normalization'))),
        Paths.get(str(DATA_PATH.joinpath('lm', 'lm.2gram.slm'))),
    )

    return normalizer.normalize(JString(text))


def fps(text: str, n) -> str:
    return ' '.join([w[: n] for w in text.split()])


def preprocess(x, stemming=None):
    x = x.strip()
    x = normalize(x)
    x = remove_punctuation(x)
    x = tokenize(x)
    x = remove_stopwords(x)
    if stemming == 'zemberek':
        x = tokenize(stem(' '.join(x)))
    elif stemming == 'fps5':
        x = tokenize(fps(' '.join(x), 5))
    elif stemming == 'fps7':
        x = tokenize(fps(' '.join(x), 7))

    return x


def remove_punctuation(x):
    return ''.join([w for w in x if w not in string.punctuation])


def tokenize(x):
    return re.split(r'\W+', x)


def remove_stopwords(x):
    return [w for w in x if not trstop.is_stop_word(w)]


file_name = 'TTC-3600/TTC-3600_Orj/ekonomi/c (1).txt'

with open(file_name) as file:
    text = file.read()


print(text)
print(preprocess(text))

x = preprocess(text, stemming='zemberek')
print(x)

x = preprocess(text, stemming='fps5')
print(x)

x = preprocess(text, stemming='fps7')
print(x)
