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

morphology = TurkishMorphology.createWithDefaults()


def stem(text: str) -> str:
    results: WordAnalysis = morphology.analyze(JString(text))
    for result in results:
        return str(result.getLemmas()[0])


normalizer = TurkishSentenceNormalizer(
    TurkishMorphology.createWithDefaults(),
    Paths.get(str(DATA_PATH.joinpath('normalization'))),
    Paths.get(str(DATA_PATH.joinpath('lm', 'lm.2gram.slm'))),
)


def normalize(text: str) -> str:
    return normalizer.normalize(JString(text))


def fps(text: str, n) -> str:
    return ' '.join([w[: n] for w in text.split()])


def preprocess(x, stemming=None, stopword=False):
    x = strip_numbers(x)
    x = normalize(x)
    x = remove_punctuation(x)
    x = tokenize(x)
    if stopword:
        x = remove_stopwords(x)
    if stemming == 'zemb':
        x = [stem(w) for w in x]
        x = [w for w in x if w]
        # x = tokenize(stem(' '.join(x)))
    elif stemming == 'fps5':
        x = tokenize(fps(' '.join(x), 5))
    elif stemming == 'fps7':
        x = tokenize(fps(' '.join(x), 7))
    
    return ' '.join(x).strip()


def remove_punctuation(x):
    return ''.join([w for w in x if w not in string.punctuation])


def tokenize(x):
    return re.split(r'\W+', x)


def remove_stopwords(x):
    return [w for w in x if not trstop.is_stop_word(w)]

def strip_numbers(x):
    return re.sub(' +', ' ', re.sub(r'\d+', '', x)).strip()


# file_name = 'TTC-3600/TTC-3600_Orj/ekonomi/c (1).txt'

# with open(file_name) as file:
#     text = file.read()

