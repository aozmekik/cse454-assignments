from typing import List
from jpype import JClass, JString, getDefaultJVMPath, shutdownJVM, startJVM, java
from examples import DATA_PATH
from pathlib import Path

ZEMBEREK_PATH = r'zemberek-full.jar'

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
