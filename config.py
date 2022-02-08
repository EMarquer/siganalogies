from os.path import exists, dirname, join
from os import mkdir

THIS_DIR = dirname(__file__)
ROOT = join(THIS_DIR, "..")

SIG2016_PATH = join(ROOT, "sigmorphon2016/data/")
SIG2016_DATASET_PATH = join(THIS_DIR, "2016_precomputed")
if not exists(SIG2016_DATASET_PATH):
    mkdir(SIG2016_DATASET_PATH)
SIG2016_LANGUAGES = ["arabic", "finnish", "georgian", "german", "hungarian", "japanese", "maltese", "navajo", "russian", "spanish", "turkish"]
SIG2016_LANGUAGES_SHORT = ["ar", "fi", "ka", "de", "hu", "mt", "nv", "ru", "es", "tr"] # for BPEmb subword embeddings
SIG2016_MODES = ["train", "dev", "test"] # "test-covered" is not used because it is not exactly the same format as the others

SIG2019_DATA_PATH = join(ROOT, "sigmorphon2019/task1")
SIG2019_DATASET_PATH = join(THIS_DIR, "2019_precomputed")
if not exists(SIG2019_DATASET_PATH):
    mkdir(SIG2019_DATASET_PATH)
SIG2019_FOLDERS = [
    "adyghe--kabardian", "albanian--breton", "arabic--classical-syriac", "arabic--maltese", "arabic--turkmen", "armenian--kabardian",
    "asturian--occitan", "bashkir--azeri", "bashkir--crimean-tatar", "bashkir--kazakh", "bashkir--khakas", "bashkir--tatar", "bashkir--turkmen",
    "basque--kashubian", "belarusian--old-irish", "bengali--greek", "bulgarian--old-church-slavonic", "czech--kashubian", "czech--latin",
    "danish--middle-high-german", "danish--middle-low-german", "danish--north-frisian", "danish--west-frisian", "danish--yiddish",
    "dutch--middle-high-german", "dutch--middle-low-german", "dutch--north-frisian", "dutch--west-frisian", "dutch--yiddish", "english--murrinhpatha",
    "english--north-frisian", "english--west-frisian", "estonian--ingrian", "estonian--karelian", "estonian--livonian", "estonian--votic",
    "finnish--ingrian", "finnish--karelian", "finnish--livonian", "finnish--votic", "french--occitan", "german--middle-high-german",
    "german--middle-low-german", "german--yiddish", "greek--bengali", "hebrew--classical-syriac", "hebrew--maltese", "hindi--bengali",
    "hungarian--ingrian", "hungarian--karelian", "hungarian--livonian", "hungarian--votic", "irish--breton", "irish--cornish",
    "irish--old-irish", "irish--scottish-gaelic", "italian--friulian", "italian--ladin", "italian--maltese", "italian--neapolitan",
    "kannada--telugu", "kurmanji--sorani", "latin--czech", "latvian--lithuanian", "latvian--scottish-gaelic", "persian--azeri",
    "persian--pashto", "polish--kashubian", "polish--old-church-slavonic", "portuguese--russian", "romanian--latin", "russian--old-church-slavonic",
    "russian--portuguese", "sanskrit--bengali", "sanskrit--pashto", "slovak--kashubian", "slovene--old-saxon", "sorani--irish", "spanish--friulian",
    "spanish--occitan", "swahili--quechua", "turkish--azeri", "turkish--crimean-tatar", "turkish--kazakh", "turkish--khakas", "turkish--tatar",
    "turkish--turkmen", "urdu--bengali", "urdu--old-english", "uzbek--azeri", "uzbek--crimean-tatar", "uzbek--kazakh", "uzbek--khakas", "uzbek--tatar",
    "uzbek--turkmen", "welsh--breton", "welsh--cornish", "welsh--old-irish", "welsh--scottish-gaelic", "zulu--swahili"]
SIG2019_HIGH = [
    "adyghe", "albanian", "arabic", "armenian", "asturian", "bashkir", "basque", "belarusian", "bengali", "bulgarian", "czech", "danish", "dutch",
    "english", "estonian", "finnish", "french", "german", "greek", "hebrew", "hindi", "hungarian", "irish", "italian", "kannada", "kurmanji",
    "latin", "latvian", "persian", "polish", "portuguese", "romanian", "russian", "sanskrit", "slovak", "slovene", "sorani", "spanish",
    "swahili", "turkish", "urdu", "uzbek", "welsh", "zulu"]
SIG2019_LOW = [
    "azeri", "bengali", "breton", "classical-syriac", "cornish", "crimean-tatar", "czech", "friulian", "greek", "ingrian", "irish", "kabardian", "karelian",
    "kashubian", "kazakh", "khakas", "ladin", "latin", "lithuanian", "livonian", "maltese", "middle-high-german", "middle-low-german", "murrinhpatha", "neapolitan", "north-frisian", "occitan", "old-church-slavonic", "old-english", "old-irish", "old-saxon", "pashto", "portuguese", "quechua", "russian", "scottish-gaelic", "sorani", "swahili", "tatar", "telugu", "turkmen", "votic", "west-frisian", "yiddish"]
SIG2019_LANGUAGES = sorted(list(set(SIG2019_HIGH + SIG2019_LOW)))
SIG2019_HIGH_LOW_PAIRS = [
    ("adyghe", "kabardian"), ("albanian", "breton"), ("arabic", "classical-syriac"), ("arabic", "maltese"), ("arabic", "turkmen"), ("armenian", "kabardian"),
    ("asturian", "occitan"), ("bashkir", "azeri"), ("bashkir", "crimean-tatar"), ("bashkir", "kazakh"), ("bashkir", "khakas"), ("bashkir", "tatar"), ("bashkir", "turkmen"), ("basque", "kashubian"), ("belarusian", "old-irish"), ("bengali", "greek"), ("bulgarian", "old-church-slavonic"), ("czech", "kashubian"), ("czech", "latin"), ("danish", "middle-high-german"), ("danish", "middle-low-german"), ("danish", "north-frisian"), ("danish", "west-frisian"), ("danish", "yiddish"), ("dutch", "middle-high-german"), ("dutch", "middle-low-german"), ("dutch", "north-frisian"), ("dutch", "west-frisian"), ("dutch", "yiddish"), ("english", "murrinhpatha"), ("english", "north-frisian"), ("english", "west-frisian"), ("estonian", "ingrian"), ("estonian", "karelian"), ("estonian", "livonian"), ("estonian", "votic"), ("finnish", "ingrian"), ("finnish", "karelian"), ("finnish", "livonian"), ("finnish", "votic"), ("french", "occitan"), ("german", "middle-high-german"), ("german", "middle-low-german"), ("german", "yiddish"), ("greek", "bengali"), ("hebrew", "classical-syriac"), ("hebrew", "maltese"), ("hindi", "bengali"), ("hungarian", "ingrian"), ("hungarian", "karelian"), ("hungarian", "livonian"), ("hungarian", "votic"), ("irish", "breton"), ("irish", "cornish"), ("irish", "old-irish"), ("irish", "scottish-gaelic"), ("italian", "friulian"), ("italian", "ladin"), ("italian", "maltese"), ("italian", "neapolitan"), ("kannada", "telugu"), ("kurmanji", "sorani"), ("latin", "czech"), ("latvian", "lithuanian"), ("latvian", "scottish-gaelic"), ("persian", "azeri"), ("persian", "pashto"), ("polish", "kashubian"), ("polish", "old-church-slavonic"), ("portuguese", "russian"), ("romanian", "latin"), ("russian", "old-church-slavonic"), ("russian", "portuguese"), ("sanskrit", "bengali"), ("sanskrit", "pashto"), ("slovak", "kashubian"), ("slovene", "old-saxon"), ("sorani", "irish"), ("spanish", "friulian"), ("spanish", "occitan"), ("swahili", "quechua"), ("turkish", "azeri"), ("turkish", "crimean-tatar"), ("turkish", "kazakh"), ("turkish", "khakas"), ("turkish", "tatar"), ("turkish", "turkmen"), ("urdu", "bengali"), ("urdu", "old-english"), ("uzbek", "azeri"), ("uzbek", "crimean-tatar"), ("uzbek", "kazakh"), ("uzbek", "khakas"), ("uzbek", "tatar"), ("uzbek", "turkmen"), ("welsh", "breton"), ("welsh", "cornish"), ("welsh", "old-irish"), ("welsh", "scottish-gaelic"), ("zulu", "swahili")]
SIG2019_HIGH_MODES = ["train-high"]
SIG2019_LOW_MODES = ["train-low", "test", "dev"] # "test-covered" is not used because it is not exactly the same format as the others

#print(set(SIG2019_HIGH).intersection(set(SIG2019_LOW)))