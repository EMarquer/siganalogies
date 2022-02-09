from .config import *
from .sig_2019 import dataset_factory as dataset_factory_2019
from .sig_2016 import dataset_factory as dataset_factory_2016
import logging

PATH_2016 = "/home/emarquer/Repositories/nn-morpho-analogy/sigmorphon2016/data"
PATH_2019 = "/home/emarquer/Repositories/nn-morpho-analogy/sigmorphon2019/task1"

def build_2019():
    lang_modes =[{"language": lang, "mode": "train-high"} for lang in SIG2019_HIGH]
    lang_modes +=[{"language": lang, "mode": mode} for lang in SIG2019_LOW for mode in SIG2019_LOW_MODES]
    for encoding in ["none", "char"]:
        for lang_mode in lang_modes:
            kwargs = {"word_encoding": encoding, **lang_mode}
            logging.info(f"processing with {kwargs=}")
            dataset = dataset_factory_2019(**kwargs, dataset_folder=PATH_2019, dataset_pkl_folder=join(THIS_DIR,"2019_precomputed"))
            logging.info(f"Number of analogies: {len(dataset.analogies)} (example: {dataset[min(2500, len(dataset)//2)]})")
            logging.info(f"Features: {len(dataset.features)}, with analogies {len(dataset.features_with_analogies)} ({len(dataset.features_with_analogies)/len(dataset.features):.2%})")

def build_2016():
    lang_modes =[{"language": lang, "mode": mode} if lang!="japanese" else {"language": lang, "mode": "train"} for lang in SIG2016_LANGUAGES for mode in SIG2016_MODES]
    for encoding in ["none", "char"]:
        for lang_mode in lang_modes:
            kwargs = {"word_encoding": encoding, **lang_mode}
            logging.info(f"processing with {kwargs=}")
            dataset = dataset_factory_2016(**kwargs, dataset_folder=PATH_2016, dataset_pkl_folder=join(THIS_DIR,"2016_precomputed"))
            logging.info(f"Number of analogies: {len(dataset.analogies)} (example: {dataset[min(2500, len(dataset)//2)]})")
            logging.info(f"Features: {len(dataset.features)}, with analogies {len(dataset.features_with_analogies)} ({len(dataset.features_with_analogies)/len(dataset.features):.2%})")

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
#build_2016()
build_2019()