from os.path import exists, join
from torch.utils.data import Dataset
from .config import DOWNLOAD, SERIALIZATION, SIG2019_HIGH_LOW_PAIRS, SIG2019_LANGUAGES, SIG2019_DATASET_PATH, SIG2019_SERIALIZATION_PATH, SIG2019_HIGH, SIG2019_LOW, SIG2019_HIGH_MODES, SIG2019_LOW_MODES, dorel_pkl_url
from datetime import datetime
import logging
from .abstract_analogy_dataset import AbstractAnalogyDataset, StateDict, save_state_dict, load_state_dict
import typing as t
from .encoders import Encoder, encoder_as_string

# create logger
module_logger = logging.getLogger(__name__)

def _load_data(language="german", mode="train-high", dataset_folder=SIG2019_DATASET_PATH):
    """Load the data from the sigmorphon files in the form of a list of triples (lemma, target_features, target_word)."""
    filename = get_file_name(language, mode, dataset_folder=dataset_folder)
    def _split_n_reorder(line): # to get elements in the same order as in Sigmorphon 2016
        a, b, feature_b = line.strip().split('\t')
        return a, feature_b, b
    with open(filename, "r", encoding="utf-8") as f:
        return [_split_n_reorder(line) for line in f]

def get_file_name(language="german", mode="train-high", dataset_folder=SIG2019_DATASET_PATH):
    """Checks that a language/mode combination is valid and return the complete filepath if it is.
    
    If it is not valid, raise an AssertError or a ValueError.
    """
    if language in SIG2019_HIGH and language in SIG2019_LOW:
        assert mode in SIG2019_HIGH_MODES or mode in SIG2019_LOW_MODES, f"Language '{language}' is both a high and a low ressource language, and the only allowed modes are {SIG2019_HIGH_MODES + SIG2019_LOW_MODES}."
        if mode in SIG2019_HIGH_MODES:
            # get the language-pair subfolder
            folder = next(f"{lang_high}--{lang_low}" for lang_high, lang_low in SIG2019_HIGH_LOW_PAIRS if lang_high == language)
        else:
            # get the language-pair subfolder
            folder = next(f"{lang_high}--{lang_low}" for lang_high, lang_low in SIG2019_HIGH_LOW_PAIRS if lang_low == language)

    elif language in SIG2019_HIGH:
        assert mode in SIG2019_HIGH_MODES, f"Language '{language}' is a high ressource language, and the only allowed modes are {SIG2019_HIGH_MODES}."
        
        # get the language-pair subfolder
        folder = next(f"{lang_high}--{lang_low}" for lang_high, lang_low in SIG2019_HIGH_LOW_PAIRS if lang_high == language)
    elif language in SIG2019_LOW:
        assert mode in SIG2019_LOW_MODES, f"Language '{language}' is a low ressource language, and the only allowed modes are {SIG2019_LOW_MODES}."

        # get the language-pair subfolder
        folder = next(f"{lang_high}--{lang_low}" for lang_high, lang_low in SIG2019_HIGH_LOW_PAIRS if lang_low == language)
    else:
        raise ValueError(f"Language '{language}' is unkown, allowed languages are {SIG2019_LANGUAGES}")

    # concatenate the folder, the language-pair subfolder, and the filename
    return join(dataset_folder, folder, f"{language}-{mode}")

class Sig2019Dataset(AbstractAnalogyDataset):
    @staticmethod
    def from_state_dict(state_dict: StateDict, dataset_folder=SIG2019_DATASET_PATH, load_data=True) -> AbstractAnalogyDataset:
        """Create a dataset from saved data."""
        dataset = Sig2019Dataset(
            language=state_dict["language"],
            mode=state_dict["mode"],
            word_encoder=state_dict["word_encoder"],
            building=False, state_dict=state_dict, dataset_folder=dataset_folder)
        return dataset

    def state_dict(self) -> StateDict:
        "Return a data dictionary, loadable for future use of the dataset."
        state_dict = {
            "timestamp": datetime.now(),
            "language": self.language,
            "mode": self.mode,
            "word_encoder": self.word_encoder,
            "analogies": self.analogies,
            "word_voc": self.word_voc,
            "features": self.features,
            "features_with_analogies": self.features_with_analogies,
            "words_with_analogies": self.words_with_analogies
        }
        return state_dict

    def __init__(self, language="german", mode="train-high", word_encoder: t.Union[t.Type, Encoder, str, None]=None,
            building=True, state_dict: StateDict=None, dataset_folder=SIG2019_DATASET_PATH, load_data=True, **kwargs):
        """A dataset class for manipultating files of task 1 of Sigmorphon2019."""
        super().__init__(word_encoder=word_encoder)
        assert language in SIG2019_LANGUAGES
        self.language = language
        self.mode = mode
        if load_data or building:
            self.raw_data = _load_data(language=language, mode=mode, dataset_folder=dataset_folder)

        if building:
            self.set_analogy_classes()
            self.prepare_encoder()
        elif state_dict is not None:
            self.analogies = state_dict["analogies"]
            self.word_voc = state_dict["word_voc"]
            self.features = state_dict["features"]
            self.features_with_analogies = state_dict["features_with_analogies"]
            self.words_with_analogies = state_dict["words_with_analogies"]

    def set_analogy_classes(self):
        """Go through the data to extract the vocabulary, the available features, and build analogies."""
        self.analogies = []
        self.word_voc = set()
        self.features = set()
        self.features_with_analogies = set()
        self.words_with_analogies = set()
        for i, (word_a_i, feature_b_i, word_b_i) in enumerate(self.raw_data):
            self.word_voc.add(word_a_i)
            self.word_voc.add(word_b_i)
            self.features.add(feature_b_i)
            self.analogies.append((i, i)) # add the identity
            for j, (word_a_j, feature_b_j, word_b_j) in enumerate(self.raw_data[i+1:]):
                if feature_b_i == feature_b_j:
                    self.analogies.append((i, i+1+j))
                    self.features_with_analogies.add(feature_b_i)
                    self.words_with_analogies.add(word_a_i)
                    self.words_with_analogies.add(word_b_i)
                    self.words_with_analogies.add(word_a_j)
                    self.words_with_analogies.add(word_b_j)

    def __getitem__(self, index):
        ab_index, cd_index = self.analogies[index]
        a, feature_b, b = self.raw_data[ab_index]
        c, feature_d, d = self.raw_data[cd_index]
        return self.encode(a, b, c, d)

class BilingualDataset(Dataset):
    @staticmethod
    def from_state_dict(state_dict: StateDict, dataset_folder=SIG2019_DATASET_PATH):
        """Create a dataset from saved data."""
        dataset = BilingualDataset(building=False, dataset_folder=dataset_folder, **state_dict)
        return dataset

    def state_dict(self) -> StateDict:
        "Return a data dictionary, loadable for future use of the dataset."
        state_dict = {
            "timestamp": datetime.now(),
            "analogies": self.analogies,
            "features_with_analogies": self.features_with_analogies,
            "words_with_analogies_high": self.words_with_analogies_high,
            "words_with_analogies_low": self.words_with_analogies_low,
            "state_dict_high": self.dataset_high.state_dict(),
            "state_dict_low": self.dataset_low.state_dict()
        }
        return state_dict

    def __init__(self, language_high="german", language_low="middle-high-german", mode_low="train-high", word_encoding="none", building=True, state_dict: StateDict=None, dataset_folder=SIG2019_DATASET_PATH, **kwargs):
        """
        :param mode_low: Dataset subset for the low-ressource language (dev, test, test-covered, train-low). 
            There is no option for the high ressource language as only train-high is available.
        """
        super(Sig2019Dataset).__init__()
        assert (language_high, language_low) in SIG2019_HIGH_LOW_PAIRS, f"({language_high}, {language_low}) is not a valid language pair for sigmorphon 2019 bilingual analogies."

        if building:
            self.dataset_high = dataset_factory(language=language_high, mode="train-high", word_encoder=word_encoding, force_rebuild=kwargs.get("force_rebuild", False), dataset_folder=dataset_folder)
            self.dataset_low = dataset_factory(language=language_low, mode=mode_low, word_encoder=word_encoding, force_rebuild=kwargs.get("force_rebuild", False), dataset_folder=dataset_folder)
            self.set_analogy_classes()
        elif state_dict is not None:
            self.dataset_high = Sig2019Dataset(building=False, **kwargs["state_dict_high"], dataset_folder=dataset_folder)
            self.dataset_low = Sig2019Dataset(building=False, **kwargs["state_dict_low"], dataset_folder=dataset_folder)
            self.analogies = kwargs["analogies"]
            self.features_with_analogies = kwargs["features_with_analogies"]
            self.words_with_analogies_high = kwargs["words_with_analogies_high"]
            self.words_with_analogies_low = kwargs["words_with_analogies_low"]

    def set_analogy_classes(self):
        """Go through the data to extract the vocabulary, the available features, and build analogies."""
        self.analogies = []
        self.features_with_analogies = set()
        self.words_with_analogies_high = set()
        self.words_with_analogies_low = set()
        for i, (word_a_i, feature_b_i, word_b_i) in enumerate(self.dataset_high.raw_data):
            for j, (word_a_j, feature_b_j, word_b_j) in enumerate(self.dataset_low.raw_data):
                if feature_b_i == feature_b_j:
                    self.analogies.append((i, j))
                    self.features_with_analogies.add(feature_b_i)
                    self.words_with_analogies_high.add(word_a_i)
                    self.words_with_analogies_high.add(word_b_i)
                    self.words_with_analogies_low.add(word_a_j)
                    self.words_with_analogies_low.add(word_b_j)

    def __getitem__(self, index):
        ab_index, cd_index = self.analogies[index]
        a, feature_b, b = self.dataset_high.raw_data[ab_index]
        c, feature_d, d = self.dataset_low.raw_data[cd_index]
        return self.dataset_high.encode_word(a), self.dataset_high.encode_word(b), self.dataset_low.encode_word(c), self.dataset_low.encode_word(d)

def dataset_factory(language="german", mode="train-high", word_encoder: t.Union[t.Type, Encoder, str, None]=None,
        dataset_pkl_folder=SIG2019_SERIALIZATION_PATH, dataset_folder=SIG2019_DATASET_PATH, force_rebuild=False,
        serialization=SERIALIZATION, download=DOWNLOAD, load_data=True) -> Sig2019Dataset:
    assert mode in SIG2019_HIGH_MODES or mode in SIG2019_LOW_MODES
    file_name = f"{language}-{mode}-{encoder_as_string(word_encoder)}.pkl"
    filepath = join(dataset_pkl_folder, file_name)


    # pickle does not exist, if allowed, attempt to fetch it online
    if not force_rebuild and not exists(filepath) and download:
        import urllib.request as r
        import urllib.error as er
        import pickle
        try:
            module_logger.info(f"Downloading remote pickle to {filepath}...")
            url = dorel_pkl_url(dataset=2019, language=language, mode=mode, word_encoder=word_encoder)
            
            if serialization: # download to file
                r.urlretrieve(url, filepath)
                module_logger.info(f"Dataset {file_name} downloaded from remote file to {filepath}.")
            else: # use on the fly, return the file directly
                state_dict = pickle.load(r.urlopen(url))
                dataset = Sig2019Dataset.from_state_dict(state_dict=state_dict, dataset_folder=dataset_folder,
                        load_data=load_data)
                module_logger.info(f"Dataset {file_name} loaded from remote file.")
                return dataset
        except er.HTTPError:
            module_logger.error(f"HTTPError while attempting to get {file_name} from {url}. Possibly, the file does not exist remotely.")
        except er.URLError:
            module_logger.error(f"URLError while attempting to get {file_name} from {url}.")

    # pickle still does not exist (error above or download is False), create it
    if force_rebuild or not exists(filepath):
        module_logger.info(f"Starting building the dataset {filepath}...")
        if mode not in {"train-high", "train-low"}:
            module_logger.info(f"Using the corresponding training dataset for  {filepath}...")
            train_dataset = dataset_factory(language=language, mode="train-low", word_encoder=word_encoder,
                    dataset_pkl_folder=dataset_pkl_folder, force_rebuild=force_rebuild, dataset_folder=dataset_folder,
                    load_data=load_data)
            state_dict = train_dataset.state_dict()
            state_dict["mode"] = mode
            dataset = Sig2019Dataset.from_state_dict(state_dict, dataset_folder=dataset_folder)
            module_logger.info(f"Computing the analogies for {filepath}...")
            dataset.set_analogy_classes()
        else:
            dataset = Sig2019Dataset(language=language, mode=mode, word_encoder=word_encoder, 
                    dataset_folder=dataset_folder, load_data=load_data)
        module_logger.info(f"Dataset {filepath} built.")
        
        if serialization:
            module_logger.info(f"Saving the dataset to {filepath}...")
            state_dict = dataset.state_dict()
            save_state_dict(state_dict, filepath)
            module_logger.info(f"Dataset saved to {filepath}.")
    
    # pickle exists
    else:
        state_dict = load_state_dict(filepath)
        dataset = Sig2019Dataset.from_state_dict(state_dict=state_dict, dataset_folder=dataset_folder,
                load_data=load_data)
        module_logger.info(f"Dataset {filepath} loaded.")
    return dataset

def bilingual_dataset_factory(language_high="german", language_low="middle-high-german", mode_low="train-low", word_encoding="none", dataset_pkl_folder=SIG2019_SERIALIZATION_PATH, dataset_folder=SIG2019_DATASET_PATH, force_rebuild=False, serialization=SERIALIZATION) -> BilingualDataset:
    filepath = join(dataset_pkl_folder, f"{language_high}--{language_low}-{mode_low}-{word_encoding}.pkl")
    if force_rebuild or not exists(filepath):
        dataset = BilingualDataset(language_high=language_high, language_low=language_low, mode_low=mode_low, word_encoding=word_encoding, dataset_folder=dataset_folder)
        
        if serialization:
            state_dict = dataset.state_dict()
            save_state_dict(state_dict, filepath)
    else:
        state_dict = load_state_dict(filepath)
        dataset = BilingualDataset.from_state_dict(state_dict=state_dict, dataset_folder=dataset_folder)
    return dataset

if __name__ == "__main__":
    from time import process_time
    from .config import THIS_DIR

    def check_load_building_time():
        features = []
        #features += [{"word_encoding": "char"}, {"word_encoding": "none"}]
        features += [{"word_encoding": "none", "language": lang, "mode": "train-high"} for lang in SIG2019_HIGH]
        features += [{"word_encoding": "none", "language": lang, "mode": "train-low"} for lang in SIG2019_LOW]
        features += [{"word_encoding": "none", "language": lang, "mode": "dev"} for lang in SIG2019_LOW]
        features += [{"word_encoding": "none", "language": lang, "mode": "test"} for lang in SIG2019_LOW]
        for kwargs in features:
            print(f"For kwargs `{kwargs}`")
            t = process_time()
            dataset = dataset_factory(force_rebuild=True, **kwargs)
            t = process_time() - t
            print(f"building time: {t}s")
            t = process_time()
            dataset = dataset_factory(**kwargs)
            t = process_time() - t
            print(f"post-building time: {t}s")
            print(f"Number of analogies: {len(dataset.analogies)} (example: {dataset[min(2500, len(dataset)//2)]})")
            print(f"Features: {len(dataset.features)}, with analogies {len(dataset.features_with_analogies)} ({len(dataset.features_with_analogies)/len(dataset.features):.2%})")

    def compare_languages():
        # summary
        features = [{"word_encoding": "none", "language": lang, "mode": "train-high"} for lang in SIG2019_HIGH]
        features += [{"word_encoding": "none", "language": lang, "mode": "train-low"} for lang in SIG2019_LOW]
        #features += [{"word_encoding": "none", "language": lang, "mode": "dev"} for lang in SIG2019_LOW]
        #features += [{"word_encoding": "none", "language": lang, "mode": "test"} for lang in SIG2019_LOW]
        import pandas
        records = []
        records_sns = []
        for kwargs in features:
            dataset = dataset_factory(**kwargs)
            record = {**kwargs,
                "words": len(dataset.word_voc),
                "words with analogies (identity excluded)": len(dataset.words_with_analogies),
                "analogies (identity included)": len(dataset.analogies),
                "features": len(dataset.features),
                "features with analogies (identity excluded)": len(dataset.features_with_analogies),
            }
            
            record_sns = [
                {**kwargs, "element": "words", "measure": "count", "value": len(dataset.word_voc)},
                #{**kwargs, "element": "words", "measure": "count with analogies", "value": len(dataset.words_with_analogies)},
                #{**kwargs, "element": "words", "measure": "analogy coverage ratio", "value": len(dataset.words_with_analogies)/len(dataset.word_voc)},
                {**kwargs, "element": "features", "measure": "count", "value": len(dataset.features)},
                #{**kwargs, "element": "features", "measure": "count with analogies", "value": len(dataset.features_with_analogies)},
                #{**kwargs, "element": "features", "measure": "analogy coverage ratio", "value": len(dataset.features_with_analogies)/len(dataset.features)},
                {**kwargs, "element": "analogies", "measure": "count", "value": len(dataset.analogies)},
            ]
            records.append(record)
            records_sns.extend(record_sns)
        df = pandas.DataFrame.from_records(records)
        df["analogy coverage word ratio (identity excluded)"] = df["words with analogies (identity excluded)"] / df["words"]
        df["analogy coverage feature ratio (identity excluded)"] = df["features with analogies (identity excluded)"] / df["features"]
        df.to_csv(join(THIS_DIR, "none-summary.csv"))
        df_sns = pandas.DataFrame.from_records(records_sns)
    
        import seaborn as sns
        import matplotlib.pyplot as plt
        fig = sns.catplot(x="language", y="value", data=df_sns[
            df_sns["language"].apply(lambda l: l in SIG2019_HIGH) & df_sns["mode"].apply(lambda l: l in SIG2019_HIGH_MODES)
            ], row="element", kind="bar", sharex="row", sharey="none", aspect=10)
        fig.savefig(join(THIS_DIR, "none-summary-high.png"))
        fig = sns.catplot(x="language", y="value", data=df_sns[
            df_sns["language"].apply(lambda l: l in SIG2019_LOW) & df_sns["mode"].apply(lambda l: l in SIG2019_LOW_MODES)
            ], row="element", col="mode", kind="bar", sharex="row", sharey="none", aspect=10)
        fig.savefig(join(THIS_DIR, "none-summary-low.png"))
    
    def check_load_building_time_bilingual():
        features = []
        #features += [{"word_encoding": "char"}, {"word_encoding": "none"}]
        features += [{"word_encoding": "none", "language_high": lang[0], "language_low": lang[1], "mode_low": "train-low"} for lang in SIG2019_HIGH_LOW_PAIRS]
        features += [{"word_encoding": "none", "language_high": lang[0], "language_low": lang[1], "mode_low": "dev"} for lang in SIG2019_HIGH_LOW_PAIRS]
        features += [{"word_encoding": "none", "language_high": lang[0], "language_low": lang[1], "mode_low": "test"} for lang in SIG2019_HIGH_LOW_PAIRS]
        for kwargs in features:
            print(f"For kwargs `{kwargs}`")
            t = process_time()
            dataset = bilingual_dataset_factory(force_rebuild=True, **kwargs)
            t = process_time() - t
            print(f"building time: {t}s")
            t = process_time()
            dataset = bilingual_dataset_factory(**kwargs)
            t = process_time() - t
            print(f"post-building time: {t}s")
            if len(dataset.analogies) < 1:
                print(f"=== Language pair ({kwargs['language_high']}, {kwargs['language_low']}) has no valid analogical pairs ===")
            else:
                print(f"Number of analogies: {len(dataset.analogies)} (example: {dataset[min(2500, len(dataset)//2)]})")
                print(f"Features: {len(dataset.dataset_high.features | dataset.dataset_high.features)}, with analogies {len(dataset.features_with_analogies)} ({len(dataset.features_with_analogies)/len(dataset.dataset_high.features | dataset.dataset_high.features):.2%})")
    
    def compare_languages_bilingual():
        # summary
        features = [{"word_encoding": "none", "language_high": lang[0], "language_low": lang[1], "mode_low": "train-low"} for lang in SIG2019_HIGH_LOW_PAIRS]
        features += [{"word_encoding": "none", "language_high": lang[0], "language_low": lang[1], "mode_low": "dev"} for lang in SIG2019_HIGH_LOW_PAIRS]
        features += [{"word_encoding": "none", "language_high": lang[0], "language_low": lang[1], "mode_low": "test"} for lang in SIG2019_HIGH_LOW_PAIRS]
        import pandas
        records = []
        records_sns = []
        for kwargs in features:
            dataset = bilingual_dataset_factory(**kwargs)
            kwargs["language_pair"] = f'{kwargs["language_high"]} -> {kwargs["language_low"]}'
            record = {**kwargs,
                "words with analogies (high language, identity excluded)": len(dataset.words_with_analogies_high),
                "words with analogies (low language, identity excluded)": len(dataset.words_with_analogies_low),
                "analogies (identity included)": len(dataset.analogies),
                "features": len(dataset.dataset_high.features | dataset.dataset_low.features),
                "features with analogies (identity excluded)": len(dataset.features_with_analogies),
            }
            
            record_sns = [
                #{**kwargs, "element": "words", "measure": "count", "value": len(dataset.dataset_high.word_voc | dataset.dataset_low.word_voc)},
                #{**kwargs, "element": "words", "measure": "count with analogies", "value": len(dataset.words_with_analogies)},
                #{**kwargs, "element": "words", "measure": "analogy coverage ratio", "value": len(dataset.words_with_analogies)/len(dataset.dataset_high.word_voc | dataset.dataset_low.word_voc)},
                {**kwargs, "element": "features", "measure": "count", "value": len(dataset.dataset_high.features | dataset.dataset_low.features)},
                {**kwargs, "element": "features with analogies", "measure": "count", "value": len(dataset.features_with_analogies)},
                #{**kwargs, "element": "features", "measure": "analogy coverage ratio", "value": len(dataset.features_with_analogies)/len(dataset.dataset_high.features | dataset.dataset_low.features)},
                {**kwargs, "element": "analogies", "measure": "count", "value": len(dataset.analogies)},
            ]
            records.append(record)
            records_sns.extend(record_sns)
        df = pandas.DataFrame.from_records(records)
        #df["analogy coverage word ratio (identity excluded)"] = df["words with analogies (identity excluded)"] / df["words"]
        df["analogy coverage feature ratio (identity excluded)"] = df["features with analogies (identity excluded)"] / df["features"]
        df.to_csv(join(THIS_DIR, "none-summary.csv"))
        df_sns = pandas.DataFrame.from_records(records_sns)
    
        import seaborn as sns
        import matplotlib.pyplot as plt
        fig = sns.catplot(x="language_pair", y="value", data=df_sns,
            row="element", kind="bar", sharex="row", sharey="row",
            hue="mode_low", aspect=20)
        #plt.show()
        fig.savefig(join(THIS_DIR, "none-summary-bilingual.png"))
        #fig = sns.catplot(x="language", y="value", data=df_sns[
        #    df_sns["language"].apply(lambda l: l in SIG2019_LOW) & df_sns["mode"].apply(lambda l: l in SIG2019_LOW_MODES)
        #    ], row="element", col="mode", kind="bar", sharex="row", sharey="none", aspect=10)
        #fig.savefig(join(SIG2019_SERIALIZATION_PATH, "none-summary-low.png"))

    #compare_languages()
    dataset = dataset_factory(word_encoder="char")
    print(len(dataset.analogies))
    print(dataset[1])
    print(dataset.analogies[1])
    print(dataset.raw_data[dataset.analogies[1][0]])
    print(dataset.raw_data[dataset.analogies[1][1]])

    print()

    dataset = dataset_factory(word_encoder=id)
    print(len(dataset.analogies))
    print(dataset[1])
    print(dataset.analogies[1])
    print(dataset.raw_data[dataset.analogies[1][0]])
    print(dataset.raw_data[dataset.analogies[1][1]])