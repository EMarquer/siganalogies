from os.path import exists, join
from torch.utils.data import Dataset
from torch import save, load, LongTensor, zeros
from .config import SIG2019_HIGH_LOW_PAIRS, SIG2019_LANGUAGES, SIG2019_DATA_PATH, SIG2019_DATASET_PATH, SIG2019_HIGH, SIG2019_LOW, SIG2019_HIGH_MODES, SIG2019_LOW_MODES
from datetime import datetime
import logging

def load_data(language="german", mode="train-high", dataset_folder=SIG2019_DATA_PATH):
    """Load the data from the sigmorphon files in the form of a list of triples (lemma, target_features, target_word)."""
    filename = get_file_name(language, mode, dataset_folder=dataset_folder)
    with open(filename, "r", encoding="utf-8") as f:
        return [line.strip().split('\t') for line in f]

def get_file_name(language="german", mode="train-high", dataset_folder=SIG2019_DATA_PATH):
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

class Task1Dataset(Dataset):
    @staticmethod
    def from_state_dict(state_dict, dataset_folder=SIG2019_DATA_PATH):
        """Create a dataset from saved data."""
        dataset = Task1Dataset(loading=True, dataset_folder=dataset_folder, **state_dict)
        return dataset

    def state_dict(self):
        "Return a data dictionary, loadable for future use of the dataset."
        state_dict = {
            "timestamp": datetime.now(),
            "language": self.language,
            "mode": self.mode,
            "word_encoding": self.word_encoding,
            "analogies": self.analogies,
            "word_voc": self.word_voc,
            "features": self.features,
            "features_with_analogies": self.features_with_analogies,
            "words_with_analogies": self.words_with_analogies
        }
        if self.word_encoding == "char":
            state_dict["char_voc"] = self.char_voc
            state_dict["char_voc_id"] = self.char_voc_id
        return state_dict

    def __init__(self, language="german", mode="train-high", word_encoding="none", loading=False, dataset_folder=SIG2019_DATA_PATH, **kwargs):
        """A dataset class for manipultating files of task 1 of Sigmorphon2019."""
        super(Task1Dataset).__init__()
        assert language in SIG2019_LANGUAGES
        self.language = language
        self.mode = mode
        self.raw_data = load_data(language = language, mode = mode, dataset_folder=dataset_folder)
        self.word_encoding = word_encoding

        if not loading:
            self.set_analogy_classes()
            self.prepare_data()
        else:
            self.analogies = kwargs["analogies"]
            self.word_voc = kwargs["word_voc"]
            self.features = kwargs["features"]
            self.features_with_analogies = kwargs["features_with_analogies"]
            self.words_with_analogies = kwargs["words_with_analogies"]
            if self.word_encoding == "char":
                self.char_voc = kwargs["char_voc"]
                self.char_voc_id = kwargs["char_voc_id"]

        

    def prepare_data(self):
        """Generate embeddings for the 4 elements.

        There are 2 modes to encode the words:
        - 'char': sequence of ids of characters, wrt. a dictioanry of values;
        - 'none' or None: no encoding, particularly useful when coupled with BERT encodings.
        """
        if self.word_encoding == "char":
            # generate character vocabulary
            voc = set()
            for word in self.word_voc:
                voc.update(word)
            self.char_voc = list(voc)
            self.char_voc.sort()
            self.char_voc_id = {character: i for i, character in enumerate(self.char_voc)}

        elif self.word_encoding == "none" or self.word_encoding is None:
            pass

        else:
            print(f"Unsupported word encoding: {self.word_encoding}")

    def set_analogy_classes(self):
        """Go through the data to extract the vocabulary, the available features, and build analogies."""
        self.analogies = []
        self.word_voc = set()
        self.features = set()
        self.features_with_analogies = set()
        self.words_with_analogies = set()
        for i, (word_a_i, word_b_i, feature_b_i) in enumerate(self.raw_data):
            self.word_voc.add(word_a_i)
            self.word_voc.add(word_b_i)
            self.features.add(feature_b_i)
            self.analogies.append((i, i)) # add the identity
            for j, (word_a_j, word_b_j, feature_b_j) in enumerate(self.raw_data[i+1:]):
                if feature_b_i == feature_b_j:
                    self.analogies.append((i, i+j))
                    self.features_with_analogies.add(feature_b_i)
                    self.words_with_analogies.add(word_a_i)
                    self.words_with_analogies.add(word_b_i)
                    self.words_with_analogies.add(word_a_j)
                    self.words_with_analogies.add(word_b_j)

    def encode_word(self, word):
        """Encode a single word using the selected encoding process."""
        if self.word_encoding == "char":
            return LongTensor([self.char_voc_id[c] if c in self.char_voc_id.keys() else -1 for c in word])
        elif self.word_encoding == "glove":
            return self.glove.embeddings.get(word, zeros(300))
        elif self.word_encoding == "none" or self.word_encoding is None:
            return word
        else:
            raise ValueError(f"Unsupported word encoding: {self.word_encoding}")
    
    def encode(self, a, b, c, d):
        """Encode 4 words using the selected encoding process."""
        return self.encode_word(a), self.encode_word(b), self.encode_word(c), self.encode_word(d)

    def decode_word(self, word):
        """Decode a single word using the selected encoding process."""
        if self.word_encoding == "char":
            return "".join([self.char_voc[char.item()] for char in word])
        elif self.word_encoding == "none" or self.word_encoding is None:
            logging.info("Word decoding not necessary when using 'none' encoding.")
            return word
        elif self.word_encoding == "glove":
            raise ValueError("Word decoding not supported with GloVe.")
        else:
            raise ValueError(f"Unsupported word encoding: {self.word_encoding}")

    def __len__(self):
        return len(self.analogies)

    def __getitem__(self, index):
        ab_index, cd_index = self.analogies[index]
        a, b, feature_b = self.raw_data[ab_index]
        c, d, feature_d = self.raw_data[cd_index]
        return self.encode(a, b, c, d)

class BilingualDataset(Dataset):
    @staticmethod
    def from_state_dict(state_dict, dataset_folder=SIG2019_DATA_PATH):
        """Create a dataset from saved data."""
        dataset = BilingualDataset(loading=True, dataset_folder=dataset_folder, **state_dict)
        return dataset

    def state_dict(self):
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

    def __init__(self, language_high="german", language_low="middle-high-german", mode_low="train-high", word_encoding="none", loading=False, dataset_folder=SIG2019_DATA_PATH, **kwargs):
        """
        :param mode_low: Dataset subset for the low-ressource language (dev, test, test-covered, train-low). 
            There is no option for the high ressource language as only train-high is available.
        """
        super(Task1Dataset).__init__()
        assert (language_high, language_low) in SIG2019_HIGH_LOW_PAIRS, f"({language_high}, {language_low}) is not a valid language pair for sigmorphon 2019 bilingual analogies."

        if not loading:
            self.dataset_high = dataset_factory(language=language_high, mode="train-high", word_encoding=word_encoding, loading=False, force_rebuild=kwargs.get("force_rebuild", False), dataset_folder=dataset_folder)
            self.dataset_low = dataset_factory(language=language_low, mode=mode_low, word_encoding=word_encoding, loading=False, force_rebuild=kwargs.get("force_rebuild", False), dataset_folder=dataset_folder)
            self.set_analogy_classes()
        else:
            self.dataset_high = Task1Dataset(loading=True, **kwargs["state_dict_high"], dataset_folder=dataset_folder)
            self.dataset_low = Task1Dataset(loading=True, **kwargs["state_dict_low"], dataset_folder=dataset_folder)
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
        for i, (word_a_i, word_b_i, feature_b_i) in enumerate(self.dataset_high.raw_data):
            for j, (word_a_j, word_b_j, feature_b_j) in enumerate(self.dataset_low.raw_data):
                if feature_b_i == feature_b_j:
                    self.analogies.append((i, j))
                    self.features_with_analogies.add(feature_b_i)
                    self.words_with_analogies_high.add(word_a_i)
                    self.words_with_analogies_high.add(word_b_i)
                    self.words_with_analogies_low.add(word_a_j)
                    self.words_with_analogies_low.add(word_b_j)

    def __len__(self):
        return len(self.analogies)

    def __getitem__(self, index):
        ab_index, cd_index = self.analogies[index]
        a, b, feature_b = self.dataset_high.raw_data[ab_index]
        c, d, feature_d = self.dataset_low.raw_data[cd_index]
        return self.dataset_high.encode_word(a), self.dataset_high.encode_word(b), self.dataset_low.encode_word(c), self.dataset_low.encode_word(d)

def dataset_factory(language="german", mode="train-high", word_encoding="none", dataset_pkl_folder=SIG2019_DATASET_PATH, dataset_folder=SIG2019_DATA_PATH, force_rebuild=False) -> Task1Dataset:
    assert mode in SIG2019_HIGH_MODES or mode in SIG2019_LOW_MODES
    filepath = join(dataset_pkl_folder, f"{language}-{mode}-{word_encoding}.tch")
    if force_rebuild or not exists(filepath):
        if mode not in {"train-high", "train-low"}:
            train_dataset = dataset_factory(language=language, mode="train-low", word_encoding=word_encoding, dataset_pkl_folder=dataset_pkl_folder, force_rebuild=force_rebuild, dataset_folder=dataset_folder)
            state_dict = train_dataset.state_dict()
            state_dict["mode"] = mode
            dataset = Task1Dataset(loading=True, **state_dict, dataset_folder=dataset_folder)
            dataset.set_analogy_classes()
        else:
            dataset = Task1Dataset(language=language, mode=mode, word_encoding=word_encoding, dataset_folder=dataset_folder)
        state_dict = dataset.state_dict()
        save(state_dict, filepath)
    else:
        state_dict = load(filepath)
        dataset = Task1Dataset.from_state_dict(state_dict, dataset_folder=dataset_folder)
    return dataset

def bilingual_dataset_factory(language_high="german", language_low="middle-high-german", mode_low="train-low", word_encoding="none", dataset_pkl_folder=SIG2019_DATASET_PATH, dataset_folder=SIG2019_DATA_PATH, force_rebuild=False) -> BilingualDataset:
    filepath = join(dataset_pkl_folder, f"{language_high}--{language_low}-{mode_low}-{word_encoding}.tch")
    if force_rebuild or not exists(filepath):
        dataset = BilingualDataset(language_high=language_high, language_low=language_low, mode_low=mode_low, word_encoding=word_encoding, dataset_folder=dataset_folder)
        state_dict = dataset.state_dict()
        save(state_dict, filepath)
    else:
        state_dict = load(filepath)
        dataset = BilingualDataset.from_state_dict(state_dict, dataset_folder=dataset_folder)
    return dataset

if __name__ == "__main__":
    from time import process_time

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
        features += [{"word_encoding": "none", "language": lang, "mode": "dev"} for lang in SIG2019_LOW]
        features += [{"word_encoding": "none", "language": lang, "mode": "test"} for lang in SIG2019_LOW]
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
        df.to_csv(join(SIG2019_DATASET_PATH, "none-summary.csv"))
        df_sns = pandas.DataFrame.from_records(records_sns)
    
        import seaborn as sns
        import matplotlib.pyplot as plt
        fig = sns.catplot(x="language", y="value", data=df_sns[
            df_sns["language"].apply(lambda l: l in SIG2019_HIGH) & df_sns["mode"].apply(lambda l: l in SIG2019_HIGH_MODES)
            ], row="element", kind="bar", sharex="row", sharey="none", aspect=10)
        fig.savefig(join(SIG2019_DATASET_PATH, "none-summary-high.png"))
        fig = sns.catplot(x="language", y="value", data=df_sns[
            df_sns["language"].apply(lambda l: l in SIG2019_LOW) & df_sns["mode"].apply(lambda l: l in SIG2019_LOW_MODES)
            ], row="element", col="mode", kind="bar", sharex="row", sharey="none", aspect=10)
        fig.savefig(join(SIG2019_DATASET_PATH, "none-summary-low.png"))
    
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
        df.to_csv(join(SIG2019_DATASET_PATH, "none-summary.csv"))
        df_sns = pandas.DataFrame.from_records(records_sns)
    
        import seaborn as sns
        import matplotlib.pyplot as plt
        fig = sns.catplot(x="language_pair", y="value", data=df_sns,
            row="element", kind="bar", sharex="row", sharey="row",
            hue="mode_low", aspect=20)
        #plt.show()
        fig.savefig(join(SIG2019_DATASET_PATH, "none-summary-bilingual.png"))
        #fig = sns.catplot(x="language", y="value", data=df_sns[
        #    df_sns["language"].apply(lambda l: l in SIG2019_LOW) & df_sns["mode"].apply(lambda l: l in SIG2019_LOW_MODES)
        #    ], row="element", col="mode", kind="bar", sharex="row", sharey="none", aspect=10)
        #fig.savefig(join(SIG2019_DATASET_PATH, "none-summary-low.png"))

    compare_languages_bilingual()