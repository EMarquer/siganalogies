from .config import SERIALIZATION, SIG2016_LANGUAGES, SIG2016_PATH, SIG2016_MODES, SIG2016_DATASET_PATH
from os.path import exists, join
from datetime import datetime
from .abstract_analogy_dataset import AbstractAnalogyDataset, StateDict, save_state_dict, load_state_dict
import logging
import typing as t
from .encoders import Encoder, encoder_as_string

def load_data(language="german", mode="train", task=2, dataset_folder=SIG2016_PATH):
    '''Load the data from the sigmorphon files in the form of a list of triples (lemma, target_features, target_word).'''
    assert language in SIG2016_LANGUAGES, f"Language '{language}' is unkown, allowed languages are {SIG2016_LANGUAGES}"
    assert mode in SIG2016_MODES, f"Mode '{mode}' is unkown, allowed modes are {SIG2016_MODES}"
    if language == 'japanese':
        assert mode == 'train', f"Mode '{mode}' is unkown for Japanese, the only allowed mode is 'train'"
    filename = f"{language}-task{task}-{mode}"
    with open(join(dataset_folder, filename), "r", encoding="utf-8") as f:
        return [line.strip().split('\t') for line in f]

class Sig2016Dataset(AbstractAnalogyDataset):
    @staticmethod
    def from_state_dict(state_dict: StateDict, dataset_folder=SIG2016_PATH):
        dataset = Sig2016Dataset(building=False, dataset_folder=dataset_folder, state_dict=state_dict,
            word_encoder=state_dict["word_encoder"])
        return dataset

    def state_dict(self) -> StateDict:
        "Return a data dictionary, loadable for future use of the dataset."
        state_dict = {
            "timestamp": datetime.now(),
            "language": self.language,
            "mode": self.mode,
            "word_encoder": self.word_encoder.state_dict(),
            "analogies": self.analogies,
            "word_voc": self.word_voc,
            "features": self.features,
            "features_with_analogies": self.features_with_analogies,
            "words_with_analogies": self.words_with_analogies
        }
        return state_dict

    """A dataset class for manipultating files of task 1 of Sigmorphon2016."""
    def __init__(self, language="german", mode="train", word_encoder: t.Union[t.Type, Encoder, str, None]=None, building=True, state_dict: StateDict=None, dataset_folder=SIG2016_PATH, **kwargs):
        super().__init__(word_encoder=word_encoder)
        self.language = language
        self.mode = mode
        self.raw_data = load_data(language=language, mode=mode, task=1, dataset_folder=dataset_folder)

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

def dataset_factory(language="german", mode="train", word_encoder="none", dataset_pkl_folder=SIG2016_DATASET_PATH, dataset_folder=SIG2016_PATH, force_rebuild=False, serialization=SERIALIZATION) -> Sig2016Dataset:
    filepath = join(dataset_pkl_folder, f"{language}-{mode}-{encoder_as_string(word_encoder)}.pkl")
    if force_rebuild or not exists(filepath):
        logging.info(f"Starting building the dataset {filepath}...")
        if mode != "train":
            logging.info(f"Using the corresponding training dataset for  {filepath}...")
            train_dataset = dataset_factory(language=language, mode="train", word_encoder=word_encoder, dataset_pkl_folder=dataset_pkl_folder, dataset_folder=dataset_folder, force_rebuild=force_rebuild)
            state_dict = train_dataset.state_dict()
            state_dict["mode"] = mode
            dataset = Sig2016Dataset.from_state_dict(state_dict, dataset_folder=dataset_folder)
            logging.info(f"Computing the analogies for {filepath}...")
            dataset.set_analogy_classes()
        else:
            dataset = Sig2016Dataset(language=language, mode=mode, word_encoder=word_encoder, dataset_folder=dataset_folder)
        logging.info(f"Dataset {filepath} built.")
        
        if serialization:
            logging.info(f"Saving the dataset to {filepath}...")
            state_dict = dataset.state_dict()
            save_state_dict(state_dict, filepath)
            logging.info(f"Dataset saved to {filepath}.")
    else:
        state_dict = load_state_dict(filepath)
        dataset = Sig2016Dataset.from_state_dict(state_dict=state_dict, dataset_folder=dataset_folder)
        logging.info(f"Dataset {filepath} loaded.")
    return dataset

if __name__ == "__main__":
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
    #print(len(Task2Dataset()))
    #print(Task2Dataset()[2500])
    #print(Task2Dataset().raw_data[2500])