
from torch import zeros, LongTensor, load, save
from torch.utils.data import Dataset
import torch.nn as nn
from .config import SIG2016_LANGUAGES, SIG2016_PATH, SIG2016_MODES, SIG2016_DATASET_PATH
from os.path import exists, join
from datetime import datetime

def load_data(language="german", mode="train", task=2, dataset_folder=SIG2016_PATH):
    '''Load the data from the sigmorphon files in the form of a list of triples (lemma, target_features, target_word).'''
    assert language in SIG2016_LANGUAGES, f"Language '{language}' is unkown, allowed languages are {SIG2016_LANGUAGES}"
    assert mode in SIG2016_MODES, f"Mode '{mode}' is unkown, allowed modes are {SIG2016_MODES}"
    if language == 'japanese':
        assert mode == 'train', f"Mode '{mode}' is unkown for Japanese, the only allowed mode is 'train'"
    filename = f"{language}-task{task}-{mode}"
    with open(join(dataset_folder, filename), "r", encoding="utf-8") as f:
        return [line.strip().split('\t') for line in f]

class Task2Dataset(Dataset):
    """A dataset class for manipultating files of task 2 of Sigmorphon2016.
    
    Not used in the article."""
    def __init__(self, language="german", mode="train", feature_encoding = "char", word_encoding="none", dataset_folder=SIG2016_PATH):
        super(Task2Dataset).__init__()
        self.language = language
        self.mode = mode
        self.feature_encoding = feature_encoding
        self.word_encoding = word_encoding
        self.raw_data = load_data(language = language, mode = mode, task = 2, dataset_folder=dataset_folder)

        self.prepare_data()

    def prepare_data(self):
        """Generate embeddings for the 4 elements.

        There are 3 modes to encode the features:
        - 'feature-value': sequence of each indivitual feature, wrt. a dictioanry of values;
        - 'sum': the one-hot vectors derived using 'feature-value' are summed, resulting in a vector of dimension corresponding to the number of possible values for all the possible features;
        - 'char': sequence of ids of characters, wrt. a dictioanry of values.

        There are 2 modes to encode the words:
        - 'glove': [only for German] pre-trained GloVe embedding of the word;
        - 'char': sequence of ids of characters, wrt. a dictioanry of values;
        - 'none' or None: no encoding, particularly useful when coupled with BERT encodings.
        """
        if self.feature_encoding == "char":
            # generate character vocabulary
            voc = set()
            for feature_a, word_a, feature_b, word_b in self.raw_data:
                voc.update(feature_a)
                voc.update(feature_b)
            self.feature_voc = list(voc)
            self.feature_voc.sort()
            self.feature_voc_id = {character: i for i, character in enumerate(self.feature_voc)}

        elif self.feature_encoding == "feature-value" or self.feature_encoding == "sum":
            # generate feature-value vocabulary
            voc = set()
            for feature_a, word_a, feature_b, word_b in self.raw_data:
                voc.update(feature_a.split(","))
                voc.update(feature_b.split(","))
            self.feature_voc = list(voc)
            self.feature_voc.sort()
            self.feature_voc_id = {character: i for i, character in enumerate(self.feature_voc)}
        else:
            print(f"Unsupported feature encoding: {self.feature_encoding}")


        if self.word_encoding == "char":
            # generate character vocabulary
            voc = set()
            for feature_a, word_a, feature_b, word_b in self.raw_data:
                voc.update(word_a)
                voc.update(word_b)
            self.char_voc = list(voc)
            self.char_voc.sort()
            self.char_voc_id = {character: i for i, character in enumerate(self.char_voc)}

        elif self.word_encoding == "glove":
            from embeddings.glove import GloVe
            self.glove = GloVe()

        elif self.word_encoding == "none" or self.word_encoding is None:
            pass

        else:
            print(f"Unsupported word encoding: {self.word_encoding}")

    def encode_word(self, word):
        if self.word_encoding == "char":
            return LongTensor([self.char_voc_id[c] for c in word])
        elif self.word_encoding == "glove":
            return self.glove.embeddings.get(word, zeros(300))
        elif self.word_encoding == "none" or self.word_encoding is None:
            return word
        else:
            raise ValueError(f"Unsupported word encoding: {self.word_encoding}")
    def encode_feature(self, feature):
        if self.feature_encoding == "char":
            return LongTensor([self.feature_voc_id[c] for c in feature])
        elif self.feature_encoding == "feature-value" or self.feature_encoding == "sum":
            feature_enc = LongTensor([self.feature_voc_id[feature] for feature in feature.split(",")])
            if self.feature_encoding == "sum":
                feature_enc = nn.functional.one_hot(feature_enc, num_classes=len(self.feature_voc_id)).sum(dim=0)
            return feature_enc
        else:
            raise ValueError(f"Unsupported feature encoding: {self.feature_encoding}")
    def encode(self, feature_a, word_a, feature_b, word_b):
        return self.encode_feature(feature_a), self.encode_word(word_a), self.encode_feature(feature_b), self.encode_word(word_b)

    def decode_word(self, word):
        if self.word_encoding == "char":
            return "".join([self.char_voc[char.item()] for char in word])
        elif self.word_encoding == "glove":
            print("Word decoding not supported with GloVe.")
        elif self.word_encoding == "none" or self.word_encoding is None:
            print("Word decoding not necessary when using 'none' encoding.")
            return word
        else:
            print(f"Unsupported word encoding: {self.word_encoding}")

    def decode_feature(self, feature):
        if self.word_encoding == "char":
            return "".join([self.feature_voc[char.item()] for char in feature])
        elif self.feature_encoding == "feature-value":
            return "".join([self.feature_voc[f.item()] for f in feature])
        elif self.feature_encoding == "sum":
            print("Feature decoding not supported with 'sum' encoding.")
        else:
            print(f"Unsupported feature encoding: {self.feature_encoding}")

    def __len__(self): return len(self.raw_data)
    def __getitem__(self, index): return self.encode(*self.raw_data[index])
    def words(self):
        for feature_a, word_a, feature_b, word_b in self:
            yield word_a
            yield word_b
    def features(self):
        for feature_a, word_a, feature_b, word_b in self:
            yield feature_a
            yield feature_b

    def get_vocab(self):
        return set(w for w1w2 in ((w1, w2) for feature1,w1,feature2,w2 in self.raw_data) for w in w1w2)

class Task1Dataset(Dataset):
    @staticmethod
    def from_state_dict(state_dict, dataset_folder=SIG2016_PATH):
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

    """A dataset class for manipultating files of task 1 of Sigmorphon2016."""
    def __init__(self, language="german", mode="train", word_encoding="none", loading=False, dataset_folder=SIG2016_PATH, **kwargs):
        super(Task1Dataset).__init__()
        self.language = language
        self.mode = mode
        self.word_encoding = word_encoding
        self.raw_data = load_data(language=language, mode=mode, task=1, dataset_folder=dataset_folder)

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
        - 'glove': [only for German if model was downloaded] pre-trained GloVe embedding of the word;
        - 'char': sequence of ids of characters, wrt. a dictioanry of values;
        - 'none' or None: no encoding, particularly useful when coupled with BERT encodings.
        """
        if self.word_encoding == "char":
            # generate character vocabulary
            voc = set()
            for word_a, feature_b, word_b in self.raw_data:
                voc.update(word_a)
                voc.update(word_b)
            self.char_voc = list(voc)
            self.char_voc.sort()
            self.char_voc_id = {character: i for i, character in enumerate(self.char_voc)}

        elif self.word_encoding == "glove":
            from embeddings.glove import GloVe
            self.glove = GloVe()

        elif self.word_encoding == "none" or self.word_encoding is None:
            pass

        else:
            print(f"Unsupported word encoding: {self.word_encoding}")

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
        elif self.word_encoding == "glove":
            print("Word decoding not supported with GloVe.")
        elif self.word_encoding == "none" or self.word_encoding is None:
            print("Word decoding not necessary when using 'none' encoding.")
            return word
        else:
            print(f"Unsupported word encoding: {self.word_encoding}")

    def __len__(self):
        return len(self.analogies)

    def __getitem__(self, index):
        ab_index, cd_index = self.analogies[index]
        a, feature_b, b = self.raw_data[ab_index]
        c, feature_d, d = self.raw_data[cd_index]
        return self.encode(a, b, c, d)

def dataset_factory(language="german", mode="train", word_encoding="none", dataset_pkl_folder=SIG2016_DATASET_PATH, dataset_folder=SIG2016_PATH, force_rebuild=False) -> Task1Dataset:
    filepath = join(dataset_pkl_folder, f"{language}-{mode}-{word_encoding}.tch")
    if force_rebuild or not exists(filepath):
        if mode != "train":
            train_dataset = dataset_factory(language=language, mode="train", word_encoding=word_encoding, dataset_pkl_folder=dataset_pkl_folder, dataset_folder=dataset_folder, force_rebuild=force_rebuild)
            state_dict = train_dataset.state_dict()
            state_dict["mode"] = mode
            dataset = Task1Dataset(loading=True, dataset_folder=dataset_folder, **state_dict)
            dataset.set_analogy_classes()
        else:
            dataset = Task1Dataset(language=language, mode=mode, word_encoding=word_encoding, dataset_folder=dataset_folder)
        state_dict = dataset.state_dict()
        save(state_dict, filepath)
    else:
        state_dict = load(filepath)
        dataset = Task1Dataset.from_state_dict(state_dict, dataset_folder=dataset_folder)
    return dataset

if __name__ == "__main__":
    dataset = dataset_factory()
    print(len(dataset.analogies))
    print(dataset[2500])
    print(len(Task2Dataset()))
    print(Task2Dataset()[2500])
    print(Task2Dataset().raw_data[2500])