
from abc import abstractmethod, ABC, abstractstaticmethod, abstractproperty
from torch import zeros, LongTensor, load, save
from torch.utils.data import Dataset
import torch.nn as nn
from os.path import exists, join
from datetime import datetime
import typing as t

class AbstractDataset(Dataset, ABC):
    """An abstract dataset class for analogies datasets."""

    @abstractstaticmethod
    def from_state_dict(state_dict): pass
    @abstractmethod
    def state_dict(self): pass
    @abstractmethod
    def set_analogy_classes(self): pass
    @abstractmethod
    def __len__(self): pass
    @abstractmethod
    def __getitem__(self, index): pass

class AbstractFeatureDataset(ABC):
    features: t.Iterable

    def __init__(self, feature_encoding = "none"):
        self.feature_encoding = feature_encoding

    def prepare_feature_encoding(self):
        """Generate embeddings for the 4 elements.

        There are 3 modes to encode the features:
        - 'feature-value': sequence of each indivitual feature, wrt. a dictioanry of values;
        - 'sum': the one-hot vectors derived using 'feature-value' are summed, resulting in a vector of dimension corresponding to the number of possible values for all the possible features;
        - 'char': sequence of ids of characters, wrt. a dictioanry of values.
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

    def decode_feature(self, feature):
        if self.word_encoding == "char":
            return "".join([self.feature_voc[char.item()] for char in feature])
        elif self.feature_encoding == "feature-value":
            return "".join([self.feature_voc[f.item()] for f in feature])
        elif self.feature_encoding == "sum":
            print("Feature decoding not supported with 'sum' encoding.")
        else:
            print(f"Unsupported feature encoding: {self.feature_encoding}")
    
class AbstractWordDataset(ABC):
    word_voc: t.Iterable

    def __init__(self, word_encoding="none"):
        self.word_encoding = word_encoding

    def prepare_word_encoding(self):
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

"""In __init__(...):

    AbstractDataset.__init__(self)
    AbstractWordDataset.__init__(self, word_encoding=word_encoding)
    AbstractFeatureDataset.__init__(feature_encoding=feature_encoding)
"""