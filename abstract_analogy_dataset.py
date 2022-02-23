from torch.utils.data import Dataset
from abc import ABC, abstractstaticmethod, abstractmethod
import collections as c

class AbstractAnalogyDataset(Dataset, ABC):
    analogies: c.Sized

    @abstractstaticmethod
    def from_state_dict(state_dict, dataset_folder=None):
        """Create a dataset from saved data."""
        raise NotImplementedError("Should be implemented in subclass.")

    @abstractmethod
    def state_dict(self):
        "Return a data dictionary, loadable for future use of the dataset."
        raise NotImplementedError("Should be implemented in subclass.")

    def __init__(self, **kwargs):
        """A dataset class for manipultating files of task 1 of Sigmorphon2019."""
        super(AbstractAnalogyDataset).__init__()
        raise NotImplementedError("Should be implemented in subclass.")

    @abstractmethod
    def prepare_data(self):
        """Generate embeddings for the 4 elements.

        There are 2 modes to encode the words:
        - 'char': sequence of ids of characters, wrt. a dictioanry of values;
        - 'none' or None: no encoding, particularly useful when coupled with BERT encodings.
        """
        raise NotImplementedError("Should be implemented in subclass.")

    @abstractmethod
    def set_analogy_classes(self):
        """Go through the data to extract the vocabulary, the available features, and build analogies."""
        raise NotImplementedError("Should be implemented in subclass.")

    @abstractmethod
    def encode_word(self, word):
        """Encode a single word using the selected encoding process."""
    
    def encode(self, a, b, c, d):
        """Encode 4 words using the selected encoding process.
        
        A wraper around self.encode_word applied to each of the arguments."""
        return self.encode_word(a), self.encode_word(b), self.encode_word(c), self.encode_word(d)

    @abstractmethod
    def decode_word(self, word):
        """Decode a single word using the selected encoding process."""
        raise NotImplementedError("Should be implemented in subclass.")
    
    def decode(self, a, b, c, d):
        """Decode 4 words using the selected encoding process.
        
        A wraper around self.decode_word applied to each of the arguments."""
        return self.decode_word(a), self.decode_word(b), self.decode_word(c), self.decode_word(d)

    #@abstractmethod
    def __len__(self):
        """Returns the number of analogies built in self.set_analogy_classes."""
        return len(self.analogies)
        #raise NotImplementedError("Should be implemented in subclass.")

    @abstractmethod
    def __getitem__(self, index):
        """Returns the index-th analogy built in self.set_analogy_classes as a quadruple A, B, C, D for a quadruple
        A:B::C:D."""
        raise NotImplementedError("Should be implemented in subclass.")
