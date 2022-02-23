from torch import LongTensor, Tensor
import typing

from .none import Encoder, StateDict

class CharEncoder(Encoder):
    id_to_char: typing.Tuple[str]
    char_to_id: typing.Dict[str, int]
    type="char"

    @staticmethod
    def from_state_dict(state_dict: StateDict):
        encoder = CharEncoder()
        encoder.id_to_char = state_dict["id_to_char"]
        encoder.char_to_id = {character: i for i, character in enumerate(encoder.id_to_char)}
        return encoder

    def state_dict(self) -> StateDict:
        "Return a data dictionary, loadable for future use of the encoder."
        state_dict = {
            "encoder_type": "char",
            "id_to_char": self.id_to_char
        }
        return state_dict

    def decode(self, tensor: Tensor):
        """Decode a tensor as a word (or list of words if multiple dimensions are detected)."""
        assert hasattr(self,"id_to_char"), "Encoder not initialized."
        
        if tensor.dim > 1:
            return [self.decode(part) for part in tensor]
        else:
            return "".join([self.id_to_char[char.int().item()] for char in tensor])

    def encode(self, word: str):
        """Encode a word."""
        assert hasattr(self,"char_to_id"), "Encoder not initialized."

        return LongTensor([self.char_to_id.get(c, -1) for c in word])

    def init(self, vocabulary: typing.Iterable[str]):
        chars = set()
        for word in vocabulary:
            chars.update(word)

        self.id_to_char = tuple(sorted(chars))
        self.char_to_id = {character: i for i, character in enumerate(self.id_to_char)}
    
    def __hash__(self) -> int:
        return hash(self.id_to_char)
