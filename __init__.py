from typing import Union

from .config import *
from .utils import enrich, generate_negative, random_sample_negative, n_pos_n_neg

from .sig_2016 import Task1Dataset as Dataset2016
from .sig_2016 import dataset_factory as dataset_factory_2016
from .sig_2019 import Task1Dataset as Dataset2019
from .sig_2019 import \
    bilingual_dataset_factory as bilingual_dataset_factory_2019
from .sig_2019 import dataset_factory as dataset_factory_2019


def dataset_factory(dataset="2016", **kwargs) -> Union[Dataset2016, Dataset2019]:
    if dataset == "2016":
        return dataset_factory_2016(**kwargs)
    else:
        return dataset_factory_2019(**kwargs)
