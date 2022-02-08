# `siganalogies` for morphological analogies using Sigmorphon 2016 and 2019
The `siganalogies` package is design to manipulate morphological analogies built upon Sigmorphon 2016 and Sigmorphon 2019 in PyTorch.

To cite this dataset, use the following:
```bib
@misc{
    To be completed in the very near future.
}
```

- [Setup](#setup)
- [Basic usage](#basic-usage)
  - [Dataset object](#dataset-object)
  - [Factories](#factories)
    - [Generic factory](#generic-factory)
    - [Dataset-specific factories](#dataset-specific-factories)
  - [Data augmentation](#data-augmentation)
- [Dataset description](#dataset-description)
  - [Sigmorphon 2019](#sigmorphon-2019)
  - [Sigmorphon 2016](#sigmorphon-2016)
- [Publications using this dataset](#publications-using-this-dataset)
- [[NOT RECOMMENDED] Minimal usage and dataset PyTorch pickle description](#not-recommended-minimal-usage-and-dataset-pytorch-pickle-description)

## Setup
The recommended file structure is as follows:
```
my_project/
├── siganalogies/ (this package)
│   ├── 2016_precomuted
│   ├── 2019_precomuted
│   └── ...
├── sigmorphon2016/
│   ├── data
│   └── ...
├── sigmorphon2019/
│   ├── task1
│   └── ...
└── ...
```

To use, either copy all the files in a folder named `siganalogies` or clone this repository into your project (typically as a submodule).

You will also need to clone Sigmorphon 2016 and Sigmorphon 2019 into corresponding folders.

This can be done using the following commands.
```bash
git clone git@github.com:EMarquer/siganalogies
git clone git@github.com:ryancotterell/sigmorphon2016 sigmorphon2016
git clone git@github.com:sigmorphon/2019.git sigmorphon2019
```

If you are in a repository, use `git submodule clone` instead of `git clone`.

## Basic usage
### Dataset object
Dataset objects are subclasses of `torch.utils.data.Dataset` and should be created using [factories](#factories).

A dataset object has the following attributes:
- `language`: the dataset language;
- `mode`: the subset of the language data, using the seperation done in Sigmorphon (for Sigmorphon 2016: `train`, `dev`, `test`; for Sigmorphon 2019 high ressource languages: `train-high`; for Sigmorphon 2019 low ressource languages: `train-low`, `dev`, `test`; test-covered is supported for neither dataset);
- `word_encoding`: the word encoding strategy, used by the dataset object (`char` to encode using character IDs based on the characters of the training dataset, or `none` to return the text itself, which is usefull when using a custom encoding strategy);
  - if `word_encoding` is `char`, two attributes are presents: `char_voc` a list of characters found in the dataset and `char_voc_id` a dictionary linking characters of `char_voc` to their index in the list;
- several other statistic-oriented attributes are computed when building the dataset:
  - `features`: a list of all the features in the dataset;
  - `word_voc`: a list of all the words in the dataset;
  - `features_with_analogies`: a list of all the features in the dataset which are present in at least one analogy;
  - `words_with_analogies`: a list of all the words in the dataset which are present in at least one analogy.

A dataset object has the key methods of a PyTorch Dataset object (`__len__` and `__getitem__`) as well as the following methods:
- `encode_word`, to encode a string according to the `word_encoding` strategy;
- `encode`, a wrapper to encode four words at once;
- `decode_word`, to decode a string according to the `word_encoding` strategy.

Finally, other methods are available, none of which should be used in a standard usage (trust the factories to deal with things accordingly):
- `state_dict`, and the corresponding static method `Dataset.from_state_dict`, used in background by the factories to save and load datasets;
- `prepare_data` to prepare the word encoding;
- `set_analogy_classes` to compute the analogical pairs;

### Factories
Manually creating a dataset using its `__init__` call is not recommended, use the [generic factory](#generic-factory) or the [dataset-specific factories](#dataset-specific-factories), as they handle saving and loading the data.

If you are unsure the data saved is correct, you can specify `force_rebuild=True` to the factory function.

#### Generic factory
The generic factory is used to unify calls to dataset-specific factories.
Typical usage is when Sigmorphon 2016 and Sigmorphon 2019 will be used interchangabely.
To use the generic factory, refer to the datataset specific explanation of the key-word arguments.

```python
from siganalogies import dataset_factory

dataset = dataset_factory(dataset="2016", **kwargs)
```
#### Dataset-specific factories
Specify `dataset_pkl_folder` if you do not use the recommended structure to store the precomputed dataset files.
Defaults are: 
- `siganalogies.config.SIG2016_DATASET_PATH="2016_precomputed"`
- `siganalogies.config.SIG2019_DATASET_PATH="2019_precomputed"`
Specify `dataset_folder` if you do not use the recommended structure to store the datasets. 
Defaults are:
- `siganalogies.config.SIG2016_PATH="../sigmorphon2016/data"`
- `siganalogies.config.SIG2019_DATA_PATH="../sigmorphon2019/task1"`

For all the datasets, `word_encoding` can be either `"char"` for character-based encoding, or either `None` or `"none"`, if no encoding is applied and the raw text data is returned.

For **Sigmorphon 2016**, the available `language`s are listed in `siganalogies.config.SIG2016_LANGUAGES`. The available `mode`s are `train`, `dev`, and `test`, also listed in `siganalogies.config.SIG2016_MODES`.

```python
from siganalogies import dataset_factory_2016, SIG2016_DATASET_PATH

dataset = dataset_factory(
    language="german",
    mode="train",
    word_encoding="none",
    dataset_pkl_folder=SIG2016_DATASET_PATH,
    dataset_folder=SIG2016_PATH,
    force_rebuild=False)
```

For **Sigmorphon 2019**, the available `language`s are split in two categories:
- high ressource languages, listed in `siganalogies.config.SIG2019_HIGH`, and the corresponding `mode` are `train-high`, also listed in `siganalogies.config.SIG2019_HIGH_MODES`;
- low ressource languages, listed in `siganalogies.config.SIG2019_LOW`, and the corresponding `mode`s are `train`, `dev`, and `test`, also listed in `siganalogies.config.SIG2019_LOW_MODES`.

```python
from siganalogies import dataset_factory_2019, SIG2019_DATASET_PATH

dataset = dataset_factory(
    language="german",
    mode="train-high",
    word_encoding="none",
    dataset_pkl_folder=SIG2019_DATASET_PATH,
    dataset_folder=SIG2019_DATA_PATH,
    force_rebuild=False)
```

### Data augmentation



## Dataset description
### Sigmorphon 2019
### Sigmorphon 2016

## Publications using this dataset
To be completed.

## [NOT RECOMMENDED] Minimal usage and dataset PyTorch pickle description
It is possible to acess the pre-comuted data directly, using the following:
```python
from torch import load

data = load("data/2019_precomputed/adyghe-train-high-none.tch")
```

**The pickled data files do not contain the data itself**, even if they are usable without the actual data from Sigmorphon 2016 and Sigmorphon 2019.

The pickled data files follow the following pattern: `<language>-<mode>-<word encoding>.tch`.

`data` will be a dictionary with the following keys:
- `timestamp`: the timestamp of the creation of the pickled data;
- all the key attributes of a dataset object:
  `language`, `mode`, `word_encoding`, `char_voc` and `char_voc_id` (if `word_encoding` is `char`), `features`,`word_voc`, `features_with_analogies`, and `words_with_analogies`.