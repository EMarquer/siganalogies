# `siganalogies` for morphological analogies using Sigmorphon 2016 and 2019
The `siganalogies` package is design to manipulate morphological analogies built upon Sigmorphon 2016 and Sigmorphon 2019 in PyTorch.

- [Changelog](#changelog)
- [How to cite](#how-to-cite)
- [References](#references)
- [Requirements](#requirements)
- [Setup](#setup)
  - [[OPTIONAL] Get pre-computed dataset files for analysis or faster loading](#optional-get-pre-computed-dataset-files-for-analysis-or-faster-loading)
- [Basic usage](#basic-usage)
  - [Dataset object](#dataset-object)
  - [Factories](#factories)
    - [Generic factory](#generic-factory)
    - [Dataset-specific factories](#dataset-specific-factories)
  - [Data augmentation](#data-augmentation)
    - [Augmented forms (i.e. permutations)](#augmented-forms-ie-permutations)
    - [Augmented forms (i.e. permutations) with central permutation not accepted as a property of analogy](#augmented-forms-ie-permutations-with-central-permutation-not-accepted-as-a-property-of-analogy)
- [Dataset description](#dataset-description)
- [Publications using this dataset](#publications-using-this-dataset)
- [[NOT RECOMMENDED] Minimal usage with `pickle` and dataset serialization files](#not-recommended-minimal-usage-with-pickle-and-dataset-serialization-files)

## Changelog
Comming soon:
- bilingual datasets (analogies between two similar languages);
- examples of usage;
- spliting of the data folowing the procédure of the articles by Alsaidi *et al.*

## How to cite
To cite this dataset, use the following reference:
- For this dataset:
  ```bib
  @misc{
      To be completed in the very near future.
  }
  ```

## References
The references of the datasets on wich the present dataset bases itself are as follow:
- For Sigmorphon 2016:
  ```bib
  @InProceedings{cotterell2016sigmorphon,
    author    = {Cotterell, Ryan and Kirov, Christo and Sylak-Glassman, John and Yarowsky, David and Eisner, Jason and Hulden, Mans},
    title     = {The {SIGMORPHON} 2016 Shared Task---Morphological Reinflection},
    booktitle = {Proceedings of the 2016 Meeting of {SIGMORPHON}},
    month     = {August},
    year      = {2016},
    address   = {Berlin, Germany},
    publisher = {Association for Computational Linguistics}
  }
  ```
- For the Japanese data added to Sigmorphon 2016, from the Japanese Bigger Analogy Test Set:
  ```bib
  @inproceedings{jap-data:2018:karpinska,
    author = {Marzena Karpinska and Bofang Li and Anna Rogers and Aleksandr Drozd},
    title = {Subcharacter Information in Japanese embeddings: when is it worth it?},
    year = {2018},
    booktitle = {Workshop on the Relevance of Linguistic Structure in Neural Architectures for NLP},
    address = {Melbourne, Australia},
    pages = {28-37},
    publisher = {ACL}
  }
  ```
- For Sigmorphon 2019:
  ```bib
  @inproceedings{mccarthy-etal-2019-sigmorphon,
    title = "The {SIGMORPHON} 2019 Shared Task: Morphological Analysis in Context and Cross-Lingual Transfer for Inflection",
    author = "McCarthy, Arya D.  and
      Vylomova, Ekaterina  and
      Wu, Shijie  and
      Malaviya, Chaitanya  and
      Wolf-Sonkin, Lawrence  and
      Nicolai, Garrett  and
      Kirov, Christo  and
      Silfverberg, Miikka  and
      Mielke, Sabrina J.  and
      Heinz, Jeffrey  and
      Cotterell, Ryan  and
      Hulden, Mans",
    booktitle = "Proceedings of the 16th Workshop on Computational Research in Phonetics, Phonology, and Morphology",
    month = aug,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/W19-4226",
    doi = "10.18653/v1/W19-4226",
    pages = "229--244",
  }
  ```

## Requirements
Python >= 3.8 and PyTorch are required for `siganalogies`.
The code was designed with PyTorch 1.10, but most versions equiped with the `torch.utils.data.Dataset` class should work.

## Setup
The recommended file structure is as follows:
```
my_project/
├── siganalogies/ (this package)
│   ├── precomputed/ (serialized datasets)
│   │   ├── 2016
│   │   └── 2019
│   ├── __init__.py
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

This can be done using the following commands, to be run from your project directory.
```bash
git clone git@github.com:EMarquer/siganalogies
git clone git@github.com:ryancotterell/sigmorphon2016 sigmorphon2016
git clone git@github.com:sigmorphon/2019.git sigmorphon2019
mv siganalogies/japanese-task1-train sigmorphon2016/japanese-task1-train
```

If you are in a repository, we recommend that you use `git submodule add` instead of `git clone`.

### [OPTIONAL] Get pre-computed dataset files for analysis or faster loading

WILL BE DONE SOON

## Basic usage
To manipulate the analogies, you will first need to load a [dataset object](#dataset-object) using the [dataset factories](#factories).
You will then be able to use the dataset as any other `torch.utils.data.Dataset`, each element being a quadruple.

The dataset object contains analogies of the form $A:B::C:D$ and $A:B::A:B$, but not the corresponding [augmented forms (i.e. permutations)](#augmented-forms-ie-permutations). We recommend to apply [data augmentation](#data-augmentation) to add said [augmented forms](#augmented-forms-ie-permutations).

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

If you do not want to use serialized dataset (hence recompute the analogies each time) you can specify `serialize=False` to the factory function.
Unless you also specify `force_rebuild=True`, the existing serialized datasets will still be used.

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
- `siganalogies.config.SIG2016_DATASET_PATH="./precomputed/2016"`
- `siganalogies.config.SIG2019_DATASET_PATH="./precomputed/2019"`
To disable precomputed dataset files once, specify `serialization=False` in the factory. To disable dataset files globaly, change `SERIALIZATION=True` to `SERIALIZATION=False` in `siganalogies/config.py`

Specify `dataset_folder` if you do not use the recommended structure to store the datasets. 
Defaults are:
- `siganalogies.config.SIG2016_PATH="../sigmorphon2016/data"`
- `siganalogies.config.SIG2019_PATH="../sigmorphon2019/task1"`

For all the datasets, `word_encoding` can be either `"char"` for character-based encoding, or either `None` or `"none"`, if no encoding is applied and the raw text data is returned.

For **Sigmorphon 2016**, the available `language`s are listed in `siganalogies.config.SIG2016_LANGUAGES`. The available `mode`s are `train`, `dev`, and `test`, also listed in `siganalogies.config.SIG2016_MODES`.

```python
from siganalogies import dataset_factory_2016, SIG2016_DATASET_PATH, SIG2016_PATH, SERIALIZATION

dataset = dataset_factory(
    language="german",
    mode="train",
    word_encoding="none",
    dataset_pkl_folder=SIG2016_DATASET_PATH,
    dataset_folder=SIG2016_PATH,
    force_rebuild=False,
    serialization=SERIALIZATION)
```

For **Sigmorphon 2019**, the available `language`s are split in two categories:
- high ressource languages, listed in `siganalogies.config.SIG2019_HIGH`, and the corresponding `mode` are `train-high`, also listed in `siganalogies.config.SIG2019_HIGH_MODES`;
- low ressource languages, listed in `siganalogies.config.SIG2019_LOW`, and the corresponding `mode`s are `train`, `dev`, and `test`, also listed in `siganalogies.config.SIG2019_LOW_MODES`.

```python
from siganalogies import dataset_factory_2019, SIG2019_DATASET_PATH, SIG2019_PATH, SERIALIZATION

dataset = dataset_factory(
    language="german",
    mode="train-high",
    word_encoding="none",
    dataset_pkl_folder=SIG2019_DATASET_PATH,
    dataset_folder=SIG2019_PATH,
    force_rebuild=False,
    serialization=SERIALIZATION)
```

### Data augmentation
To be completed.

Further explanations can be found in the article **To be completed**

#### Augmented forms (i.e. permutations)
From $A:B::C:D$ we can have the following equivalent permutations (or forms):
- $A:B::C:D$ (the base form)
- $A:C::B:D$
- $B:A::D:C$
- $B:D::A:C$
- $C:D::A:B$
- $C:A::D:B$
- $D:C::B:A$
- $D:B::C:A$

We can also compute forms which should not be valid analogies, by permuting each analogical form $A:B::C:D$ above to obtain:
- $A:A::C:D$
- $B:A::C:D$
- $C:B::A:D$


#### Augmented forms (i.e. permutations) with central permutation not accepted as a property of analogy
To be completed.

## Dataset description
See [`dataset_description.pdf`](dataset_description.pdf).

## Publications using this dataset
To be completed.

## [NOT RECOMMENDED] Minimal usage with `pickle` and dataset serialization files
It is possible to acess the pre-comuted data directly, using the following:
```python
from pickle import load

data = load(open("precomputed/2019/adyghe-train-high-none.pkl", "rb"))
```

**The pickled data files do not contain the data itself**, even if they are usable without the actual data from Sigmorphon 2016 and Sigmorphon 2019.

The pickled data files follow the following pattern: `<language>-<mode>-<word encoding>.pkl`.

`data` will be a dictionary with the following keys:
- `timestamp`: the timestamp of the creation of the pickled data;
- all the key attributes of a dataset object:
  `language`, `mode`, `word_encoding`, `char_voc` and `char_voc_id` (if `word_encoding` is `char`), `features`,`word_voc`, `features_with_analogies`, and `words_with_analogies`.