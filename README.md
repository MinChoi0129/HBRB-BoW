# HBRB-BoW

A C++ library for converting and indexing image features into a bag-of-words representation.
Modified based on the [original DBoW2](https://github.com/dorian3d/DBoW2).

## Core File

The core functionality is implemented in `include/DBoW2/TemplatedVocabulary.h`.

## Vocabulary Training

To train a vocabulary yourself, build and run `executable/create_vocabulary.cpp` with the Bovisa dataloader located in `dataloaders`.

```bash
./create_vocabulary <kmeans_type>
```

| kmeans_type | Description |
|-------------|-------------|
| `0` | Default DBoW2 k-means |
| `1` | HBRB-KMeans (local BRB) |
| `2` | HBRB-KMeans (global BRB, **proposed algorithm**) |

> **Note**: The training process requires a large amount of system memory. Please ensure sufficient memory is available before running.

## Vocabulary File Format

Vocabulary files are saved in `.bin` binary format for efficient saving and loading.

## Dataset

The exact name of the training dataset is **Bovisa_2008_09_01_Frontal**.
Due to the inconvenience of accessing the original dataset, we provide it directly. Please request it via the email below.

[![Email](https://img.shields.io/badge/Email-wjdchs0129%40gnu.ac.kr-0078D4?style=flat-square&logo=microsoftoutlook&logoColor=white)](mailto:wjdchs0129@gnu.ac.kr)

## Citation

If you need to cite this work, please refer to the link or bibtex below.

[![arXiv](https://img.shields.io/badge/arXiv-2305.06500v2-b31b1b.svg)]([https://arxiv.org/](https://arxiv.org/pdf/2603.04144))

```bibtex
@misc{2603.04144,
Author = {Minjae Lee and Sang-Min Choi and Gun-Woo Kim and Suwon Lee},
Title = {HBRB-BoW: A Retrained Bag-of-Words Vocabulary for ORB-SLAM via Hierarchical BRB-KMeans},
Year = {2026},
Eprint = {arXiv:2603.04144},
}
```
