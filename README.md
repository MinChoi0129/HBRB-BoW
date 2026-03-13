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
