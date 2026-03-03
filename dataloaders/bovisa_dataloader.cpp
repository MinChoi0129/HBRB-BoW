/**
 * File: bovisa_dataloader.cpp
 * Description: Bovisa dataset loader implementation
 */

#include "bovisa_dataloader.h"
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <random>

namespace fs = std::filesystem;

BovisaDataLoader::BovisaDataLoader(const std::string &dataset_path,
                                   int num_images, int random_seed)
    : dataset_path_(dataset_path), num_images_(num_images),
      random_seed_(random_seed) {
  loadAllImagePaths();
  selectRandomImages();
}

void BovisaDataLoader::loadAllImagePaths() {
  all_image_paths_.clear();

  if (!fs::exists(dataset_path_) || !fs::is_directory(dataset_path_)) {
    std::cerr << "Error: Dataset path does not exist or is not a directory: "
              << dataset_path_ << std::endl;
    return;
  }

  // Load all image files
  for (const auto &entry : fs::recursive_directory_iterator(dataset_path_)) {
    if (entry.is_regular_file()) {
      std::string filename = entry.path().filename().string();
      if (isValidImageFile(filename)) {
        all_image_paths_.push_back(entry.path().string());
      }
    }
  }

  std::cout << "Found " << all_image_paths_.size()
            << " images in Bovisa dataset" << std::endl;
}

void BovisaDataLoader::selectRandomImages() {
  if (all_image_paths_.empty()) {
    std::cerr << "Error: No images found in dataset" << std::endl;
    return;
  }

  // Initialize random number generator with fixed seed
  std::mt19937 rng(random_seed_);

  // If we need more images than available, use all available
  int num_to_select =
      std::min(num_images_, static_cast<int>(all_image_paths_.size()));

  // Create a shuffled copy of indices
  std::vector<size_t> indices(all_image_paths_.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::shuffle(indices.begin(), indices.end(), rng);

  // Select first num_to_select images
  selected_image_paths_.clear();
  selected_image_paths_.reserve(num_to_select);
  for (int i = 0; i < num_to_select; ++i) {
    selected_image_paths_.push_back(all_image_paths_[indices[i]]);
  }

  // Sort selected paths for consistent ordering
  std::sort(selected_image_paths_.begin(), selected_image_paths_.end());

  std::cout << "Selected " << selected_image_paths_.size()
            << " images for training" << std::endl;
}

bool BovisaDataLoader::isValidImageFile(const std::string &filename) const {
  // Check common image extensions
  std::string lower_filename = filename;
  std::transform(lower_filename.begin(), lower_filename.end(),
                 lower_filename.begin(), ::tolower);

  // C++17 compatible: use substr instead of ends_with (C++20)
  size_t len = lower_filename.length();
  return (len >= 4 && lower_filename.substr(len - 4) == ".jpg") ||
         (len >= 5 && lower_filename.substr(len - 5) == ".jpeg") ||
         (len >= 4 && lower_filename.substr(len - 4) == ".png") ||
         (len >= 4 && lower_filename.substr(len - 4) == ".bmp");
}

std::string BovisaDataLoader::getImagePath(size_t index) const {
  if (index >= selected_image_paths_.size()) {
    std::cerr << "Error: Image index out of range: " << index << std::endl;
    return "";
  }
  return selected_image_paths_[index];
}
