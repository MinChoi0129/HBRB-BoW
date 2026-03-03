/**
 * File: bovisa_dataloader.h
 * Description: Bovisa dataset loader
 */

#ifndef BOVISA_DATALOADER_H
#define BOVISA_DATALOADER_H

#include <string>
#include <vector>

class BovisaDataLoader {
public:
  BovisaDataLoader(const std::string &dataset_path, int num_images, int random_seed);
  ~BovisaDataLoader() = default;

  // Get total number of images to load
  size_t getNumImages() const { return selected_image_paths_.size(); }

  // Get image path by index
  std::string getImagePath(size_t index) const;

  // Get all selected image paths
  const std::vector<std::string> &getAllImagePaths() const { return selected_image_paths_; }

private:
  std::string dataset_path_;
  std::vector<std::string> all_image_paths_;
  std::vector<std::string> selected_image_paths_;
  int num_images_;
  int random_seed_;

  void loadAllImagePaths();
  void selectRandomImages();
  bool isValidImageFile(const std::string &filename) const;
};

#endif // BOVISA_DATALOADER_H
