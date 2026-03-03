/**
 * File: create_vocabulary.cpp
 * Description: Create vocabulary using DBoW2 with ORB descriptors
 */

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <vector>

#include "../dataloaders/bovisa_dataloader.h"
#include "DBoW2.h"

using namespace DBoW2;
using namespace std;
using namespace cv;

void changeStructure(const Mat &plain, vector<Mat> &out) {
  out.resize(plain.rows);
  for (int i = 0; i < plain.rows; ++i) {
    out[i] = plain.row(i);
  }
}

void extractFeatures(const vector<string> &image_paths, vector<vector<Mat>> &features,
                     Ptr<ORB> orb) {
  features.clear();
  features.reserve(image_paths.size());

  cout << "Extracting ORB features from " << image_paths.size() << " images..." << endl;

  for (size_t i = 0; i < image_paths.size(); ++i) {
    Mat image = imread(image_paths[i], IMREAD_GRAYSCALE);
    if (image.empty()) {
      cerr << "Warning: Cannot load image: " << image_paths[i] << endl;
      features.push_back(vector<Mat>());
      continue;
    }

    vector<KeyPoint> keypoints;
    Mat descriptors;
    orb->detectAndCompute(image, noArray(), keypoints, descriptors);

    vector<Mat> image_features;
    if (!descriptors.empty()) {
      changeStructure(descriptors, image_features);
    }
    features.push_back(image_features);

    if ((i + 1) % 1000 == 0) {
      cout << "Processed " << (i + 1) << " / " << image_paths.size() << " images" << endl;
    }
  }

  cout << "Feature extraction completed!" << endl;
}

int main(int argc, char *argv[]) {
  // Check command line arguments
  if (argc < 2) {
    cerr << "Usage: " << argv[0] << " <kmeans_type>" << endl;
    cerr << "  kmeans_type: 0, 1, or 2" << endl;
    return -1;
  }

  // Parse kmeans_type from command line
  int kmeans_type = atoi(argv[1]);
  if (kmeans_type < 0 || kmeans_type > 2) {
    cerr << "Error: kmeans_type must be 0, 1, or 2" << endl;
    return -1;
  }

  // ORB parameters
  int nfeatures = 1000;
  double scaleFactor = 1.2;
  int nlevels = 8;
  int edgeThreshold = 31;
  int firstLevel = 0;
  int WTA_K = 2;
  int scoreType = 0;
  int patchSize = 31;
  int fastThreshold = 20;

  // DBoW2 parameters
  int k = 10;
  int L = 6;
  WeightingType weighting = TF_IDF;
  ScoringType scoring = L1_NORM;

  // Dataset paths
  string bovisa_path = "/home/ssd_data/ROOT_BRBSLAM/Bovisa_2008_09_01_frontal";
  std::cout << "Dataset path : " << bovisa_path << std::endl;

  // Training parameters
  int num_images = 10000; // 10000
  int random_seed = 42;   // 42, 1234, 5678, ...

  // Vocabulary save path
  string voc_save_path;
  if (kmeans_type == 0)
    voc_save_path = "vocabulary_default.bin";
  else if (kmeans_type == 1)
    voc_save_path = "vocabulary_brb_1.bin";
  else if (kmeans_type == 2)
    voc_save_path = "vocabulary_brb_2.bin";

  cout << "=== Vocabulary Creation Configuration ===" << endl;
  cout << "ORB Parameters:" << endl;
  cout << "  nfeatures: " << nfeatures << endl;
  cout << "  scaleFactor: " << scaleFactor << endl;
  cout << "  nlevels: " << nlevels << endl;
  cout << "DBoW2 Parameters:" << endl;
  cout << "  k: " << k << endl;
  cout << "  L: " << L << endl;
  cout << "  weighting: TF_IDF" << endl;
  cout << "  scoring: L1_NORM" << endl;
  cout << "K-Means Type: " << kmeans_type
       << " (0: Default DBoW2, 1: BRB-KMeans, 2: BRB-KMeans Real-space)" << endl;
  cout << "=========================================" << endl;

  // Create ORB detector
  Ptr<ORB> orb = ORB::create(nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K,
                             static_cast<cv::ORB::ScoreType>(scoreType), patchSize, fastThreshold);

  // Load dataset
  BovisaDataLoader dataloader(bovisa_path, num_images, random_seed);
  const vector<string> &image_paths = dataloader.getAllImagePaths();

  if (image_paths.empty()) {
    cerr << "Error: No images loaded from dataset" << endl;
    return -1;
  }

  // Extract features
  vector<vector<Mat>> features;
  extractFeatures(image_paths, features, orb);

  // Filter out empty feature vectors
  vector<vector<Mat>> valid_features;
  for (const auto &feat : features) {
    if (!feat.empty()) {
      valid_features.push_back(feat);
    }
  }

  if (valid_features.empty()) {
    cerr << "Error: No valid features extracted" << endl;
    return -1;
  }

  cout << "Valid features from " << valid_features.size() << " images" << endl;

  // Create vocabulary
  auto start_time = std::chrono::high_resolution_clock::now();
  cout << "\nCreating vocabulary with k=" << k << ", L=" << L << "..." << endl;
  OrbVocabulary voc(k, L, weighting, scoring);

  voc.create(valid_features, kmeans_type);

  cout << "\nVocabulary created successfully!" << endl;
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
  cout << "Time taken: " << duration.count() << " seconds" << endl;
  cout << "Vocabulary information:" << endl;
  cout << voc << endl;

  // Save vocabulary
  cout << "\nSaving vocabulary to: " << voc_save_path << endl;
  voc.save(voc_save_path);
  cout << "Vocabulary saved!" << endl;

  return 0;
}
