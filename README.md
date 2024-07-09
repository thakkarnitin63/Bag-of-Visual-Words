# Bag-of-Visual-Words

## Introduction

This project implements a Bag-of-Visual-Words (BoVW) model for image classification using OpenCV and C++. The BoVW model is a popular technique in computer vision for representing images as feature vectors. This project includes functionalities for extracting features from images, creating a visual dictionary using K-means clustering, generating histograms of visual words, and applying TF-IDF weighting to improve the robustness of the representation.

## Prerequisites

Before you begin, ensure you have met the following requirements:
- A C++ compiler supporting C++11 or later
- CMake for building the project
- OpenCV library installed on your system
- A dataset of images for training (e.g., Caltech101)




## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/your-username/Bag-of-Visual-Words.git
    cd Bag-of-Visual-Words
    ```

2. Create the build directory and navigate to it:
    ```sh
    mkdir build
    cd build
    ```

3. Configure the project with CMake:
    ```sh
    cmake ..
    ```

4. Build the project:
    ```sh
    make
    ```

## Usage

1. Prepare your dataset:
    - Place your training images in the `data/Caltech101BOW/train` directory.

2. Run the program:
    ```sh
    ./BOW
    ```

3. The program will:
    - Extract SIFT features from the training images.
    - Save the extracted features to a file (`train_descriptors.ply`).
    - Perform K-means clustering to create a visual dictionary.
    - Generate histograms of visual words for each image.
    - Apply TF-IDF weighting to the histograms.
    - Save the visual dictionary and TF-IDF histograms to files.
    - Load a query image and find the most similar images from the training set.
    - Visualize the results.

## Code Overview

### `main.cpp`

The main entry point of the program. It initializes the BagofWords class, trains the model, and performs image query and visualization.

### `BagofWords.hpp`

Header file for the BagofWords class, containing method declarations for training, saving, loading descriptors, and performing image queries.

### `BagofWords.cpp`

Implementation file for the BagofWords class. It includes methods for feature extraction, K-means clustering, histogram creation, TF-IDF application, and result visualization.

### Key Functions:

- `train(const std::string& trainPath)`: Trains the BoVW model by extracting descriptors from images or loading them from a file if available.
- `saveDescriptors(const std::string& filename) const`: Saves the extracted descriptors to a file.
- `std::vector<std::vector<cv::Mat>> loadDescriptors(const std::string& filename)`: Loads descriptors from a file.
- `cv::Mat performKMeans(const std::vector<std::vector<cv::Mat>>& descriptorsList) const`: Performs K-means clustering to create the visual dictionary.
- `std::vector<cv::Mat> createHistograms(const std::vector<std::vector<cv::Mat>>& descriptorsList, const cv::Mat& visualDictionary) const`: Creates histograms of visual words for each image.
- `std::vector<cv::Mat> applyTFIDF(const std::vector<cv::Mat>& histograms)`: Applies TF-IDF weighting to the histograms.
- `void saveTFIDFHistogram(const std::string& filename) const`: Saves the TF-IDF histograms to a file.
- `cv::Mat processQueryImage(const cv::Mat& queryImage, const cv::Mat& visualDictionary, const cv::Mat& idf) const`: Processes a query image to get its TF-IDF histogram.
- `std::vector<std::pair<double, int>> findMostSimilarImages(const cv::Mat& queryHistogram, int N) const`: Finds the most similar images to a query image.
- `void visualizeResults(const cv::Mat& queryImage, const std::vector<std::pair<double, int>>& mostSimilarImages, const std::vector<std::string>& sortedFilePaths) const`: Visualizes the query results.




