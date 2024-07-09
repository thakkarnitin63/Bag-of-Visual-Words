#include "BagofWords.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    std::string trainPath = "/home/nitin/NitinWs/data/Caltech101BOW/train";
    std::string queryImagePath = "/home/nitin/NitinWs/BOW/plane.jpg";
    int dictionarySize = 2000;

    // Create an instance of BagOfWords
    BagofWords bow(dictionarySize);

    // Train the model
    bow.train(trainPath); //checked

    // Perform K-means clustering to create the visual dictionary
    cv::Mat visualDictionary = bow.performKMeans(bow.getDescriptors());

    // Save the visual dictionary to a file
    bow.saveVisualDictionary("visual_dictionary.yml");

    std::cout << "Visual dictionary created and saved." << std::endl;

    // Create histograms of visual words for each image
    std::vector<cv::Mat> histograms = bow.createHistograms(bow.getDescriptors(), visualDictionary);

    // Apply TF-IDF to histograms
    std::vector<cv::Mat> tfidfHistograms = bow.applyTFIDF(histograms);

    // Save TF-IDF histograms to file
    bow.saveTFIDFHistogram("tfidf_histograms.yml");

    std::cout << "TF-IDF histograms created and saved." << std::endl;

    // Load a query image
    cv::Mat queryImage = cv::imread(queryImagePath, cv::IMREAD_GRAYSCALE);
    if (queryImage.empty()) {
        std::cerr << "Failed to load query image." << std::endl;
        return -1;
    }

    // Process the query image to get the TF-IDF histogram
    cv::Mat queryHistogram = bow.processQueryImage(queryImage, visualDictionary, bow.getIDF());

    // Find the most similar images
    int N = 10;
    std::vector<std::pair<double, int>> mostSimilarImages = bow.findMostSimilarImages(queryHistogram, N);

    // // Visualize the results
    bow.visualizeResults(queryImage, mostSimilarImages, bow.getSortedFilePaths());

    return 0;
}
