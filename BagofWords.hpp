#pragma once

#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <filesystem>
#include <regex>

namespace fs = std::filesystem;

class BagofWords {
public:
    BagofWords(int dictionarySize);

    // Methods for Training and Saving
    void train(const std::string& trainPath); // checked
    void saveDescriptors(const std::string& filename) const;

    std::vector<std::vector<cv::Mat>> loadDescriptors(const std::string& filename);
    cv::Mat performKMeans(const std::vector<std::vector<cv::Mat>>& descriptorsList) const;

    void saveVisualDictionary(const std::string& filename) const;
    std::vector<cv::Mat> createHistograms(const std::vector<std::vector<cv::Mat>>& descriptorsList, const cv::Mat& visualDictionary) const;
    std::vector<cv::Mat> applyTFIDF(const std::vector<cv::Mat>& histograms) ;

    void saveTFIDFHistogram(const std::string& filename) const;

    // Methods for Query
    cv::Mat processQueryImage(const cv::Mat& queryImage, const cv::Mat& visualDictionary, const cv::Mat& idf) const;
    std::vector<std::pair<double, int>> findMostSimilarImages(const cv::Mat& queryHistogram, int N) const;
    void visualizeResults(const cv::Mat& queryImage, const std::vector<std::pair<double, int>>& mostSimilarImages, const std::vector<std::string>& sortedFilePaths) const;

    // // Getters
    std::vector<std::vector<cv::Mat>> getDescriptors() const { return descriptorsList; }
    cv::Mat getIDF() const { return idf; }
    const std::vector<std::string>& getSortedFilePaths() const { return sortedFilePaths; }


private:
    int extractNumericPart(const std::string& filename) const;
    std::vector<std::string> collectAndSortFilePaths(const std::string& directory) const;
    std::vector<std::vector<cv::Mat>> extractDescriptors(const std::string& directory);
    double cosineDistance(const cv::Mat& a, const cv::Mat& b) const;

    int dictionarySize;
    std::vector<std::vector<cv::Mat>> descriptorsList;
    std::vector<std::string> sortedFilePaths;
    cv::Mat visualDictionary;
    std::vector<cv::Mat> tfidfHistograms;
    cv::Mat idf;
};
