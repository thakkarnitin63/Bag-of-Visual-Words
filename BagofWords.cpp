#include "BagofWords.hpp"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <opencv2/flann.hpp>

BagofWords::BagofWords(int dictionarySize) : dictionarySize(dictionarySize) {}

void BagofWords::train(const std::string& trainPath) {
    
    std::string descriptorFile = "/home/nitin/NitinWs/BOW/train_descriptors.ply";

    // check if the descriptor file already exists
    if(fs::exists(descriptorFile)) {
        // Load the descriptors from file
        descriptorsList = loadDescriptors(descriptorFile);
        std::cout << "Descriptors list size :"<<descriptorsList.size() << "Descriptors Loaded from file" << std::endl;
    } else {
        descriptorsList = extractDescriptors(trainPath);
        // Save the descriptors to file
        saveDescriptors(descriptorFile);
        std::cout << "Descriptors extracted and Saved to File. "<<  std::endl;
    }
    sortedFilePaths = collectAndSortFilePaths(trainPath);
}


void BagofWords::saveDescriptors(const std::string& filename) const {
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return;
    }
    size_t totalDescriptors = 0;
    for (const auto& imageDescriptors : descriptorsList) {
        for (const auto& descriptors : imageDescriptors) {
            totalDescriptors += descriptors.rows;
        }
    }
    ofs << "ply\n";
    ofs << "format binary_little_endian 1.0\n";
    ofs << "element descriptor " << totalDescriptors << "\n";
    ofs << "property list uchar float descriptor\n";
    ofs << "end_header\n";
    for (const auto& imageDescriptors : descriptorsList) {
        for (const auto& descriptors : imageDescriptors) {
            int cols = descriptors.cols;
            uchar descSize = static_cast<uchar>(cols);
            for (int i = 0; i < descriptors.rows; ++i) {
                ofs.write(reinterpret_cast<const char*>(&descSize), sizeof(descSize));
                ofs.write(reinterpret_cast<const char*>(descriptors.ptr<float>(i)), cols * sizeof(float));
            }
        }
        uchar separator = 0;
        ofs.write(reinterpret_cast<const char*>(&separator), sizeof(separator));
    }
    std::cout<<"Model Saved"<< std::endl;
    ofs.close();
}


std::vector<std::vector<cv::Mat>> BagofWords::loadDescriptors(const std::string& filename) {
    std::vector<std::vector<cv::Mat>> loadedDescriptors;
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs) {
        std::cerr << "Failed to open file for reading: " << filename << std::endl;
        return loadedDescriptors;
    }
    std::string line;
    size_t totalDescriptors = 0;
    while (std::getline(ifs, line)) {
        if (line == "end_header") {
            break;
        }
        if (line.find("element descriptor") != std::string::npos) {
            totalDescriptors = std::stoul(line.substr(line.find_last_of(' ') + 1));
        }
    }
    std::vector<cv::Mat> currentImageDescriptors;
    while (ifs.peek() != EOF) {
        uchar descSize;
        ifs.read(reinterpret_cast<char*>(&descSize), sizeof(descSize));
        if (descSize == 0) {
            if (!currentImageDescriptors.empty()) {
                loadedDescriptors.push_back(currentImageDescriptors);
                currentImageDescriptors.clear();
            }
            continue;
        }
        cv::Mat descriptor(1, descSize, CV_32F);
        ifs.read(reinterpret_cast<char*>(descriptor.ptr<float>(0)), descSize * sizeof(float));
        currentImageDescriptors.push_back(descriptor);
    }
    if (!currentImageDescriptors.empty()) {
        loadedDescriptors.push_back(currentImageDescriptors);
    }
    ifs.close();
    std::cout << "Image descriptors loaded: " << loadedDescriptors.size() << std::endl;
    return loadedDescriptors;
}



std::vector<std::vector<cv::Mat>> BagofWords::extractDescriptors(const std::string& directory) {
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
    std::vector<std::string> filepaths = collectAndSortFilePaths(directory);
    for (const auto& filepath : filepaths) {
        // std::cout << "Accessing image: " << filepath << std::endl;
        cv::Mat image = cv::imread(filepath, cv::IMREAD_GRAYSCALE);
        if (!image.empty()) {
            std::vector<cv::KeyPoint> keypoints;
            cv::Mat descriptors;
            sift->detectAndCompute(image, cv::noArray(), keypoints, descriptors);
            if (!descriptors.empty()) {
                descriptorsList.push_back({ descriptors });
            }
        } else {
            std::cerr << "Failed to load image: " << filepath << std::endl;
        }
    }
    std::cout <<"Descriptors Check here : "<< descriptorsList.size()<<std::endl;
    return descriptorsList;
}

cv::Mat BagofWords::performKMeans(const std::vector<std::vector<cv::Mat>>& descriptorsList) const {
    cv::Mat allDescriptors;
    for (const auto& descriptors : descriptorsList) {
        for (const auto& descriptor : descriptors) {
            allDescriptors.push_back(descriptor);
        }
    }
    cv::Mat1f allDescriptorsFloat;
    allDescriptors.convertTo(allDescriptorsFloat, CV_32F);
    cvflann::KMeansIndexParams params;
    cv::Mat1f centers(dictionarySize, allDescriptors.cols);
    int iterations = cv::flann::hierarchicalClustering<cvflann::L2<float>>(allDescriptorsFloat, centers, params);
    return centers;
}

void BagofWords::saveVisualDictionary(const std::string& filename) const {
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);
    fs << "vocabulary" << visualDictionary;
    fs.release();
}

std::vector<cv::Mat> BagofWords::createHistograms(const std::vector<std::vector<cv::Mat>>& descriptorsList, const cv::Mat& visualDictionary) const {
    std::vector<cv::Mat> histograms;
    cv::BFMatcher matcher(cv::NORM_L2);
    for (const auto& descriptors : descriptorsList) {
        cv::Mat histogram = cv::Mat::zeros(1, visualDictionary.rows, CV_32F);
        for (const auto& descriptor : descriptors) {
            std::vector<cv::DMatch> matches;
            matcher.match(descriptor, visualDictionary, matches);
            for (const auto& match : matches) {
                histogram.at<float>(0, match.trainIdx)++;
            }
        }
        histograms.push_back(histogram);
    }
    std::cout<<"Histograms size: "<<histograms.size()<<std::endl;
    return histograms;
}

std::vector<cv::Mat> BagofWords::applyTFIDF(const std::vector<cv::Mat>& histograms) {
    int numImages = histograms.size();
    int dictionarySize = histograms[0].cols;

    // std::cout << "Number of images: " << numImages << std::endl;
    // std::cout << "Dictionary size: " << dictionarySize << std::endl;

    // Calculate document frequency
    cv::Mat df = cv::Mat::zeros(1, dictionarySize, CV_32F);
    for (size_t imgIdx = 0; imgIdx < histograms.size(); ++imgIdx) {
        const auto& histogram = histograms[imgIdx];
        for (int i = 0; i < dictionarySize; ++i) {
            if (histogram.at<float>(0, i) > 0) {
                df.at<float>(0, i)++;
            }
        }
    }

    // Calculate inverse document frequency
    cv::Mat idf = cv::Mat::zeros(1, dictionarySize, CV_32F);
    for (int i = 0; i < dictionarySize; ++i) {
        idf.at<float>(0, i) = log(static_cast<float>(numImages) / (1 + df.at<float>(0, i)));
    }
    this->idf = idf;

    // Apply TF-IDF to each histogram
    std::vector<cv::Mat> tfidfHistograms;
    for (size_t imgIdx = 0; imgIdx < histograms.size(); ++imgIdx) {
        const auto& histogram = histograms[imgIdx];
        cv::Mat tfidfHistogram = cv::Mat::zeros(1, dictionarySize, CV_32F);
        float nd = cv::sum(histogram)[0];
        for (int i = 0; i < dictionarySize; ++i) {
            float nid = histogram.at<float>(0, i);
            tfidfHistogram.at<float>(0, i) = (nid / nd) * idf.at<float>(0, i);
        }
        tfidfHistograms.push_back(tfidfHistogram);
    }

    this->tfidfHistograms = tfidfHistograms;

    return tfidfHistograms;
}


void BagofWords::saveTFIDFHistogram(const std::string& filename) const {
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);
    fs << "tfidf_histograms" << tfidfHistograms;
    fs.release();
}

cv::Mat BagofWords::processQueryImage(const cv::Mat& queryImage, const cv::Mat& visualDictionary, const cv::Mat& idf) const {
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    sift->detectAndCompute(queryImage, cv::noArray(), keypoints, descriptors);
    cv::Mat histogram = cv::Mat::zeros(1, visualDictionary.rows, CV_32F);
    cv::BFMatcher matcher(cv::NORM_L2);
    std::vector<cv::DMatch> matches;
    matcher.match(descriptors, visualDictionary, matches);
    for (const auto& match : matches) {
        histogram.at<float>(0, match.trainIdx)++;
    }
    float nd = cv::sum(histogram)[0];
    cv::Mat tfidfHistogram = cv::Mat::zeros(1, visualDictionary.rows, CV_32F);
    for (int i = 0; i < visualDictionary.rows; ++i) {
        float nid = histogram.at<float>(0, i);
        tfidfHistogram.at<float>(0, i) = (nid / nd) * idf.at<float>(0, i);
    }
    return tfidfHistogram;
}

std::vector<std::pair<double, int>> BagofWords::findMostSimilarImages(const cv::Mat& queryHistogram, int N) const {
    std::vector<std::pair<double, int>> distances;

    // Calculate distances
    for (size_t i = 0; i < tfidfHistograms.size(); ++i) {
        double distance = cosineDistance(queryHistogram, tfidfHistograms[i]);
        distances.push_back(std::make_pair(distance, i));
    }

    // Sort distances by ascending order
    std::sort(distances.begin(), distances.end());

    // Filter out invalid indices
    std::vector<std::pair<double, int>> validDistances;
    for (const auto& pair : distances) {
        if (pair.second < tfidfHistograms.size()) {
            validDistances.push_back(pair);
        }
        if (validDistances.size() >= N) {
            break;
        }
    }

    // Return top N valid results
    return validDistances;
}

void BagofWords::visualizeResults(const cv::Mat& queryImage, const std::vector<std::pair<double, int>>& mostSimilarImages, const std::vector<std::string>& sortedFilePaths) const {
    cv::namedWindow("Query Image", cv::WINDOW_AUTOSIZE);
    cv::imshow("Query Image", queryImage);

    for (size_t i = 0; i < mostSimilarImages.size(); ++i) {
        int index = mostSimilarImages[i].second;
        if (index < 0 || index >= sortedFilePaths.size()) {
            std::cerr << "Invalid index: " << index << std::endl;
            continue;
        }

        std::string imagePath = sortedFilePaths[index];
        std::cout << "Loading image: " << imagePath << std::endl;
        cv::Mat similarImage = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);

        if (!similarImage.empty()) {
            cv::namedWindow("Similar Image " + std::to_string(i + 1), cv::WINDOW_AUTOSIZE);
            cv::imshow("Similar Image " + std::to_string(i + 1), similarImage);
        } else {
            std::cerr << "Failed to load image:: " << imagePath << std::endl;
        }
    }

    cv::waitKey(0);
}

double BagofWords::cosineDistance(const cv::Mat& a, const cv::Mat& b) const {
    return 1.0 - (a.dot(b) / (cv::norm(a) * cv::norm(b)));
}

int BagofWords::extractNumericPart(const std::string& filename) const {
    std::regex regex("image_(\\d+)\\.jpg");
    std::smatch match;
    if (std::regex_search(filename, match, regex)) {
        try {
            return std::stoi(match[1].str());
        } catch (const std::invalid_argument&) {
            return 0;
        } catch (const std::out_of_range&) {
            return 0;
        }
    }
    return 0;
}

std::vector<std::string> BagofWords::collectAndSortFilePaths(const std::string& directory) const {
    std::vector<std::string> filepaths;
    std::vector<std::string> validExtensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}; // Add more extensions if needed
    
    for (const auto& entry : fs::directory_iterator(directory)) {
        std::string filePath = entry.path().string();
        std::string extension = entry.path().extension().string();
        
        // Check if the file has a valid image extension
        if (std::find(validExtensions.begin(), validExtensions.end(), extension) != validExtensions.end()) {
            filepaths.push_back(filePath);
        }
    }
    
    std::sort(filepaths.begin(), filepaths.end(), [this](const std::string& a, const std::string& b) {
        return extractNumericPart(a) < extractNumericPart(b);
    });
    
    return filepaths;
}
