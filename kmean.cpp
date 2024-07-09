#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <cstdlib>
#include <ctime>

// Function to calculate Euclidean distance between two points
double euclideanDistance(const std::pair<double, double>& point1, const std::pair<double, double>& point2) {
    return std::sqrt(std::pow(point1.first - point2.first, 2) + std::pow(point1.second - point2.second, 2));
}

// Function to initialize centroids randomly
std::vector<std::pair<double, double>> initializeCentroids(const std::vector<std::pair<double, double>>& data, int k) {
    std::vector<std::pair<double, double>> centroids;
    std::srand(std::time(0));
    for (int i = 0; i < k; ++i) {
        centroids.push_back(data[std::rand() % data.size()]);
    }
    return centroids;
}

// Function to assign points to the nearest centroid
std::vector<int> assignPointsToCentroids(const std::vector<std::pair<double, double>>& data, const std::vector<std::pair<double, double>>& centroids) {
    std::vector<int> assignments(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        double minDistance = std::numeric_limits<double>::max();
        for (size_t j = 0; j < centroids.size(); ++j) {
            double distance = euclideanDistance(data[i], centroids[j]);
            if (distance < minDistance) {
                minDistance = distance;
                assignments[i] = j;
            }
        }
    }
    return assignments;
}

// Function to update centroids based on current assignments
std::vector<std::pair<double, double>> updateCentroids(const std::vector<std::pair<double, double>>& data, const std::vector<int>& assignments, int k) {
    std::vector<std::pair<double, double>> newCentroids(k, {0.0, 0.0});
    std::vector<int> counts(k, 0);
    
    for (size_t i = 0; i < data.size(); ++i) {
        newCentroids[assignments[i]].first += data[i].first;
        newCentroids[assignments[i]].second += data[i].second;
        counts[assignments[i]]++;
    }
    
    for (int i = 0; i < k; ++i) {
        newCentroids[i].first /= counts[i];
        newCentroids[i].second /= counts[i];
    }
    
    return newCentroids;
}

// K-means clustering function
std::vector<int> kmeans(const std::vector<std::pair<double, double>>& data, int k, int maxIterations) {
    std::vector<std::pair<double, double>> centroids = initializeCentroids(data, k);
    std::vector<int> assignments;
    
    for (int i = 0; i < maxIterations; ++i) {
        assignments = assignPointsToCentroids(data, centroids);
        std::vector<std::pair<double, double>> newCentroids = updateCentroids(data, assignments, k);
        
        if (newCentroids == centroids) {
            break; // Convergence
        }
        centroids = newCentroids;
    }
    
    return assignments;
}

int main() {
    std::vector<std::pair<double, double>> data = {{1.0, 2.0}, {1.5, 1.8}, {5.0, 8.0}, {8.0, 8.0}, {1.0, 0.6}, {9.0, 11.0}, {8.0, 2.0}, {10.0, 2.0}, {9.0, 3.0}};
    std::cout<<"data size: "<< data.size()<<std::endl;
    int k = 3; // Number of clusters
    int maxIterations = 100;
    
    std::vector<int> assignments = kmeans(data, k, maxIterations);
    
    for (size_t i = 0; i < data.size(); ++i) {
        std::cout << "Point (" << data[i].first << ", " << data[i].second << ") is in cluster " << assignments[i] << std::endl;
    }
    
    return 0;
}
