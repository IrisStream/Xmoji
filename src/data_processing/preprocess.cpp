#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <future>
#include <thread>
#include <mutex>
#include <filesystem>

namespace fs = std::filesystem;

std::mutex outputMutex;

bool hasEnoughWords(const std::string& text, std::size_t minWordCount) {
    std::istringstream textStream(text);
    std::string word;
    std::size_t wordCount = 0;

    while (textStream >> word) {
        ++wordCount;
    }

    return wordCount >= minWordCount;
}

void preprocessFile(const std::string& inputFilePath, const std::string& outputFilePath, std::size_t minWordCount) {
    // Open the input file
    std::ifstream inputFile(inputFilePath);
    if (!inputFile.is_open()) {
        std::cerr << "Error opening input file: " << inputFilePath << std::endl;
        return;
    }

    // Process each line in the input file
    std::string line;
    std::ostringstream outputLines;
    while (std::getline(inputFile, line)) {
        // Split the line into text and emoji using '\t' as the delimiter
        std::istringstream iss(line);
        std::string text, emoji;

        if (std::getline(iss, text, '\t') && std::getline(iss, emoji, '\t')) {
            // Check if the line follows the specified format (TEXT\tEMOJI)
            if (!text.empty() && !emoji.empty() && hasEnoughWords(text, minWordCount)) {
                // Remove leading spaces in text
                text.erase(text.begin(), std::find_if_not(text.begin(), text.end(), [](int c) {
                    return std::isspace(c);
                }));

                // Replace multiple spaces with a single space
                std::istringstream textStream(text);
                text = "";
                std::string word;
                while (textStream >> word) {
                    text += word + " ";
                }
                text.pop_back();  // Remove the trailing space

                // Write the valid line to the output buffer
                outputLines << text << '\t' << emoji << '\n';
            }
        }
    }

    // Close the input file
    inputFile.close();

    // Lock the output buffer with a mutex before writing to the output file
    std::lock_guard<std::mutex> lock(outputMutex);

    // Open the output file in append mode
    std::ofstream outputFile(outputFilePath, std::ios::app);
    if (!outputFile.is_open()) {
        std::cerr << "Error opening output file: " << outputFilePath << std::endl;
        return;
    }

    // Write the contents of the output buffer to the output file
    outputFile << outputLines.str();
    
    // Close the output file
    outputFile.close();

    std::cout << "Preprocessing completed for file: " << inputFilePath << std::endl;
}

int main() {
    // Specify the input and output directories
    std::string inputDirectory = "../../data/output";
    std::string outputDirectory = "../../data/normalized_output";

    // Minimum word count for a line to be considered
    std::size_t minWordCount = 5;

    // Container for futures
    std::vector<std::future<void>> futures;

    // Iterate over files in the input directory
    for (const auto& entry : fs::directory_iterator(inputDirectory)) {
        // Skip directories
        if (entry.is_directory()) {
            continue;
        }

        // Get file paths
        std::string inputFilePath = entry.path().string();
        std::string outputFilePath = fs::path(outputDirectory) / entry.path().filename();

        // Start asynchronous processing for each file
        futures.emplace_back(std::async(std::launch::async, preprocessFile, inputFilePath, outputFilePath, minWordCount));
    }

    // Wait for all asynchronous tasks to finish
    for (auto& future : futures) {
        future.wait();
    }

    return 0;
}
