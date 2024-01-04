#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <regex>
#include <vector>
#include <thread>
#include <mutex>
#include <future>
#include <queue>
#include <condition_variable>
#include <filesystem>

namespace fs = std::filesystem;

std::mutex outputMutex;
std::mutex fileMutex;
std::condition_variable cv;

std::queue<std::pair<std::string, std::string>> filesToProcess;

void processFile(const std::string& inputFilePath, const std::string& outputFilePath) {
    // ... (Same as before)
    // Open the input file
    std::ifstream inputFile(inputFilePath);
    if (!inputFile.is_open()) {
        std::cerr << "Error opening input file: " << inputFilePath << std::endl;
        return;
    }

    // Process each line in the input file
    std::ostringstream outputLines;
    std::string line;
    while (std::getline(inputFile, line)) {
        // Perform text processing
        std::transform(line.begin(), line.end(), line.begin(), ::tolower);
        std::regex_replace(line, std::regex("’"), "\x27");
        std::regex_replace(line, std::regex("…"), "...");
        std::regex_replace(line, std::regex("cannot "), "can not ");
        std::regex_replace(line, std::regex("n't "), " n't ");
        std::regex_replace(line, std::regex("n 't "), " n't ");
        std::regex_replace(line, std::regex("ca n't"), "can't");
        std::regex_replace(line, std::regex("ai n't"), "ain't");
        std::regex_replace(line, std::regex("'m "), " 'm ");
        std::regex_replace(line, std::regex("'re "), " 're ");
        std::regex_replace(line, std::regex("'s "), " 's ");
        std::regex_replace(line, std::regex("'ll "), " 'll ");
        std::regex_replace(line, std::regex("'d "), " 'd ");
        std::regex_replace(line, std::regex("'ve "), " 've ");
        std::regex_replace(line, std::regex(" p\\. m\\."), " p.m.");
        std::regex_replace(line, std::regex(" p\\. m "), " p.m ");
        std::regex_replace(line, std::regex(" a\\. m\\."), " a.m.");
        std::regex_replace(line, std::regex(" a\\. m "), " a.m ");
        std::regex_replace(line, std::regex("@[a-zA-Z0-9_]+"), "@USER");
        std::regex_replace(line, std::regex("(http[^[:space:]]+|www[^\\s]+)"), "HTTPURL");
        
        outputLines << line << '\n';
    }

    // Close the input file
    inputFile.close();

    // Lock the output buffer with a mutex before writing to the output file
    std::lock_guard<std::mutex> lock(outputMutex);

    // Open the output file in append mode
    std::ofstream outputFile(outputFilePath);
    if (!outputFile.is_open()) {
        std::cerr << "Error opening output file: " << outputFilePath << std::endl;
        return;
    }

    // Write the contents of the output buffer to the output file
    outputFile << outputLines.str();

    // Close the output file
    outputFile.close();

    std::cout << "Processing completed for file: " << inputFilePath << std::endl;
}


void processFiles() {
    while (true) {
        std::unique_lock<std::mutex> lock(fileMutex);
        cv.wait(lock, [] { return !filesToProcess.empty(); });

        auto filePair = filesToProcess.front();
        filesToProcess.pop();
        lock.unlock();

        if (filePair.first.empty() || filePair.second.empty()) {
            break; // Signal to exit
        }

        processFile(filePair.first, filePair.second);
    }
}

void processFile(const std::string& inputFilePath, const std::string& outputFilePath) {
    // ... (Same as before)
    // Open the input file
    std::ifstream inputFile(inputFilePath);
    if (!inputFile.is_open()) {
        std::cerr << "Error opening input file: " << inputFilePath << std::endl;
        return;
    }

    // Process each line in the input file
    std::ostringstream outputLines;
    std::string line;
    while (std::getline(inputFile, line)) {
        // Perform text processing
        std::transform(line.begin(), line.end(), line.begin(), ::tolower);
        std::regex_replace(line, std::regex("’"), "\x27");
        std::regex_replace(line, std::regex("…"), "...");
        std::regex_replace(line, std::regex("cannot "), "can not ");
        std::regex_replace(line, std::regex("n't "), " n't ");
        std::regex_replace(line, std::regex("n 't "), " n't ");
        std::regex_replace(line, std::regex("ca n't"), "can't");
        std::regex_replace(line, std::regex("ai n't"), "ain't");
        std::regex_replace(line, std::regex("'m "), " 'm ");
        std::regex_replace(line, std::regex("'re "), " 're ");
        std::regex_replace(line, std::regex("'s "), " 's ");
        std::regex_replace(line, std::regex("'ll "), " 'll ");
        std::regex_replace(line, std::regex("'d "), " 'd ");
        std::regex_replace(line, std::regex("'ve "), " 've ");
        std::regex_replace(line, std::regex(" p\\. m\\."), " p.m.");
        std::regex_replace(line, std::regex(" p\\. m "), " p.m ");
        std::regex_replace(line, std::regex(" a\\. m\\."), " a.m.");
        std::regex_replace(line, std::regex(" a\\. m "), " a.m ");
        std::regex_replace(line, std::regex("@[a-zA-Z0-9_]+"), "@USER");
        std::regex_replace(line, std::regex("(http[^[:space:]]+|www[^\\s]+)"), "HTTPURL");
        
        outputLines << line << '\n';
    }

    // Close the input file
    inputFile.close();

    // Lock the output buffer with a mutex before writing to the output file
    std::lock_guard<std::mutex> lock(outputMutex);

    // Open the output file in append mode
    std::ofstream outputFile(outputFilePath);
    if (!outputFile.is_open()) {
        std::cerr << "Error opening output file: " << outputFilePath << std::endl;
        return;
    }

    // Write the contents of the output buffer to the output file
    outputFile << outputLines.str();

    // Close the output file
    outputFile.close();

    std::cout << "Processing completed for file: " << inputFilePath << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " input_directory output_directory" << std::endl;
        return 1;
    }

    std::string inputDirectory = argv[1];
    std::string outputDirectory = argv[2];

    // Create the output directory if it doesn't exist
    if (!fs::exists(outputDirectory)) {
        fs::create_directory(outputDirectory);
    }

    // Determine the number of concurrent threads
    unsigned int numThreads = std::thread::hardware_concurrency();
    numThreads = (numThreads == 0) ? 2 : numThreads; // Default to 2 if unknown

    // Start thread pool
    std::vector<std::thread> threadPool;
    for (unsigned int i = 0; i < numThreads; ++i) {
        threadPool.emplace_back(processFiles);
    }

    // Iterate over files in the input directory
    for (const auto& entry : fs::directory_iterator(inputDirectory)) {
        // Skip directories
        if (entry.is_directory()) {
            continue;
        }

        // Get file paths
        std::string inputFilePath = entry.path().string();
        std::string outputFilePath = fs::path(outputDirectory) / entry.path().filename();

        // Enqueue file for processing
        std::unique_lock<std::mutex> lock(fileMutex);
        filesToProcess.emplace(inputFilePath, outputFilePath);
        lock.unlock();
        cv.notify_one();
    }

    // Signal threads to exit
    for (unsigned int i = 0; i < numThreads; ++i) {
        std::unique_lock<std::mutex> lock(fileMutex);
        filesToProcess.emplace("", ""); // Signal to exit
        lock.unlock();
        cv.notify_one();
    }

    // Wait for all threads to finish
    for (auto& thread : threadPool) {
        thread.join();
    }

    return 0;
}
