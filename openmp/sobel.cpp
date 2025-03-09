#include <iostream>
#include <vector>
#include <cmath>
#include "lodepng.h"
#include <omp.h>
#include <cstdlib>
#include <filesystem>
#include <string>

using namespace std;
namespace fs = std::filesystem;

void convertToGrayscale(const vector<unsigned char>& rgba,
                        vector<unsigned char>& gray,
                        unsigned width, unsigned height) {
    gray.resize(width * height);
    #pragma omp parallel for
    for (unsigned i = 0; i < height; i++) {
        for (unsigned j = 0; j < width; j++) {
            int idx = 4 * (i * width + j);
            gray[i * width + j] = static_cast<unsigned char>(
                0.299 * rgba[idx] + 0.587 * rgba[idx + 1] + 0.114 * rgba[idx + 2]
            );
        }
    }
}

void sobelFilter(const vector<unsigned char>& gray,
                 vector<unsigned char>& result,
                 unsigned width, unsigned height) {
    result.resize(width * height, 0);

    int Gx[3][3] = { {-1, 0, 1},
                     {-2, 0, 2},
                     {-1, 0, 1} };
    int Gy[3][3] = { {-1, -2, -1},
                     { 0,  0,  0},
                     { 1,  2,  1} };

    #pragma omp parallel for schedule(static)
    for (unsigned i = 1; i < height - 1; i++) {
        for (unsigned j = 1; j < width - 1; j++) {
            int sumX = 0, sumY = 0;
            for (int k = -1; k <= 1; k++) {
                for (int l = -1; l <= 1; l++) {
                    int pixel = gray[(i + k) * width + (j + l)];
                    sumX += pixel * Gx[k + 1][l + 1];
                    sumY += pixel * Gy[k + 1][l + 1];
                }
            }
            int mag = static_cast<int>(sqrt(sumX * sumX + sumY * sumY));
            if (mag > 255) mag = 255;
            result[i * width + j] = static_cast<unsigned char>(mag);
        }
    }
}

void convertToRGBA(const vector<unsigned char>& gray,
                   vector<unsigned char>& rgba,
                   unsigned width, unsigned height) {
    rgba.resize(width * height * 4);
    #pragma omp parallel for
    for (unsigned i = 0; i < height; i++) {
        for (unsigned j = 0; j < width; j++) {
            unsigned char val = gray[i * width + j];
            int idx = 4 * (i * width + j);
            rgba[idx]     = val;  // R
            rgba[idx + 1] = val;  // G
            rgba[idx + 2] = val;  // B
            rgba[idx + 3] = 255;  // A
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        cout << "Usage: " << argv[0] << " <input_directory> <output_directory> [num_threads]" << endl;
        return 1;
    }

    fs::path inputDir = argv[1];
    fs::path outputDir = argv[2];

    if (!fs::exists(inputDir) || !fs::is_directory(inputDir)) {
        cout << "Error: Input directory does not exist or is not a directory." << endl;
        return 1;
    }

    if (!fs::exists(outputDir)) {
        fs::create_directories(outputDir);
    }

    if (argc >= 4) {
        int num_threads = atoi(argv[3]);
        if (num_threads > 0) {
            omp_set_num_threads(num_threads);
        }
    }
    int num_threads = omp_get_max_threads();
    cout << "Using " << num_threads << " OpenMP threads." << endl;

    double totalStart = omp_get_wtime();
    double totalProcessingTime = 0.0;
    double totalIOTime = 0.0;
    int fileCount = 0;

    for (const auto& entry : fs::directory_iterator(inputDir)) {
        if (entry.is_regular_file()) {
            fs::path filePath = entry.path();
            if (filePath.extension() == ".png" || filePath.extension() == ".PNG") {
                cout << "Processing file: " << filePath.filename().string() << endl;
                
                // --- Measure I/O time: Reading ---
                double ioReadStart = omp_get_wtime();
                vector<unsigned char> image;
                unsigned width, height;
                unsigned error = lodepng::decode(image, width, height, filePath.string());
                double ioReadEnd = omp_get_wtime();
                if (error) {
                    cout << "Error decoding PNG: " << lodepng_error_text(error) << endl;
                    continue;
                }
                double readTime = ioReadEnd - ioReadStart;
                
                // --- Measure Processing Time (computation only) ---
                double procStart = omp_get_wtime();
                vector<unsigned char> gray;
                convertToGrayscale(image, gray, width, height);
                vector<unsigned char> sobelResult;
                sobelFilter(gray, sobelResult, width, height);
                vector<unsigned char> outputImage;
                convertToRGBA(sobelResult, outputImage, width, height);
                double procEnd = omp_get_wtime();
                double processingTime = procEnd - procStart;
                
                // --- Measure I/O time: Writing ---
                double ioWriteStart = omp_get_wtime();
                string stem = filePath.stem().string();
                fs::path outputFile = outputDir / (stem + "_output" + filePath.extension().string());
                error = lodepng::encode(outputFile.string(), outputImage, width, height);
                double ioWriteEnd = omp_get_wtime();
                if (error) {
                    cout << "Error encoding PNG: " << lodepng_error_text(error) << endl;
                    continue;
                }
                double writeTime = ioWriteEnd - ioWriteStart;
                double fileIOTime = readTime + writeTime;

                cout << "Processed " << filePath.filename().string() 
                     << " in " << processingTime << " seconds (processing only), I/O time: " 
                     << fileIOTime << " seconds. Saved to " 
                     << outputFile.string() << endl;
                fileCount++;
                totalProcessingTime += processingTime;
                totalIOTime += fileIOTime;
            }
        }
    }
    
    double totalEnd = omp_get_wtime();
    double overallTime = totalEnd - totalStart;
    cout << "Processed " << fileCount << " file(s)." << endl;
    cout << "Total processing time (excluding I/O): " << totalProcessingTime << " seconds." << endl;
    cout << "Total I/O time: " << totalIOTime << " seconds." << endl;
    cout << "Overall time (including I/O): " << overallTime << " seconds." << endl;
    return 0;
}
