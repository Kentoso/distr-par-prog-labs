#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include "lodepng.h"
#include <cstdlib>
#include <filesystem>
#include <string>
#include <sstream>
#include <algorithm>

using namespace std;
namespace fs = std::filesystem;

// Convert an RGBA image to grayscale.
void convertToGrayscale(const vector<unsigned char>& rgba,
                        vector<unsigned char>& gray,
                        unsigned width, unsigned height) {
    gray.resize(width * height);
    for (unsigned i = 0; i < height; i++) {
        for (unsigned j = 0; j < width; j++) {
            int idx = 4 * (i * width + j);
            gray[i * width + j] = static_cast<unsigned char>(
                0.299 * rgba[idx] + 0.587 * rgba[idx + 1] + 0.114 * rgba[idx + 2]
            );
        }
    }
}

// Apply the Sobel filter on the grayscale image.
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

// Convert a grayscale image back to an RGBA image.
void convertToRGBA(const vector<unsigned char>& gray,
                   vector<unsigned char>& rgba,
                   unsigned width, unsigned height) {
    rgba.resize(width * height * 4);
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
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (argc < 3) {
        if (rank == 0)
            cout << "Usage: " << argv[0] << " <input_directory> <output_directory>" << endl;
        MPI_Finalize();
        return 1;
    }
    
    fs::path inputDir = argv[1];
    fs::path outputDir = argv[2];
    
    // Rank 0 lists the PNG files in the input directory.
    vector<string> fileList;
    if (rank == 0) {
        if (!fs::exists(inputDir) || !fs::is_directory(inputDir)) {
            cout << "Error: Input directory does not exist or is not a directory." << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        for (const auto& entry : fs::directory_iterator(inputDir)) {
            if (entry.is_regular_file()) {
                fs::path filePath = entry.path();
                if (filePath.extension() == ".png" || filePath.extension() == ".PNG") {
                    fileList.push_back(filePath.string());
                }
            }
        }
        sort(fileList.begin(), fileList.end());

        if (!fs::exists(outputDir)) {
            fs::create_directories(outputDir);
        }
    }

    // Broadcast the number of files.
    int numFiles = fileList.size();
    MPI_Bcast(&numFiles, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Pack the file list into one big string (each filename separated by a newline).
    string fileNamesCombined;
    if (rank == 0) {
        for (int i = 0; i < numFiles; i++) {
            fileNamesCombined += fileList[i] + "\n";
        }
    }
    // Broadcast the size of the combined string.
    int combinedSize = fileNamesCombined.size();
    MPI_Bcast(&combinedSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Ensure all processes have a buffer for the string.
    if (rank != 0)
        fileNamesCombined.resize(combinedSize);
    
    // Broadcast the combined file names string.
    MPI_Bcast(&fileNamesCombined[0], combinedSize, MPI_CHAR, 0, MPI_COMM_WORLD);
    
    // All processes parse the received string into a file list.
    if (rank != 0) {
        fileList.clear();
        istringstream iss(fileNamesCombined);
        string line;
        while(getline(iss, line)) {
            if (!line.empty())
                fileList.push_back(line);
        }
    }
    
    // Each process handles a subset of files in round-robin fashion.
    double localProcessingTime = 0.0;
    double localIOTime = 0.0;
    int localFileCount = 0;
    
    for (int i = rank; i < numFiles; i += size) {
        string filePathStr = fileList[i];
        fs::path filePath(filePathStr);
        cout << "Rank " << rank << " processing file: " 
             << filePath.filename().string() << endl;
        
        // Measure I/O time: reading.
        double ioReadStart = MPI_Wtime();
        vector<unsigned char> image;
        unsigned width, height;
        unsigned error = lodepng::decode(image, width, height, filePath.string());
        double ioReadEnd = MPI_Wtime();
        if (error) {
            cerr << "Rank " << rank << ": Error decoding PNG " 
                 << filePath << ": " << lodepng_error_text(error) << endl;
            continue;
        }
        double readTime = ioReadEnd - ioReadStart;
        
        // Measure processing (computation) time.
        double procStart = MPI_Wtime();
        vector<unsigned char> gray;
        convertToGrayscale(image, gray, width, height);
        vector<unsigned char> sobelResult;
        sobelFilter(gray, sobelResult, width, height);
        vector<unsigned char> outputImage;
        convertToRGBA(sobelResult, outputImage, width, height);
        double procEnd = MPI_Wtime();
        double processingTime = procEnd - procStart;
        
        // Measure I/O time: writing.
        double ioWriteStart = MPI_Wtime();
        fs::path outputFile = outputDir / (fs::path(filePath).stem().string() + "_output" + filePath.extension().string());
        error = lodepng::encode(outputFile.string(), outputImage, width, height);
        double ioWriteEnd = MPI_Wtime();
        if (error) {
            cerr << "Rank " << rank << ": Error encoding PNG " 
                 << outputFile << ": " << lodepng_error_text(error) << endl;
            continue;
        }
        double writeTime = ioWriteEnd - ioWriteStart;
        double fileIOTime = readTime + writeTime;
        
        localProcessingTime += processingTime;
        localIOTime += fileIOTime;
        localFileCount++;
        
        cout << "Rank " << rank << ": Processed " 
             << filePath.filename().string() << " in " << processingTime 
             << " sec (processing) and " << fileIOTime 
             << " sec (I/O). Saved to " << outputFile.string() << endl;
    }
    
    double totalProcessingTime = 0.0;
    double totalIOTime = 0.0;
    int totalFileCount = 0;
    
    MPI_Reduce(&localProcessingTime, &totalProcessingTime, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&localIOTime, &totalIOTime, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&localFileCount, &totalFileCount, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        cout << "\nTotal files processed: " << totalFileCount << endl;
        cout << "Total processing time (computation only): " << totalProcessingTime << " sec." << endl;
        cout << "Total I/O time: " << totalIOTime << " sec." << endl;
    }
    
    MPI_Finalize();
    return 0;
}
