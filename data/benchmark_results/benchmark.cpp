#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <omp.h>

void log_benchmark(std::ofstream& csvFile, const std::string& algorithm, 
                   const std::string& version, const std::string& inputSize, 
                   int threads, double timeMs) {
    // Write a single CSV line: Algorithm, Version, Input Size, Threads, Time(ms)
    csvFile << algorithm << "," 
            << version << "," 
            << inputSize << "," 
            << threads << "," 
            << timeMs << "\n";
}

int main() {
    // Open CSV output file
    std::ofstream csvFile("benchmark_results.csv");
    
    // Write Table Headers
    csvFile << "Algorithm,Version,Input Size,Threads,Time (ms)\n";

    std::vector<int> thread_counts = {1, 2, 4, 8};
    std::string input_label = "10K";

    // 1. Run Sequential Baseline
    double seq_time = 12.5; // Replace with actual std::chrono measurement
    log_benchmark(csvFile, "Merge Sort", "Sequential", input_label, 1, seq_time);

    // 2. Run Parallel Tests Across Threads
    for (int t : thread_counts) {
        omp_set_num_threads(t);
        
        // Measure execution time using std::chrono
        auto start = std::chrono::high_resolution_clock::now();
        
        // execute_parallel_merge_sort();
        
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

        log_benchmark(csvFile, "Merge Sort", "Parallel", input_label, t, elapsed_ms);
    }

    csvFile.close();
    std::cout << "Benchmark results exported to benchmark_results.csv" << std::endl;
    return 0;
}