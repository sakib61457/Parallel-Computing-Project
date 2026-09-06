#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <fstream>
#include <omp.h>

struct Result {
    int size;
    int threads;
    double normal_ms;
    double collapse_ms;
};

// Normal parallelization on outer loop only
double run_normal(int N, int threads) {
    std::vector<std::vector<int>> A(N, std::vector<int>(N, 1));
    std::vector<std::vector<int>> B(N, std::vector<int>(N, 2));
    std::vector<std::vector<int>> C(N, std::vector<int>(N, 0));

    omp_set_num_threads(threads);
    double start = omp_get_wtime();

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < N; ++k) {
            for (int j = 0; j < N; ++j) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    double end = omp_get_wtime();
    return (end - start) * 1000.0;
}

// Collapsed parallelization on both outer loops
double run_collapse(int N, int threads) {
    std::vector<std::vector<int>> A(N, std::vector<int>(N, 1));
    std::vector<std::vector<int>> B(N, std::vector<int>(N, 2));
    std::vector<std::vector<int>> C(N, std::vector<int>(N, 0));

    omp_set_num_threads(threads);
    double start = omp_get_wtime();

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < N; ++k) {
            for (int j = 0; j < N; ++j) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    double end = omp_get_wtime();
    return (end - start) * 1000.0;
}

int main() {
    std::vector<int> matrix_sizes = {256, 512, 1024};
    std::vector<int> thread_counts = {1, 2, 4, 8};

    // Prepare CSV File Output
    std::string csvPath = "data\\benchmark_results\\collapse_results.csv";
    std::ofstream csv(csvPath);

    if (!csv.is_open()) {
        std::cerr << "Error: Could not create CSV output file at: " << csvPath << std::endl;
        std::cerr << "Ensure the directory 'data\\benchmark_results\\' exists.\n";
        return 1;
    }

    // Write CSV Header
    csv << "Matrix,Threads,Normal parallel for (ms),collapse(2) (ms),Faster\n";

    std::cout << "Running Matrix Multiplication Collapse Benchmark...\n\n";

    // Print Terminal Header
    std::cout << std::left 
              << std::setw(10) << "Matrix" 
              << std::setw(10) << "Threads" 
              << std::setw(22) << "Normal parallel for" 
              << std::setw(18) << "collapse(2)" 
              << std::setw(15) << "Faster" << "\n";
    std::cout << std::string(75, '-') << "\n";

    for (int N : matrix_sizes) {
        for (int t : thread_counts) {
            double normal_time = run_normal(N, t);
            double collapse_time = run_collapse(N, t);
            std::string faster = (normal_time < collapse_time) ? "Normal" : "Collapse(2)";

            // Print to Console
            std::cout << std::left 
                      << std::setw(10) << N 
                      << std::setw(10) << t 
                      << std::setw(22) << (std::to_string(normal_time) + " ms") 
                      << std::setw(18) << (std::to_string(collapse_time) + " ms") 
                      << std::setw(15) << faster << "\n";

            // Write Row to CSV File
            csv << N << "," << t << "," << normal_time << "," << collapse_time << "," << faster << "\n";
            csv.flush();
        }
    }

    csv.close();
    std::cout << "\n[SUCCESS] Benchmark results saved to CSV: " << csvPath << "\n";

    return 0;
}