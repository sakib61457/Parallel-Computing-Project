#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <iomanip>
#include <fstream>
#include <omp.h>

// Initialize zero matrix for output storage
void reset_matrix(std::vector<std::vector<int>>& C, int N) {
    for (int i = 0; i < N; ++i) {
        std::fill(C[i].begin(), C[i].end(), 0);
    }
}

// 1. Sequential Implementation
double run_sequential(const std::vector<std::vector<int>>& A, 
                       const std::vector<std::vector<int>>& B, 
                       std::vector<std::vector<int>>& C, int N) {
    reset_matrix(C, N);
    double start = omp_get_wtime();

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

// 2. Parallel Static Implementation
double run_static(const std::vector<std::vector<int>>& A, 
                   const std::vector<std::vector<int>>& B, 
                   std::vector<std::vector<int>>& C, int N, int threads) {
    reset_matrix(C, N);
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

// 3. Parallel Dynamic Implementation
double run_dynamic(const std::vector<std::vector<int>>& A, 
                    const std::vector<std::vector<int>>& B, 
                    std::vector<std::vector<int>>& C, int N, int threads) {
    reset_matrix(C, N);
    omp_set_num_threads(threads);
    double start = omp_get_wtime();

    #pragma omp parallel for schedule(dynamic, 16)
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

// 4. Parallel Collapse(2) Implementation
double run_collapse(const std::vector<std::vector<int>>& A, 
                     const std::vector<std::vector<int>>& B, 
                     std::vector<std::vector<int>>& C, int N, int threads) {
    reset_matrix(C, N);
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
    std::vector<int> matrix_sizes = {256, 512};
    std::vector<int> thread_counts = {1, 2, 3, 4};

    // Prepare CSV File Output
    std::string csvPath = "data\\benchmark_results\\controlled_matrix_results.csv";
    std::ofstream csv(csvPath);

    if (!csv.is_open()) {
        std::cerr << "Error: Could not create CSV output file at: " << csvPath << std::endl;
        std::cerr << "Ensure the target path exists.\n";
        return 1;
    }

    // Write CSV Header
    csv << "Matrix,Threads,Sequential (ms),Static (ms),Dynamic (ms),Collapse(2) (ms)\n";

    std::cout << "========================================================================\n";
    std::cout << " CONTROLLED MATRIX MULTIPLICATION BENCHMARK (Identical Input Data)\n";
    std::cout << "========================================================================\n\n";

    for (int N : matrix_sizes) {
        std::cout << ">>> Generating Controlled Master Input Matrices A and B (" << N << "x" << N << ")... ";
        
        // Master input data generation with fixed random seed
        std::mt19937 rng(42);
        std::vector<std::vector<int>> A(N, std::vector<int>(N));
        std::vector<std::vector<int>> B(N, std::vector<int>(N));
        std::vector<std::vector<int>> C(N, std::vector<int>(N, 0));

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                A[i][j] = rng() % 100;
                B[i][j] = rng() % 100;
            }
        }
        std::cout << "Done.\n\n";

        // 1. Run Sequential once per Matrix Size
        double seq_time = run_sequential(A, B, C, N);

        // Print Terminal Header for current size
        std::cout << std::left 
                  << std::setw(10) << "Matrix" 
                  << std::setw(10) << "Threads" 
                  << std::setw(16) << "Sequential" 
                  << std::setw(16) << "Static" 
                  << std::setw(16) << "Dynamic" 
                  << std::setw(16) << "Collapse(2)" << "\n";
        std::cout << std::string(84, '-') << "\n";

        // 2. Run Parallel Variants across Thread Counts using exact same A and B
        for (int t : thread_counts) {
            double static_time   = run_static(A, B, C, N, t);
            double dynamic_time  = run_dynamic(A, B, C, N, t);
            double collapse_time = run_collapse(A, B, C, N, t);

            // Output to Terminal
            std::cout << std::left 
                      << std::setw(10) << N 
                      << std::setw(10) << t 
                      << std::setw(16) << (std::to_string(seq_time) + " ms") 
                      << std::setw(16) << (std::to_string(static_time) + " ms") 
                      << std::setw(16) << (std::to_string(dynamic_time) + " ms") 
                      << std::setw(16) << (std::to_string(collapse_time) + " ms") << "\n";

            // Write Row to CSV
            csv << N << "," << t << "," << seq_time << "," << static_time << "," 
                << dynamic_time << "," << collapse_time << "\n";
            csv.flush();
        }
        std::cout << "\n";
    }

    csv.close();
    std::cout << "[SUCCESS] Controlled benchmark results logged to: " << csvPath << "\n";

    return 0;
}