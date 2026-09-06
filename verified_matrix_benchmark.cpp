#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <omp.h>

// Reset output matrix C to zeros without reallocation
void reset_matrix(std::vector<std::vector<double>>& C, int N) {
    for (int i = 0; i < N; ++i) {
        std::fill(C[i].begin(), C[i].end(), 0.0);
    }
}

// Correctness Verification with 1e-9 Floating-Point Tolerance
bool matricesEqual(const std::vector<std::vector<double>>& GroundTruth,
                   const std::vector<std::vector<double>>& TestMatrix,
                   int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (std::fabs(GroundTruth[i][j] - TestMatrix[i][j]) > 1e-9) {
                return false;
            }
        }
    }
    return true;
}

// 1. Sequential (Ground Truth Reference)
double run_sequential(const std::vector<std::vector<double>>& A, 
                       const std::vector<std::vector<double>>& B, 
                       std::vector<std::vector<double>>& C, int N) {
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

// 2. Parallel Static
double run_static(const std::vector<std::vector<double>>& A, 
                   const std::vector<std::vector<double>>& B, 
                   std::vector<std::vector<double>>& C, int N, int threads) {
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

// 3. Parallel Dynamic
double run_dynamic(const std::vector<std::vector<double>>& A, 
                    const std::vector<std::vector<double>>& B, 
                    std::vector<std::vector<double>>& C, int N, int threads) {
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

// 4. Parallel Collapse(2)
double run_collapse(const std::vector<std::vector<double>>& A, 
                     const std::vector<std::vector<double>>& B, 
                     std::vector<std::vector<double>>& C, int N, int threads) {
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

    // CSV Setup
    std::string csvPath = "data\\benchmark_results\\verified_matrix_results.csv";
    std::ofstream csv(csvPath);

    if (!csv.is_open()) {
        std::cerr << "Error: Could not open output file " << csvPath << std::endl;
        return 1;
    }

    csv << "Matrix,Threads,Sequential (ms),Static (ms),Dynamic (ms),Collapse(2) (ms),Verified\n";

    std::cout << "========================================================================\n";
    std::cout << " STRICT VERIFICATION & BENCHMARKING (Sequential == Parallel Check)\n";
    std::cout << "========================================================================\n\n";

    for (int N : matrix_sizes) {
        std::cout << ">>> Initializing Master Input Matrices A and B (" << N << "x" << N << ")... ";
        
        std::mt19937 rng(42);
        std::uniform_real_distribution<double> dist(0.0, 100.0);

        std::vector<std::vector<double>> A(N, std::vector<double>(N));
        std::vector<std::vector<double>> B(N, std::vector<double>(N));
        
        // Output buffers
        std::vector<std::vector<double>> C_seq(N, std::vector<double>(N, 0.0));
        std::vector<std::vector<double>> C_test(N, std::vector<double>(N, 0.0));

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                A[i][j] = dist(rng);
                B[i][j] = dist(rng);
            }
        }
        std::cout << "Done.\n";

        // Step 1: Run Sequential Ground Truth
        std::cout << ">>> Running Sequential Baseline... ";
        double seq_time = run_sequential(A, B, C_seq, N);
        std::cout << seq_time << " ms\n\n";

        // Print Header
        std::cout << std::left 
                  << std::setw(8)  << "Matrix" 
                  << std::setw(8)  << "Threads" 
                  << std::setw(16) << "Sequential" 
                  << std::setw(16) << "Static" 
                  << std::setw(16) << "Dynamic" 
                  << std::setw(16) << "Collapse(2)" 
                  << std::setw(10) << "Verified" << "\n";
        std::cout << std::string(90, '-') << "\n";

        for (int t : thread_counts) {
            // Step 2: Run and Verify Static
            double static_time = run_static(A, B, C_test, N, t);
            if (!matricesEqual(C_seq, C_test, N)) {
                std::cerr << "\n[CORRUPTION DETECTED] Static parallelization output mismatch at N=" << N << ", threads=" << t << "!\n";
                return 1;
            }

            // Step 3: Run and Verify Dynamic
            double dynamic_time = run_dynamic(A, B, C_test, N, t);
            if (!matricesEqual(C_seq, C_test, N)) {
                std::cerr << "\n[CORRUPTION DETECTED] Dynamic parallelization output mismatch at N=" << N << ", threads=" << t << "!\n";
                return 1;
            }

            // Step 4: Run and Verify Collapse(2)
            double collapse_time = run_collapse(A, B, C_test, N, t);
            if (!matricesEqual(C_seq, C_test, N)) {
                std::cerr << "\n[CORRUPTION DETECTED] Collapse(2) output mismatch at N=" << N << ", threads=" << t << "!\n";
                return 1;
            }

            // Step 5: Log Clean Results
            std::cout << std::left 
                      << std::setw(8)  << N 
                      << std::setw(8)  << t 
                      << std::setw(16) << (std::to_string(seq_time) + " ms") 
                      << std::setw(16) << (std::to_string(static_time) + " ms") 
                      << std::setw(16) << (std::to_string(dynamic_time) + " ms") 
                      << std::setw(16) << (std::to_string(collapse_time) + " ms") 
                      << std::setw(10) << "PASSED" << "\n";

            csv << N << "," << t << "," << seq_time << "," << static_time << "," 
                << dynamic_time << "," << collapse_time << ",PASSED\n";
            csv.flush();
        }
        std::cout << "\n";
    }

    csv.close();
    std::cout << "[SUCCESS] All parallel variants verified match Sequential within 1e-9 tolerance.\n";
    std::cout << "Results saved to: " << csvPath << "\n";

    return 0;
}