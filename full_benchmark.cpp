#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <omp.h>

void reset_matrix(std::vector<std::vector<double>>& C, int N) {
    for (int i = 0; i < N; ++i) {
        std::fill(C[i].begin(), C[i].end(), 0.0);
    }
}

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
    // 3 Matrix Sizes x 4 Thread Counts = 12 core test points per parallel variant
    std::vector<int> matrix_sizes = {256, 512, 1024};
    std::vector<int> thread_counts = {1, 2, 4, 8};

    std::string csvPath = "data\\benchmark_results\\week3_full_benchmark.csv";
    std::ofstream csv(csvPath);

    if (!csv.is_open()) {
        std::cerr << "Error: Cannot open output CSV at " << csvPath << std::endl;
        return 1;
    }

    // CSV Header with speedup metrics relative to Sequential baseline
    csv << "Matrix,Threads,Sequential_ms,Static_ms,Dynamic_ms,Collapse2_ms,Static_Speedup,Dynamic_Speedup,Collapse2_Speedup,Status\n";

    std::cout << "========================================================================================\n";
    std::cout << " WEEK 3 MAJOR BENCHMARK EXPERIMENT (3 Sizes x 4 Threads = 12 Runs per Variant)\n";
    std::cout << "========================================================================================\n\n";

    for (int N : matrix_sizes) {
        std::cout << ">>> Generating Matrix Data N=" << N << " (" << N << "x" << N << ")... " << std::flush;
        
        std::mt19937 rng(42);
        std::uniform_real_distribution<double> dist(0.0, 100.0);

        std::vector<std::vector<double>> A(N, std::vector<double>(N));
        std::vector<std::vector<double>> B(N, std::vector<double>(N));
        std::vector<std::vector<double>> C_seq(N, std::vector<double>(N, 0.0));
        std::vector<std::vector<double>> C_test(N, std::vector<double>(N, 0.0));

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                A[i][j] = dist(rng);
                B[i][j] = dist(rng);
            }
        }
        std::cout << "Done.\n";

        // Step 1: Run Ground Truth Sequential Baseline
        std::cout << ">>> Executing Ground Truth Sequential Baseline... " << std::flush;
        double seq_time = run_sequential(A, B, C_seq, N);
        std::cout << std::fixed << std::setprecision(2) << seq_time << " ms\n\n";

        // Print Console Table Header
        std::cout << std::left 
                  << std::setw(8)  << "Matrix" 
                  << std::setw(9)  << "Threads" 
                  << std::setw(15) << "Sequential" 
                  << std::setw(14) << "Static" 
                  << std::setw(14) << "Dynamic" 
                  << std::setw(15) << "Collapse(2)" 
                  << std::setw(10) << "Verified" << "\n";
        std::cout << std::string(85, '-') << "\n";

        // Step 2: Parallel Sweep across 1, 2, 4, 8 threads
        for (int t : thread_counts) {
            double static_time = run_static(A, B, C_test, N, t);
            if (!matricesEqual(C_seq, C_test, N)) {
                std::cerr << "\n[CORRUPTION ERROR] Static failed correctness test at N=" << N << ", t=" << t << "!\n";
                return 1;
            }

            double dynamic_time = run_dynamic(A, B, C_test, N, t);
            if (!matricesEqual(C_seq, C_test, N)) {
                std::cerr << "\n[CORRUPTION ERROR] Dynamic failed correctness test at N=" << N << ", t=" << t << "!\n";
                return 1;
            }

            double collapse_time = run_collapse(A, B, C_test, N, t);
            if (!matricesEqual(C_seq, C_test, N)) {
                std::cerr << "\n[CORRUPTION ERROR] Collapse(2) failed correctness test at N=" << N << ", t=" << t << "!\n";
                return 1;
            }

            // Print Console Metrics
            std::cout << std::left 
                      << std::setw(8)  << N 
                      << std::setw(9)  << t 
                      << std::setw(15) << (std::to_string((int)seq_time) + " ms") 
                      << std::setw(14) << (std::to_string((int)static_time) + " ms") 
                      << std::setw(14) << (std::to_string((int)dynamic_time) + " ms") 
                      << std::setw(15) << (std::to_string((int)collapse_time) + " ms") 
                      << std::setw(10) << "PASSED" << "\n";

            // Record to CSV File
            double static_sp = seq_time / static_time;
            double dynamic_sp = seq_time / dynamic_time;
            double collapse_sp = seq_time / collapse_time;

            csv << N << "," << t << "," << seq_time << "," << static_time << "," 
                << dynamic_time << "," << collapse_time << "," 
                << static_sp << "," << dynamic_sp << "," << collapse_sp << ",PASSED\n";
            csv.flush();
        }
        std::cout << "\n";
    }

    csv.close();
    std::cout << "========================================================================================\n";
    std::cout << "[SUCCESS] Week 3 full benchmark complete. All 12 test points passed verification.\n";
    std::cout << "Dataset written to: " << csvPath << "\n";
    std::cout << "========================================================================================\n";

    return 0;
}