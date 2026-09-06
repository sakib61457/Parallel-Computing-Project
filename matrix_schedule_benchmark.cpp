#include <iostream>
#include <vector>
#include <string>
#include <omp.h>

void run_matrix_mult(int N, int threads, const std::string& schedule_type) {
    std::vector<std::vector<int>> A(N, std::vector<int>(N, 1));
    std::vector<std::vector<int>> B(N, std::vector<int>(N, 2));
    std::vector<std::vector<int>> C(N, std::vector<int>(N, 0));

    omp_set_num_threads(threads);
    double start = omp_get_wtime();

    if (schedule_type == "static") {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; ++i) {
            for (int k = 0; k < N; ++k) {
                for (int j = 0; j < N; ++j) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
    } 
    else if (schedule_type == "dynamic") {
        #pragma omp parallel for schedule(dynamic, 16)
        for (int i = 0; i < N; ++i) {
            for (int k = 0; k < N; ++k) {
                for (int j = 0; j < N; ++j) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
    } 
    else if (schedule_type == "guided") {
        #pragma omp parallel for schedule(guided, 16)
        for (int i = 0; i < N; ++i) {
            for (int k = 0; k < N; ++k) {
                for (int j = 0; j < N; ++j) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
    }

    double end = omp_get_wtime();
    double time_ms = (end - start) * 1000.0;

    std::cout << "Threads: " << threads 
              << " | Schedule: " << schedule_type 
              << " | Time: " << time_ms << " ms\n";
}

int main() {
    int N = 512;
    std::vector<int> threads_to_test = {1, 2, 3, 4};
    std::vector<std::string> schedules = {"static", "dynamic", "guided"};

    std::cout << "=== Matrix Multiplication Schedule Benchmark (" << N << "x" << N << ") ===\n\n";

    for (int t : threads_to_test) {
        std::cout << "--- Thread Count: " << t << " ---\n";
        for (const auto& sched : schedules) {
            run_matrix_mult(N, t, sched);
        }
        std::cout << "\n";
    }

    return 0;
}

//g++ -fopenmp matrix_schedule_benchmark.cpp -o matrix_schedule_benchmark.exe
//>> .\matrix_schedule_benchmark.exe
//=== Matrix Multiplication Schedule Benchmark (512x512) ===

//--- Thread Count: 1 ---
//Threads: 1 | Schedule: static | Time: 6795 ms
//Threads: 1 | Schedule: dynamic | Time: 6147 ms
//Threads: 1 | Schedule: guided | Time: 6351 ms

//--- Thread Count: 2 ---
//Threads: 2 | Schedule: static | Time: 3708 ms
//Threads: 2 | Schedule: dynamic | Time: 3685 ms
//Threads: 2 | Schedule: guided | Time: 4341 ms

//--- Thread Count: 3 ---
//Threads: 3 | Schedule: static | Time: 3807 ms
//Threads: 3 | Schedule: dynamic | Time: 4064 ms
//Threads: 3 | Schedule: guided | Time: 3377 ms

//--- Thread Count: 4 ---
//Threads: 4 | Schedule: static | Time: 2588 ms
//Threads: 4 | Schedule: dynamic | Time: 2310 ms
//Threads: 4 | Schedule: guided | Time: 2294 ms