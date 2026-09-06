#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>

int main() {
    int n = 512; // 512 x 512 Matrix
    std::vector<std::vector<int>> A(n, std::vector<int>(n, 1));
    std::vector<std::vector<int>> B(n, std::vector<int>(n, 2));
    std::vector<std::vector<int>> C(n, std::vector<int>(n, 0));

    auto start = std::chrono::high_resolution_clock::now();

    // Parallelize outer row loop
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n; ++k) {
            for (int j = 0; j < n; ++j) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "Matrix Multiplication (Parallel For) Time: " << duration << " ms\n";
    return 0;
}

// g++ -fopenmp matrix_multiplication_parallel.cpp -o matrix_multiplication_parallel.exe                      
// >> $env:OMP_NUM_THREADS=1; .\matrix_multiplication_parallel.exe
// >> $env:OMP_NUM_THREADS=2; .\matrix_multiplication_parallel.exe
// >> $env:OMP_NUM_THREADS=4; .\matrix_multiplication_parallel.exe
// >> $env:OMP_NUM_THREADS=8; .\matrix_multiplication_parallel.exe
// Matrix Multiplication (Parallel For) Time: 5839.39 ms
// Matrix Multiplication (Parallel For) Time: 3738.13 ms
// Matrix Multiplication (Parallel For) Time: 2508.24 ms
// Matrix Multiplication (Parallel For) Time: 1431.78 ms