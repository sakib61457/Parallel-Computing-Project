#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>

void merge(std::vector<int>& arr, int l, int m, int r) {
    std::vector<int> left(arr.begin() + l, arr.begin() + m + 1);
    std::vector<int> right(arr.begin() + m + 1, arr.begin() + r + 1);

    int i = 0, j = 0, k = l;
    while (i < left.size() && j < right.size()) {
        arr[k++] = (left[i] <= right[j]) ? left[i++] : right[j++];
    }
    while (i < left.size()) arr[k++] = left[i++];
    while (j < right.size()) arr[k++] = right[j++];
}

void parallel_merge_sort(std::vector<int>& arr, int l, int r) {
    if (l < r) {
        int m = l + (r - l) / 2;

        // Cutoff threshold: use tasks only for sub-arrays larger than 1000 elements
        if (r - l > 1000) {
            #pragma omp task shared(arr)
            parallel_merge_sort(arr, l, m);

            #pragma omp task shared(arr)
            parallel_merge_sort(arr, m + 1, r);

            #pragma omp taskwait
        } else {
            parallel_merge_sort(arr, l, m);
            parallel_merge_sort(arr, m + 1, r);
        }

        merge(arr, l, m, r);
    }
}

int main() {
    int n = 1000000; // 1M elements
    std::vector<int> arr(n);
    for (int i = 0; i < n; ++i) arr[i] = n - i;

    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel
    {
        #pragma omp single
        {
            parallel_merge_sort(arr, 0, n - 1);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "Merge Sort (Parallel Tasks) Time: " << duration << " ms\n";
    return 0;
}

// g++ -fopenmp merge_sort_parallel.cpp -o merge_sort_parallel.exe
// >> $env:OMP_NUM_THREADS=1; .\merge_sort_parallel.exe
// >> $env:OMP_NUM_THREADS=2; .\merge_sort_parallel.exe
// >> $env:OMP_NUM_THREADS=4; .\merge_sort_parallel.exe
// >> $env:OMP_NUM_THREADS=8; .\merge_sort_parallel.exe
// Merge Sort (Parallel Tasks) Time: 967.777 ms
// Merge Sort (Parallel Tasks) Time: 615.144 ms
// Merge Sort (Parallel Tasks) Time: 543.704 ms
// Merge Sort (Parallel Tasks) Time: 326.917 ms