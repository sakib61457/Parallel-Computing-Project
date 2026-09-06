#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
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

void par_merge_sort(std::vector<int>& arr, int l, int r) {
    if (l < r) {
        int m = l + (r - l) / 2;
        if (r - l > 1000) {
            // shared(arr) ensures tasks access the main vector
            // l, m, r are passed by value (firstprivate implicitly or explicitly)
            #pragma omp task shared(arr) firstprivate(l, m)
            par_merge_sort(arr, l, m);

            #pragma omp task shared(arr) firstprivate(m, r)
            par_merge_sort(arr, m + 1, r);

            #pragma omp taskwait
        } else {
            par_merge_sort(arr, l, m);
            par_merge_sort(arr, m + 1, r);
        }
        merge(arr, l, m, r);
    }
}

int main() {
    int N = 100000;
    int num_runs = 10;
    int threads = 4;
    
    omp_set_num_threads(threads);
    std::cout << "Testing Parallel Merge Sort for Race Conditions (" << num_runs << " runs, " << threads << " threads)...\n\n";

    bool all_passed = true;

    for (int run = 1; run <= num_runs; ++run) {
        // Generate a new random dataset for each run
        std::mt19937 rng(run * 100); 
        std::vector<int> arr(N);
        for (int i = 0; i < N; ++i) arr[i] = rng() % 100000;

        #pragma omp parallel
        {
            #pragma omp single
            par_merge_sort(arr, 0, N - 1);
        }

        // Verify sorting correctness
        bool is_correct = std::is_sorted(arr.begin(), arr.end());

        if (is_correct) {
            std::cout << "Run " << run << " -> Correct\n";
        } else {
            std::cout << "Run " << run << " -> INCORRECT (Race Condition Detected!)\n";
            all_passed = false;
        }
    }

    std::cout << "\n----------------------------------------\n";
    if (all_passed) {
        std::cout << "SUCCESS: All " << num_runs << " runs sorted correctly. No race conditions detected!\n";
    } else {
        std::cout << "FAILURE: Race condition present in shared variables or task boundaries.\n";
    }

    return 0;
}