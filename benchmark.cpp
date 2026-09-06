#include <iostream>
#include <vector>
#include <fstream>
#include <random>
#include <omp.h>

// ==========================================
// CSV LOGGING FUNCTION
// ==========================================
void log_csv(std::ofstream& csv, const std::string& algo, const std::string& version, 
             const std::string& inputSize, int threads, double timeMs) {
    csv << algo << "," << version << "," << inputSize << "," << threads << "," << timeMs << "\n";
    csv.flush();
}

// ==========================================
// 1. MERGE SORT IMPLEMENTATION
// ==========================================
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

void seq_merge_sort(std::vector<int>& arr, int l, int r) {
    if (l < r) {
        int m = l + (r - l) / 2;
        seq_merge_sort(arr, l, m);
        seq_merge_sort(arr, m + 1, r);
        merge(arr, l, m, r);
    }
}

void par_merge_sort(std::vector<int>& arr, int l, int r) {
    if (l < r) {
        int m = l + (r - l) / 2;
        if (r - l > 1000) { // Task cutoff threshold
            #pragma omp task shared(arr)
            par_merge_sort(arr, l, m);
            #pragma omp task shared(arr)
            par_merge_sort(arr, m + 1, r);
            #pragma omp taskwait
        } else {
            par_merge_sort(arr, l, m);
            par_merge_sort(arr, m + 1, r);
        }
        merge(arr, l, m, r);
    }
}

// ==========================================
// 2. MATRIX MULTIPLICATION IMPLEMENTATION
// ==========================================
void seq_matrix_mult(const std::vector<std::vector<int>>& A, 
                     const std::vector<std::vector<int>>& B, 
                     std::vector<std::vector<int>>& C, int n) {
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n; ++k) {
            for (int j = 0; j < n; ++j) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void par_matrix_mult(const std::vector<std::vector<int>>& A, 
                     const std::vector<std::vector<int>>& B, 
                     std::vector<std::vector<int>>& C, int n) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n; ++k) {
            for (int j = 0; j < n; ++j) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// ==========================================
// 3. BFS IMPLEMENTATION
// ==========================================
void seq_bfs(int startNode, const std::vector<std::vector<int>>& adj, int n) {
    std::vector<bool> visited(n, false);
    std::vector<int> frontier;
    visited[startNode] = true;
    frontier.push_back(startNode);

    while (!frontier.empty()) {
        std::vector<int> next_frontier;
        for (int u : frontier) {
            for (int v : adj[u]) {
                if (!visited[v]) {
                    visited[v] = true;
                    next_frontier.push_back(v);
                }
            }
        }
        frontier = next_frontier;
    }
}

void par_bfs(int startNode, const std::vector<std::vector<int>>& adj, int n) {
    std::vector<bool> visited(n, false);
    std::vector<int> frontier;
    visited[startNode] = true;
    frontier.push_back(startNode);

    while (!frontier.empty()) {
        std::vector<int> next_frontier;
        #pragma omp parallel for
        for (size_t i = 0; i < frontier.size(); ++i) {
            int u = frontier[i];
            for (int v : adj[u]) {
                if (!visited[v]) {
                    #pragma omp critical
                    {
                        if (!visited[v]) {
                            visited[v] = true;
                            next_frontier.push_back(v);
                        }
                    }
                }
            }
        }
        frontier = next_frontier;
    }
}

// ==========================================
// MAIN BENCHMARK DRIVER
// ==========================================
int main() {
    std::string csvPath = "F:\\BRACU\\CSE706 - Parallel Algorithm\\Parallel_Computing_Project\\benchmark_results.csv";
    std::ofstream csv(csvPath);

    if (!csv.is_open()) {
        std::cerr << "Error: Could not open output file at: " << csvPath << std::endl;
        return 1;
    }

    // Write CSV Header
    csv << "Algorithm,Version,Input Size,Threads,Time (ms)\n";

    std::vector<int> threads_to_test = {1, 2, 4, 8};

    std::cout << "Starting Benchmarks using omp_get_wtime()...\n";

    // ------------------------------------------
    // BENCHMARK 1: MERGE SORT
    // ------------------------------------------
    std::vector<int> sort_sizes = {10000, 100000};
    for (int N : sort_sizes) {
        std::string label = (N == 10000) ? "10K" : "100K";
        
        std::mt19937 rng(42);
        std::vector<int> master_data(N);
        for (int i = 0; i < N; ++i) master_data[i] = rng() % 100000;

        // 1. Sequential Execution (omp_get_wtime)
        std::vector<int> seq_data = master_data;
        double start = omp_get_wtime();
        seq_merge_sort(seq_data, 0, N - 1);
        double end = omp_get_wtime();
        double seq_ms = (end - start) * 1000.0;
        log_csv(csv, "Merge Sort", "Sequential", label, 1, seq_ms);

        // 2. Parallel Executions (omp_get_wtime)
        for (int t : threads_to_test) {
            omp_set_num_threads(t);
            std::vector<int> par_data = master_data;

            start = omp_get_wtime();
            #pragma omp parallel
            {
                #pragma omp single
                par_merge_sort(par_data, 0, N - 1);
            }
            end = omp_get_wtime();
            double par_ms = (end - start) * 1000.0;
            log_csv(csv, "Merge Sort", "Parallel", label, t, par_ms);
        }
    }
    std::cout << "[DONE] Merge Sort\n";

    // ------------------------------------------
    // BENCHMARK 2: MATRIX MULTIPLICATION
    // ------------------------------------------
    std::vector<int> mat_sizes = {256, 512};
    for (int N : mat_sizes) {
        std::string label = std::to_string(N) + "x" + std::to_string(N);

        std::vector<std::vector<int>> A(N, std::vector<int>(N, 1));
        std::vector<std::vector<int>> B(N, std::vector<int>(N, 2));

        // Sequential
        std::vector<std::vector<int>> C_seq(N, std::vector<int>(N, 0));
        double start = omp_get_wtime();
        seq_matrix_mult(A, B, C_seq, N);
        double end = omp_get_wtime();
        double seq_ms = (end - start) * 1000.0;
        log_csv(csv, "Matrix Multiplication", "Sequential", label, 1, seq_ms);

        // Parallel
        for (int t : threads_to_test) {
            omp_set_num_threads(t);
            std::vector<std::vector<int>> C_par(N, std::vector<int>(N, 0));

            start = omp_get_wtime();
            par_matrix_mult(A, B, C_par, N);
            end = omp_get_wtime();
            double par_ms = (end - start) * 1000.0;
            log_csv(csv, "Matrix Multiplication", "Parallel", label, t, par_ms);
        }
    }
    std::cout << "[DONE] Matrix Multiplication\n";

    // ------------------------------------------
    // BENCHMARK 3: BFS
    // ------------------------------------------
    std::vector<int> bfs_sizes = {1000, 10000};
    for (int V : bfs_sizes) {
        std::string label = std::to_string(V) + " Nodes";

        std::vector<std::vector<int>> adj(V);
        for (int i = 0; i < V; ++i) {
            adj[i].push_back((i + 1) % V);
            adj[i].push_back((i + 2) % V);
        }

        // Sequential
        double start = omp_get_wtime();
        seq_bfs(0, adj, V);
        double end = omp_get_wtime();
        double seq_ms = (end - start) * 1000.0;
        log_csv(csv, "BFS", "Sequential", label, 1, seq_ms);

        // Parallel
        for (int t : threads_to_test) {
            omp_set_num_threads(t);
            start = omp_get_wtime();
            par_bfs(0, adj, V);
            end = omp_get_wtime();
            double par_ms = (end - start) * 1000.0;
            log_csv(csv, "BFS", "Parallel", label, t, par_ms);
        }
    }
    std::cout << "[DONE] BFS\n";

    csv.close();
    std::cout << "\nAll benchmarks executed successfully using omp_get_wtime()! Output exported to:\n" << csvPath << std::endl;
    return 0;
}