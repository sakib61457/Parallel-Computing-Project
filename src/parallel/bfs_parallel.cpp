#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>

void parallel_bfs(int startNode, const std::vector<std::vector<int>>& adj, int n) {
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

int main() {
    int n = 10000;
    std::vector<std::vector<int>> adj(n);
    
    // Generate simple graph connections
    for (int i = 0; i < n - 1; ++i) {
        adj[i].push_back(i + 1);
        adj[i + 1].push_back(i);
    }

    auto start = std::chrono::high_resolution_clock::now();

    parallel_bfs(0, adj, n);

    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "BFS (Parallel Level-Synchronous) Time: " << duration << " ms\n";
    return 0;
}

//g++ -fopenmp bfs_parallel.cpp -o bfs_parallel.exe
//>> $env:OMP_NUM_THREADS=1; .\bfs_parallel.exe
//>> $env:OMP_NUM_THREADS=2; .\bfs_parallel.exe
//>> $env:OMP_NUM_THREADS=4; .\bfs_parallel.exe
//>> $env:OMP_NUM_THREADS=8; .\bfs_parallel.exe
//BFS (Parallel Level-Synchronous) Time: 92.3389 ms
//BFS (Parallel Level-Synchronous) Time: 501.17 ms
//BFS (Parallel Level-Synchronous) Time: 784.36 ms
//BFS (Parallel Level-Synchronous) Time: 1025.31 ms