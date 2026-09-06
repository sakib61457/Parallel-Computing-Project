#include <iostream>
#include <vector>
#include <queue>

// Function to perform Breadth-First Search
void bfs(int startNode, const std::vector<std::vector<int>>& adj, int numVertices) {
    std::vector<bool> visited(numVertices, false);
    std::queue<int> q;

    // Initialize BFS with starting node
    visited[startNode] = true;
    q.push(startNode);

    std::cout << "\nBFS Traversal starting from node " << startNode << ": ";

    while (!q.empty()) {
        int currentNode = q.front();
        q.pop();

        std::cout << currentNode << " ";

        // Visit all unvisited neighbors of currentNode
        for (int neighbor : adj[currentNode]) {
            if (!visited[neighbor]) {
                visited[neighbor] = true;
                q.push(neighbor);
            }
        }
    }
    std::cout << std::endl;
}

int main() {
    int numVertices, numEdges;

    std::cout << "Enter the number of vertices: ";
    std::cin >> numVertices;

    std::cout << "Enter the number of edges: ";
    std::cin >> numEdges;

    if (numVertices <= 0) {
        std::cout << "Number of vertices must be greater than 0." << std::endl;
        return 0;
    }

    // Adjacency list representation
    std::vector<std::vector<int>> adj(numVertices);

    std::cout << "\nEnter " << numEdges << " edges (u v pairs, 0-indexed):\n";
    for (int i = 0; i < numEdges; ++i) {
        int u, v;
        std::cin >> u >> v;

        // Validating 0-based index range
        if (u >= 0 && u < numVertices && v >= 0 && v < numVertices) {
            adj[u].push_back(v);
            adj[v].push_back(u); // Remove this line if working with a directed graph
        } else {
            std::cout << "Invalid edge (" << u << ", " << v << "). Node values must be between 0 and " << numVertices - 1 << ".\n";
            i--; // Re-prompt for this edge
        }
    }

    int startNode;
    std::cout << "\nEnter the starting source node for BFS: ";
    std::cin >> startNode;

    if (startNode < 0 || startNode >= numVertices) {
        std::cout << "Invalid start node!" << std::endl;
        return 0;
    }

    // Execute BFS
    bfs(startNode, adj, numVertices);

    return 0;
}