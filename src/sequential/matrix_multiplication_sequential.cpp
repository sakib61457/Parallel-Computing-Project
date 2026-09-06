#include <iostream>
#include <vector>

int main() {
    int r1, c1, r2, c2;

    std::cout << "Enter rows and columns for Matrix A (R1 C1): ";
    std::cin >> r1 >> c1;

    std::cout << "Enter rows and columns for Matrix B (R2 C2): ";
    std::cin >> r2 >> c2;

    // Matrix multiplication constraint check: C1 must equal R2
    if (c1 != r2) {
        std::cout << "\nError: Matrix multiplication not possible!" << std::endl;
        std::cout << "Columns of Matrix A (" << c1 
                  << ") must equal Rows of Matrix B (" << r2 << ")." << std::endl;
        return 1;
    }

    std::vector<std::vector<int>> A(r1, std::vector<int>(c1));
    std::vector<std::vector<int>> B(r2, std::vector<int>(c2));
    std::vector<std::vector<int>> C(r1, std::vector<int>(c2, 0));

    // Input Matrix A
    std::cout << "\nEnter elements of Matrix A (" << r1 << "x" << c1 << "):\n";
    for (int i = 0; i < r1; ++i) {
        for (int j = 0; j < c1; ++j) {
            std::cin >> A[i][j];
        }
    }

    // Input Matrix B
    std::cout << "\nEnter elements of Matrix B (" << r2 << "x" << c2 << "):\n";
    for (int i = 0; i < r2; ++i) {
        for (int j = 0; j < c2; ++j) {
            std::cin >> B[i][j];
        }
    }

    // Sequential Matrix Multiplication O(N^3)
    for (int i = 0; i < r1; ++i) {
        for (int j = 0; j < c2; ++j) {
            for (int k = 0; k < c1; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    // Display Result
    std::cout << "\nResultant Matrix C (" << r1 << "x" << c2 << "):\n";
    for (int i = 0; i < r1; ++i) {
        for (int j = 0; j < c2; ++j) {
            std::cout << C[i][j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}