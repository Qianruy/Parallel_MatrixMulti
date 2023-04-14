#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#define NDIM 2

using namespace std;

int main(int argc, char *argv[]) {
    MPI_Status status;
    // Initialize the MPI environment
    MPI_Comm comm;

    MPI_Init(&argc, &argv);

    // Get the number and rank of processes
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    // cartesian topology
    int dims[2] = {0, 0};
    MPI_Dims_create(world_size, NDIM, dims);

    int periods[2] = {1, 1};
    MPI_Cart_create(MPI_COMM_WORLD, NDIM, dims, periods, 1, &comm);

    // input stream
    if (world_rank == 0) {
        if (argc != 4) { return 1; }

        ifstream mat_file(argv[1]);
        ifstream vec_file(argv[2]);

        // Check if the files were opened successfully
        if (!mat_file || !vec_file) {
            cout << "Error opening input files" << endl;
            return 1;
        }
        // Read input_mat.txt
        int n;
        mat_file >> n;
        vector<vector<double>> A(n, vector<double>(n));
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                mat_file >> A[i][j];
            }
        }
        // Read input_vec.txt
        vector<double> b(n);
        for (int i = 0; i < n; i++) {
            vec_file >> b[i];
        }
        // Close the input files
        mat_file.close();
        vec_file.close();
        // debug print
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                cout << A[i][j] << " ";
            }
            cout << endl;
        }
        for (int i = 0; i < n; i++) {
            cout << b[i] << endl;
        }
    }


    // Finalize the MPI environment. No more MPI calls can be made after this
    MPI_Finalize();
    return 0;
}