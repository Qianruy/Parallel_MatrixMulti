#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>

#define NDIM 2

using namespace std;

// compute the local length for each processor
int block_decompose(int mat_size, int mesh_size, int rank) {
    return mat_size / mesh_size + ((rank < mat_size % mesh_size) ? 1 : 0);
}

// block distributes vectors along the first column
vector<double> distribute_vect(int n, vector<double> & ori_vect, MPI_Comm comm) {

    vector<double> local_vec;

    int rank, size;
    int coords[NDIM];

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    MPI_Cart_coords(comm, rank, NDIM, coords);

    int remain_dims[2] = {true, false};
    MPI_Comm comm_col;
    MPI_Cart_sub(comm, remain_dims, &comm_col);

    int mesh_size = (int) sqrt(size);
    int subsize = block_decompose(n, mesh_size, coords[0]);

    if (coords[1] == 0) {
        local_vec.resize(subsize);
        
        // prepare arguments for Scatterv
        int* scounts = new int[mesh_size];
        int* displs = new int[mesh_size];
        displs[0] = 0;

        for (int i = 0; i < mesh_size; i++) {
            scounts[i] = block_decompose(n, mesh_size, i);
            if (i > 0) {
                displs[i] = displs[i - 1] + scounts[i - 1];
            }
        }

        MPI_Scatterv(&ori_vect[0], scounts, displs, MPI_DOUBLE, &local_vec[0], subsize, MPI_DOUBLE, 0, comm_col);
    }

    MPI_Comm_free(&comm_col);
    
    return local_vec;
}

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
    int dims[2];
    int q = (int)sqrt(world_size);
    dims[0] = dims[1] = q;

    MPI_Dims_create(world_size, NDIM, dims);

    int periods[2] = {1, 1};
    MPI_Cart_create(MPI_COMM_WORLD, NDIM, dims, periods, 1, &comm);

    int rootRank;
    int root_2d[2] = {0, 0};
    MPI_Cart_rank(comm, root_2d, &rootRank);
    int rank_2d, corr_2d[2];
    MPI_Comm_rank(comm, &rank_2d);
    MPI_Cart_coords(comm, rank_2d,2, corr_2d);

    int n;
    vector<double> b;

    // input stream
    if (world_rank == rootRank) {
        if (argc != 4) { return 1; }

        ifstream mat_file(argv[1]);
        ifstream vec_file(argv[2]);

        // Check if the files were opened successfully
        if (!mat_file || !vec_file) {
            cout << "Error opening input files" << endl;
            return 1;
        }
        // Read input_mat.txt
        mat_file >> n;
        vector<vector<double>> A(n, vector<double>(n));
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                mat_file >> A[i][j];
            }
        }
        // Read input_vec.txt
        b.resize(n);
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

    MPI_Bcast(&n, 1, MPI_INT, rootRank, comm);

    vector<double> local_b = distribute_vect(n, b, comm);

    int coords[NDIM];
    MPI_Cart_coords(comm, world_rank, NDIM, coords);

    MPI_Barrier(comm);

    if (coords[1] == 0) {
        cout << "world rank: " << world_rank << " dim0:" << coords[0] << " dim1:" << coords[1] << endl;
        cout << "vector size:" << local_b.size() << endl;
    }

    // Finalize the MPI environment. No more MPI calls can be made after this
    MPI_Finalize();
    return 0;
}