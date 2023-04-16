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

// Compute the local length for each processor
int block_division(int mat_size, int mesh_size, int rank) {
    return mat_size / mesh_size + ((rank < mat_size % mesh_size) ? 1 : 0);
}

// Block distributes vectors along the first column
vector<double> distribute_vect(int n, vector<double> &ori_vect, MPI_Comm comm) {

    vector<double> local_vec;

    int rank, size;
    int coords[NDIM];
    
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    MPI_Cart_coords(comm, rank, NDIM, coords);

    // Create a new communicator that represents a column of processes 
    int remain_dims[NDIM] = {true, false};
    MPI_Comm comm_col;
    MPI_Cart_sub(comm, remain_dims, &comm_col);

    int mesh_size = (int) sqrt(size);
    // Division based on column coordinate
    int subsize = block_division(n, mesh_size, coords[0]); 

    // Deal with the first column
    if (coords[1] == 0) {
        local_vec.resize(subsize);
        
        // Prepare arguments for Scatterv
        int* scounts = new int[mesh_size];
        int* displs = new int[mesh_size];
        displs[0] = 0;

        for (int i = 0; i < mesh_size; i++) {
            scounts[i] = block_division(n, mesh_size, i);
            if (i > 0) {
                displs[i] = displs[i - 1] + scounts[i - 1];
            }
        }

        MPI_Scatterv(&ori_vect[0], scounts, displs, MPI_DOUBLE, &local_vec[0], 
            subsize, MPI_DOUBLE, 0, comm_col);
    }

    MPI_Comm_free(&comm_col);

    return local_vec;
}

vector<vector<double>> distribute_matrix(int n, vector<double> &ori_matrix, MPI_Comm comm) {
    
    vector<vector<double>> local_matrix;

    int rank, size;
    int coords[NDIM];
    
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    MPI_Cart_coords(comm, rank, NDIM, coords);

    // Step 1: distribute the matrix to the first column of the processers
    // The matrix experiences 1D partitioning
    // Create a new communicator that represents a column of processes 
    int remain_dims[NDIM] = {true, false};
    MPI_Comm comm_col;
    MPI_Cart_sub(comm, remain_dims, &comm_col);

    // Prepare arguments for Scatterv
    int mesh_size = (int) sqrt(size);
    int* scounts = new int[mesh_size];
    int* displs = new int[mesh_size];
    for (int i = 0; i < mesh_size; i++) {
        scounts[i] = n * block_division(n, mesh_size, i); // n * number of rows
        displs[i] = (i == 0) ? 0 : (displs[i - 1] + scounts[i - 1]);
    }

    // Memory allocation
    vector<double> transfer_data;
    transfer_data.resize(scounts[coords[0]]);

    // Use the first column as transfer station
    if (coords[1] == 0) {
        MPI_Scatterv(&ori_matrix[0], scounts, displs, MPI_DOUBLE, &transfer_data[0], 
            scounts[coords[0]], MPI_DOUBLE, 0, comm_col);
        MPI_Barrier(comm_col);
        #ifdef DEBUG
            for (int i = 0; i < scounts[coords[0]]; i++)
                cout << transfer_data[i] << " ";
            cout << endl;
        #endif 
    }

    // Step 2: distribute the blocks in the first processor of each row 
    // among the row communicator to the rest of processors in the same row
    // Create a new communicator that represents a row of processes 
    remain_dims[0] = false; 
    remain_dims[1] = true;
    MPI_Comm comm_row;
    MPI_Cart_sub(comm, remain_dims, &comm_row);

    // Division based on row and column coordinate
    int subsize_row = block_division(n, mesh_size, coords[1]); 
    int subsize_col = block_division(n, mesh_size, coords[0]); 
    int subsize = subsize_row * subsize_col;

    // Prepare arguments for Scatterv
    int* scounts_row = new int[mesh_size];
    int* displs_row = new int[mesh_size];
    for (int i = 0; i < mesh_size; i++) {
        scounts_row[i] = block_division(n, mesh_size, i); // division among columns
        displs_row[i] = (i == 0) ? 0 : (displs_row[i - 1] + scounts_row[i - 1]);
    }

    // Use the row communicator: scatter by each row
    local_matrix.resize(subsize_row);
    for (int i = 0; i < subsize_row; i++) {
        local_matrix[i].resize(subsize_col);
        MPI_Scatterv(&transfer_data[i * n], scounts_row, displs_row, MPI_DOUBLE, 
            &local_matrix[i][0], subsize_col, MPI_DOUBLE, 0, comm_row);
    }

    MPI_Barrier(comm_row);

    return local_matrix;
}

int main(int argc, char *argv[]) {
    MPI_Status status;
    // Initialize the MPI environment
    MPI_Comm comm;

    MPI_Init(&argc, &argv);

    // Get the number and rank of processes
    int world_size, world_rank, cart_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Cartesian topology
    int q = (int)sqrt(world_size); 
    int dims[NDIM] = {q, q}; // Mesh dimension
    int periods[NDIM] = {1, 1}; // Whether the mesh is periodic or not in each dimension
    int coords[NDIM]; // Cartesian coordinates of the current process

    // Function is called to find the optimal dimensions of the Cartesian grid 
    // given the total number of processes world_size and the number of dimensions NDIM 
    // The resulting dimensions are stored in dims.
    MPI_Dims_create(world_size, NDIM, dims);

    // Function is called to create the Cartesian communicator comm using the dimensions in dims. 
    // The last argument 1 specifies that the Cartesian coordinates are in row-major order.
    MPI_Cart_create(MPI_COMM_WORLD, NDIM, dims, periods, 1, &comm);

    // Returns the rank of the process with the given coordinates in the Cartesian communicator
    // Determine the root process in the Cartesian communicator and stored in rootRank
    // int rootRank, root_coords[NDIM] = {0, 0};
    // MPI_Cart_rank(comm, root_coords, &rootRank);

    // Determine the Cartesian coordinates of the current process with Cartesian rank cart_rank
    // and store them in coords
    MPI_Comm_rank(comm, &cart_rank);
    MPI_Cart_coords(comm, cart_rank, NDIM, coords);

    int n;
    vector<double> flatten_A, b;
    vector<vector<double>> A;

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
        mat_file >> n;
        A.resize(n);
        flatten_A.resize(n * n);
        for (int i = 0; i < n; i++) {
            A[i].resize(n);
            for (int j = 0; j < n; j++) {
                mat_file >> A[i][j];
                flatten_A[i * n + j] = A[i][j];
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
        #ifdef DEBUG
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    cout << A[i][j] << " ";
                }
                cout << endl;
            }
            for (int i = 0; i < n; i++) {
                cout << b[i] << endl;
            }
        #endif
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, comm);

    vector<double> local_b = distribute_vect(n, b, comm);
    MPI_Barrier(comm);
    #ifdef DEBUG
        if (coords[1] == 0) {
            cout << "world rank: " << world_rank << " dim0:" << coords[0] << " dim1:" << coords[1] << endl;
            cout << "vector size: " << local_b.size() << endl;
        }
    #endif

    vector<vector<double>> local_A = distribute_matrix(n, flatten_A, comm);
    #ifdef DEBUG
        for (int i = 0; i < local_A.size(); i++)
            for (int j = 0; j < local_A[0].size(); j++)
                cout << local_A[i][j] << " ";
            cout << endl;
    #endif

    // Finalize the MPI environment. No more MPI calls can be made after this
    MPI_Comm_free(&comm);
    MPI_Finalize();
    return 0;
}