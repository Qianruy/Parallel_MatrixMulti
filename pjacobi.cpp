#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>

#define NDIM 2
#define MAXITER 1E6
#define THRESHOLD 10E-9

using namespace std;

// Compute the local length for each processor
int block_division(int mat_size, int mesh_size, int rank) {
    return mat_size / mesh_size + ((rank < mat_size % mesh_size) ? 1 : 0);
}

// function overloading
int block_division(int mat_size, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    return block_division(mat_size, size, rank);
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
    // Division based on row coordinate
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

// transpose vector x
void transpose_vect(int n, vector<double> &x, vector<double> &result, MPI_Comm comm) {

    int rank, size;
    int coords[NDIM];
    
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    MPI_Cart_coords(comm, rank, NDIM, coords);

    // Create a new communicator that represents a row of processes 
    int remain_dims[NDIM] = {false, true};
    MPI_Comm comm_row;
    MPI_Cart_sub(comm, remain_dims, &comm_row);

    // Create a new communicator that represents a column of processes 
    remain_dims[0] = true;
    remain_dims[1] = false;
    MPI_Comm comm_col;
    MPI_Cart_sub(comm, remain_dims, &comm_col);

    int rowSize, rowRank, colSize, colRank;
    MPI_Comm_size(comm_row,&rowSize);
    MPI_Comm_rank(comm_row,&rowRank);
    MPI_Comm_size(comm_col,&colSize);
    MPI_Comm_rank(comm_col,&colRank);

    // rank of (i, 0)
    int rank00;
    int rootCoords = 0;
    MPI_Cart_rank(comm_row,&rootCoords,&rank00);

    // rank of diagonal
    int diagCords = colRank;
    int rankRowDiag;
    MPI_Cart_rank(comm_row,&diagCords,&rankRowDiag);

    // col rank of diagonal
    diagCords = rowRank;
    int rankColDiag;
    MPI_Cart_rank(comm_col,&diagCords,&rankColDiag);

    int nCols = block_division(n, colSize, colRank);

    // send elements from first column processoer to diagonal
    if (coords[1] == 0) {
        if (coords[0] == 0) {
            for (int i = 0; i < nCols; i++) {
                result[i] = x[i];
            }
        } else {
            MPI_Send(&x[0], nCols, MPI_DOUBLE, rankRowDiag, 0, comm_row);
        }
    }

    if (coords[0] == coords[1] && coords[0] != 0) {
        MPI_Recv(&result[0], nCols, MPI_DOUBLE, rank00, 0, comm_row, MPI_STATUS_IGNORE);
    }

    MPI_Barrier(comm);

    int nRows = block_division(n, rowSize, rowRank);

    // broadcast received elements from (i, i) along each column
    MPI_Bcast(&result[0], nRows, MPI_DOUBLE, rankColDiag, comm_col);

    MPI_Barrier(comm);

    MPI_Comm_free(&comm_row);
    MPI_Comm_free(&comm_col);

}

// parallel matrix-vector multiplication
vector<double> mat_vect_mult(int n, vector<vector<double>> &A, vector<double> &x, MPI_Comm comm) {
    vector<double> result;

    int rank, size;
    int coords[NDIM];
    
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    MPI_Cart_coords(comm, rank, NDIM, coords);

    // split by row and column separately
    MPI_Comm comm_row, comm_col;
    MPI_Comm_split(comm, coords[0], coords[1], &comm_row);
    MPI_Comm_split(comm, coords[1], coords[0], &comm_col);

    // local row and column length for current process
    int rowLen = block_division(n, comm_row);
    int colLen = block_division(n, comm_col);

    vector<double> local_row(rowLen);
    transpose_vect(n, x, local_row, comm);

    vector<double> local_result(colLen);

    for (int i = 0; i < colLen; i++) {
        for (int j = 0; j < rowLen; j++) {
            local_result[i] += local_row[j] * A[i][j];
        }
    }

    // Reduction: sum up local results to the first column
    result.resize(colLen);
    MPI_Reduce(&local_result[0], &result[0], colLen, MPI_DOUBLE, MPI_SUM, 0, comm_row);

    // set processor not in first column as empty
    if (coords[1] != 0) { result.clear(); }

    MPI_Comm_free(&comm_row);
    MPI_Comm_free(&comm_col);

    return result;
}

// Gather local vectors in (i, 0) to the processor (0, 0)
void gatherToRoot(int n, vector<double> &pro_vec, vector<double> &result_vec, MPI_Comm comm) {
    int rank, size;
    int coords[NDIM];

    // Get grid info in 2D meesh
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    MPI_Cart_coords(comm, rank, NDIM, &coords[0]);

   
    int mesh_size = (int)sqrt(size);
    int subsize_row  = block_division(n, mesh_size, coords[0]);

    // Create Column communicator
    int remain_dims[NDIM] = {true, false};
    MPI_Comm comm_col;
    MPI_Cart_sub(comm, remain_dims, &comm_col);

    // Perform GatherV: all the elements are only on the 1st column
    if (coords[1] == 0) {
        //Prepare parameters
        int* rcounts = new int[mesh_size];
        int* displs = new int[mesh_size];
        for (int i = 0; i < mesh_size ; i++) {
            rcounts[i] = block_division(n, mesh_size, i);
            displs[i] = (i == 0) ? 0 : displs[i - 1] + rcounts[i - 1];
        }

        cout << "debug1: " << coords[0] << ", " << coords[1] << endl;
        MPI_Gatherv(&pro_vec[0], subsize_row, MPI_DOUBLE, &result_vec[0], rcounts, displs, MPI_DOUBLE, 0, comm_col);
        cout << "debug2: " << coords[0] << ", " << coords[1] << endl;
    }
    MPI_Comm_free(&comm_col);
    return;
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
    int rootRank, root_coords[NDIM] = {0, 0};
    MPI_Cart_rank(comm, root_coords, &rootRank);

    // Determine the Cartesian coordinates of the current process with Cartesian rank cart_rank
    // and store them in coords
    MPI_Comm_rank(comm, &cart_rank);
    MPI_Cart_coords(comm, cart_rank, NDIM, coords);

    int n;
    vector<double> flatten_A, flatten_R, b, d;
    vector<vector<double>> A;

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
        A.resize(n);
        d.resize(n);
        flatten_A.resize(n * n);
        flatten_R.resize(n * n);
        for (int i = 0; i < n; i++) {
            A[i].resize(n);
            for (int j = 0; j < n; j++) {
                mat_file >> A[i][j];
                flatten_A[i * n + j] = A[i][j];
                flatten_R[i * n + j] = A[i][j] * (1 - (i == j));
            }
            d[i] = A[i][i];
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

    MPI_Bcast(&n, 1, MPI_INT, rootRank, comm);

    // Prepare for Jacobia Iteration
    vector<vector<double>> local_A = distribute_matrix(n, flatten_A, comm);
    vector<vector<double>> local_R = distribute_matrix(n, flatten_R, comm);
    vector<double> local_b = distribute_vect(n, b, comm);
    // Distribute the diagnal elements D along the first column of the grid
    vector<double> local_d = distribute_vect(n, d, comm);
    // Store intermediate results
    vector<double> local_res, local_b_hat;
    // Initiate x to [0, ..., 0]
    vector<double> x;
    if (world_rank == rootRank) { 
        x.resize(n);
        fill_n(x.begin(), n, 0); 
    }
    vector<double> local_x = distribute_vect(n, x, comm);
    // Initiate termination criteria
    double ssd = 1.0;
    int iter_num = 0;
    MPI_Barrier(comm);

    // Create Column communicator
    int remain_dims[NDIM] = {true, false};
    MPI_Comm comm_col;
    MPI_Cart_sub(comm, remain_dims, &comm_col);
    
    // Begin Iteration
    while (ssd > THRESHOLD && iter_num < MAXITER) {
        // Update x:
        // Calculte local Rx 
        local_res = mat_vect_mult(n, local_R, local_x, comm);
        // Calculate local D^(-1)(b - Rx)
        if (coords[1] == 0) {
            for (int i = 0; i < local_b.size(); i++) {
                local_x[i] = (local_b[i] - local_res[i]) / local_d[i];
            }
        }
        MPI_Barrier(comm);
        // Compute termination criteria:
        // Calculate the sum of square diff locally
        local_b_hat = mat_vect_mult(n, local_A, local_x, comm);
        double local_ssd = 0.0;
        if (coords[1] == 0) {
            for (int i = 0; i < local_b.size(); i++) {
                local_ssd += (local_b_hat[i] - local_b[i]) * (local_b_hat[i] - local_b[i]);
            }
        }
        MPI_Barrier(comm);
        // Perform parallel reduction along the first column
        if (coords[1] == 0) {
            MPI_Allreduce(&local_ssd, &ssd, 1, MPI_DOUBLE, MPI_SUM, comm_col);
        }
        if (world_rank == rootRank) {
            cout << iter_num << endl;
            cout << ssd << endl;
        }
        iter_num++;
    }
    #ifdef DEBUG
        if (coords[1] == 0) {
            cout << "Final result: " << endl;
            for (int i = 0; i < local_x.size(); i++){
                cout << local_x[i] << " ";
            }
            cout<<endl;
        }
    #endif
    MPI_Comm_free(&comm_col);

    vector<double> x_final;
    // Gatehr vector results to root processor 
    if (world_rank == 0) {
        x_final.resize(n);
    }
    
    gatherToRoot(n, local_x, x_final, comm);
    #ifdef DEBUG
        if (coords[0] == 0 && coords[1] == 0) {
            cout << "Final result: " << endl;
            for (int i = 0; i < x_final.size(); i++){
                cout << x_final[i] << " ";
            }
            cout<<endl;
        }
    #endif

    // Finalize the MPI environment. No more MPI calls can be made after this
    MPI_Comm_free(&comm);
    MPI_Finalize();
    return 0;
}