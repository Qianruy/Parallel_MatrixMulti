# Introduction
Implement Parallel Matrix-Vector Multiplication and Parallel Jacobi's Method
# Input & Output Format
The input to the algorithm is two files containing the matrix A and vector b. \
The output of the program is the vector x. File format is described in more details below. \
The program should take 3 command line arguments: `input mat.txt`, `input vec.txt` and `output.txt`. \
Sample command line input: \
   `$ mpiexec -np 4 ./pjacobi input_mat.txt input_vec.txt output.txt`
# File Format
`input_mat.txt`: contains n + 1 lines for an n × n matrix A 
* First line: value of n
* Following n lines: space-delimited elements of the matrix \
(Datatype - Doubles)

`input_vec.txt`: contains 1 line for the vector b
* First line: n space-delimited elements of vector b \
(Datatype - Doubles)

`output.txt`: contains 1 line for the vector x
* First line: n space-delimited elements of vector x \
(Datatype - Doubles, up to 16 decimal points)
