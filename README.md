# parallel_floyd

Solves the shortest Path problem using a MPI based pipelined parallel processed implementation of Floyd-Warshall Algorithm.

https://en.wikipedia.org/wiki/Parallel_all-pairs_shortest_path_algorithm#Pipelined_2-D_block_mapping

*** How Matrix is generated
- Random if i != j with arbitrary connect rate. Infinity is represented by INT_MAX
- 0 if i == j 

*** How the original array is scattered
- The 2D matrix is transformed in a 1D array, where E[i][j] is mapped to A[i*n+j]
- A spiral traversal algorithm distribute to each processor a square section of the original matrix using basic math and then scatter the result to each processor

*** How communication is established
- 2 MPI_Comm_split is used. All ith col are in the same comm and all jth row are in the same comm
- To know how is the sender at dimension D, we use the new comm's rank to determine the correct sending processor
- Basic math arithmatic is used to know which section is to be sent

*** Difference between Pipeline and Non-pipeline
- Two algo are almost identical. In Pipeline MPI_Isend is in lieu of broadcast

*** How results are validated
- After inputs are generated, the program first executes parallel floyd. Results are gathered in Master.
- Using the same input, Master executes serial floyd. The result of Parallel floyd and Serial floyd are outputed in two files. The program compares the two results for discrepancies 

# How to run

Compile using 
mpicc -std=c11 -lm -o floyd_2.o floyd_pipeline.c

Execute without printing matrix and stack trace
mpirun -np #Processor floyd_2.o #SizeOfN

Execute with printed matrix, but without stack trace
mpirun -np #Processor floyd_2.o #SizeOfN anyArg

Execute with printed matrix, but without stack trace
mpirun -np #Processor floyd_2.o #SizeOfN anyArg anyArg

Note that #Processor must be a sqrt of Size of Matrix
