#include <stdio.h>
#include "mpi.h"
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <limits.h>
#include <time.h>
#include <stdbool.h> 


#define MASTER 0

void srandom(unsigned seed);
int findFloydMin(int a, int b, int c);
void serialFloyd(int *matrix, int n, bool print, bool trace, int *comparisonMatrix);
/*
Compile using 
mpicc -std=c11 -lm -o floyd_2.o floyd_pipeline.c

Execute without printing matrix and stack trace
mpirun -np #Processor floyd_2.o #SizeOfN

Execute with printed matrix, but without stack trace
mpirun -np #Processor floyd_2.o #SizeOfN anyArg

Execute with printed matrix, but without stack trace
mpirun -np #Processor floyd_2.o #SizeOfN anyArg anyArg

Note that #Processor must be a sqrt of Size of Matrix

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

*/

int connectionRate = 85;

int main(int argc, char*argv[])
{

    if (argc < 2 || argc >4)
    {
        printf("Enter n and [optional] print and trace args to enable printing and tracing", argv[0]);
        return 0;
    }

    int n = atoi(argv[1]);

    bool print = false;
    if (argc == 3){
        print = true;
    }

    bool trace = false;
    if (argc == 4){
        trace = true;
    }

    int p, my_rank;
    MPI_Comm childcomm;

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    printf("\nMy rank = %d", my_rank);

    int pRoot = (int) sqrt(p);
    int spacing = n/pRoot;
    

    if ((n*n)%p != 0){
        printf("Matrix must be divisible by square root of P");
        return 0;
    }
    int *entireMatrix;
    int *tempMatrix;
    int *processorMatrix;
    int *x_axis;
    int *y_axis;
    int x = 0;
    int y = 0;

    FILE * fp_parallel;
    

    MPI_Request request;

    printf("\nStarting with %d processors\n", p);
    // Initializing Matrix
    clock_t start_time, end_time_p, end_time_s;
    if (my_rank == 0){
        entireMatrix = malloc(n*n*sizeof(int));
        int count = 0;
        for (int i = 0; i < n; i++){
            for (int j=0; j < n; j++){
                if (i == j) entireMatrix[count] = 0;
                else if (rand()%100 > connectionRate){
                    //printf("inf\t", count);
                    entireMatrix[count] = INT_MAX;
                }
                else{
                    entireMatrix[count] = rand()%20;
                    //printf("%d\t", entireMatrix[count]);
                }
                count++;
            }
        }

        if (print) printf("\nOriginal Matrix:\n");
        for (int i = 0; i < n; i++){
            if (print) printf("\n");
            for (int j=0; j<n;j++){
                if (print){
                    if (entireMatrix[i*n+j] == INT_MAX) printf("(inf)");
                    else printf("(%d)", entireMatrix[i*n+j]);
                    printf("\t");
                }
            }
        }
        int index = 0;
        int pCount = 0;
        tempMatrix = malloc(n*n*sizeof(int));
        x_axis = malloc(p*sizeof(int));
        y_axis = malloc(p*sizeof(int));

        // Scattering matrix
        for (int i =0; i < n; i = i + n/pRoot){
            for (int j =0; j < n; j = j + n/pRoot){
                x_axis[pCount] = i;
                y_axis[pCount] = j;
                if (i != 0 && j !=0){
                    //MPI_Isend(&x_axis[pCount], 1, MPI_INT, pCount, 0, MPI_COMM_WORLD, &request);
                    //MPI_Isend(&y_axis[pCount], 1, MPI_INT, pCount, 1, MPI_COMM_WORLD, &request);
                }
                pCount++;
                if (print) printf("\n");
                for (int i_ = i; i_<i + n/pRoot; i_++){
                    for (int j_= j; j_<j + n/pRoot; j_++){
                        if (print){
                            printf("[%d, %d]", i_, j_);
                            if (entireMatrix[n*i_ + j_] == INT_MAX) printf("[inf]");
                            else printf("[%d]", entireMatrix[n*i_ + j_]);
                            printf("\t");
                        }
                        tempMatrix[index++] = entireMatrix[n*i_ + j_];
                    }
                }
            }
        }

        //printf("\npCount = %d", pCount);
        processorMatrix = malloc(n*n/p*sizeof(int));
        MPI_Scatter(tempMatrix, n*n/p, MPI_INT, processorMatrix, n*n/p, MPI_INT, 0, MPI_COMM_WORLD);
        //MPI_Bcast(&x_axis, p, MPI_INT, 0, MPI_COMM_WORLD);
        //MPI_Bcast(&y_axis, p, MPI_INT, 0, MPI_COMM_WORLD);
        start_time = clock();
    }
    else{
        //printf("Broadcasting to non-master processors");
        //MPI_Recv(&x, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        //MPI_Recv(&y, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        processorMatrix = malloc(n*n/p*sizeof(int));
        MPI_Scatter(tempMatrix, n*n/p, MPI_INT, processorMatrix, n*n/p, MPI_INT, 0, MPI_COMM_WORLD);
    }

    //MPI_Bcast(&x_axis, 4, MPI_INT, 0, MPI_COMM_WORLD);
    //MPI_Bcast(&y_axis, 4, MPI_INT, 0, MPI_COMM_WORLD);

    int sRow;
    int sCol;
    int pCount = 0;

    int* rowToParent = malloc(n*sizeof(int));
    int* colToParent = malloc(n*sizeof(int));
    // Calculate the starting top left corner of the matrix for each [i][j] processor
    for (int i =0; i < n; i = i + n/pRoot){
        for (int j =0; j < n; j = j + n/pRoot){
            if (pCount == my_rank){
                sRow = i;
                sCol = j;
            }
            pCount++;
        }
    }
    // Splitting comms
    MPI_Comm rowComm;
    MPI_Comm colComm;
    int row_rank;
    int col_rank;
    MPI_Comm_split(MPI_COMM_WORLD, sRow, my_rank, &colComm);
    MPI_Comm_split(MPI_COMM_WORLD, sCol, my_rank, &rowComm);
    MPI_Comm_rank(rowComm, &row_rank);
    MPI_Comm_rank(colComm, &col_rank);

    for (int k=0; k < n; k++){

        int* kRow = malloc(n/pRoot*sizeof(int));
        int* kCol = malloc(n/pRoot*sizeof(int));
        int length = n/pRoot;
        // Use basic math and split comm's rank to figure out who is sending at each k
        if (k/length == row_rank){
            //printf("\n ****K=%d, Row sender=%d:\n", k, my_rank);
            int kOffset = k - sRow;
            // Use basic math to figure out how much to send
            for (int i = 0; i < n/pRoot; i++){
                kRow[i] = processorMatrix[kOffset*n/pRoot +i];
            }

            for (int i=0; i < pRoot; i++){
                if (i != k/length){
                    MPI_Isend(kRow, n/pRoot, MPI_INT, i, k, rowComm, &request);
                }
            }
        }
        else{
            MPI_Recv(kRow, n/pRoot, MPI_INT, k/length, k, rowComm, MPI_STATUS_IGNORE);
        }

        // Use basic math and split comm's rank to figure out who is sending at each k

        if (k/length == col_rank){
            if (trace) printf("\n ****K=%d, Col sender=%d:\n", k, my_rank);
            int kOffset = k - sCol;

            for (int i = 0; i < n/pRoot; i++){
                kCol[i] = processorMatrix[n/pRoot*i +kOffset];
            }
            for (int i=0; i < pRoot; i++){
                if (i != k/length){
                    MPI_Isend(kCol, n/pRoot, MPI_INT, i, k, colComm, &request);
                }
            }
        }
        else{
            MPI_Recv(kCol, n/pRoot, MPI_INT, k/length, k, colComm, MPI_STATUS_IGNORE);
        }
        
        // Calculate floyd
        int kRowOffset = k - sRow;
        int kColOffset = k - sCol;
        int *newMatrix = malloc(n*n/p*sizeof(int));
        for (int i=0; i< n/pRoot; i++){
            for (int j =0; j<n/pRoot; j++){
                if (trace) printf("\nFor k=%d, M[%d][%d] = min(%d, %d + %d)\n", k, i+sRow, j+sCol, processorMatrix[i*n/pRoot+j], kRow[j], kCol[i]);
                newMatrix[i*n/pRoot+j] = findFloydMin(processorMatrix[i*n/pRoot+j], kRow[j], kCol[i]);
                //printf("testing test");
            }
        }
        processorMatrix = newMatrix;
        free(kRow);
        free(kCol);
    }



    int* gatherData;

    if (my_rank == 0){
        // Gathering the result and applying comparison with serial floyd
        gatherData = malloc(n*n*sizeof(int));

        MPI_Gather(processorMatrix, n*n/p, MPI_INT, gatherData, n*n/p, MPI_INT, 0, MPI_COMM_WORLD);

        int* finalMatrix = malloc(n*n*sizeof(int));
        int c = 0;

        for (int i =0; i < n; i = i + n/pRoot){
            for (int j =0; j < n; j = j + n/pRoot){
                for (int i_ = i; i_<i + n/pRoot; i_++){
                    for (int j_= j; j_<j + n/pRoot; j_++){
                        finalMatrix[n*i_ + j_] = gatherData[c++];
                    }
                }
            }
        }
        
        printf("\n**Time of Parallel execution %f seconds", (double)(clock()-start_time)/CLOCKS_PER_SEC);
        fp_parallel = fopen("p2_output.txt" ,"a");
        if (print) printf("\nParallel Floy result Matrix:\n");
        for (int i = 0; i < n ; i++){
            if (print) printf("\n");
            fprintf(fp_parallel, "\n");
            for (int j = 0; j < n; j++){
                if (print){
                    if (finalMatrix[i*n+j] == INT_MAX)  printf("(inf)\t");
                    else printf("%d\t", finalMatrix[i*n+j]);
                }
                if (finalMatrix[i*n+j] == INT_MAX)  fprintf(fp_parallel, "(inf)\t");
                else fprintf(fp_parallel, "%d\t", finalMatrix[i*n+j]);
            }
        }
        

        serialFloyd(entireMatrix, n, print, trace, finalMatrix);
        //printf("\n");
        fclose (fp_parallel);

    }
    else{
        MPI_Gather(processorMatrix, n*n/p, MPI_INT, gatherData, n*n/p, MPI_INT, 0, MPI_COMM_WORLD);
    }

    /*
    
    printf("\nBroadcasted");
    printf("\nMy rank = %d", my_rank);
    //if (my_rank != 0){
        printf("\nx=%d: y=%d\n",sRow, sCol);

        for (int i=0; i < n*n/p; i++){
            printf("%d\t", processorMatrix[i]);
        }

        printf("\n");
    //}*/


    MPI_Finalize();

    return 0;
}


int findFloydMin(int a, int b, int c){
    int dp;

    if (b == INT_MAX || c == INT_MAX){
        dp = INT_MAX;
    }
    else {
        dp = b + c;
    }

    if (a < dp){
        return a;
    }

    return dp;
}


void serialFloyd(int* original, int n, bool print, bool trace, int* comparisonMatrix){
    clock_t start_time;
    start_time = clock();
    for (int k = 0; k < n; k++){
        int *newMatrix = malloc(n*n*sizeof(int));

        for (int i=0; i< n; i++){
            for (int j =0; j<n; j++){
                if (trace)printf("\nSerial For k=%d, M[%d][%d] = min(%d, %d + %d)\n", k, i, j, original[i*n+j], original[k*n+j], original[i*n+k]);
                newMatrix[i*n+j] = findFloydMin(original[i*n+j], original[k*n+j], original[i*n+k]);
                //printf("testing test");
            }
        }
        original = newMatrix;
    }
    
    printf("\n**Time of Serial execution %f seconds", (double)(clock()-start_time)/CLOCKS_PER_SEC);
    FILE * fp_serial;
    fp_serial = fopen("s_output.txt" ,"a");
    bool identicalMatrix = true;
    if (print) printf("\nSerial Floyd result:\n");
    for (int i = 0; i < n; i++){
        if (print) printf("\n");
        for (int j=0; j <n; j++){
            if (print){
                if (original[i*n+j] == INT_MAX) printf("inf\t");
                else printf("%d\t", original[i*n+j]);
            }
            if (original[i*n+j] == INT_MAX) fprintf(fp_serial, "inf\t");
            else fprintf(fp_serial, "%d\t", original[i*n+j]);
            if (original[i*n+j] != comparisonMatrix[i*n+j]) identicalMatrix = false;
        }
    }
    fclose(fp_serial);

    if (identicalMatrix) printf("\nSerial Matrix is identical to Parallel Matrix\n");
    else printf("\nWarning: Serial Matrix is not identical to Parallel Matrix\n");


}
