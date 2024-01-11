#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"
#include <cuda_runtime.h>

matrix_t * alloc_matrix_cpu(unsigned rows, unsigned columns);

int main()
{
    unsigned rows = 2;
    unsigned columns = 3;

    // Allocate the matrix on the CPU
    matrix_t * A = alloc_matrix_cpu(rows, columns);

    // Fill the matrix with random data
    for (unsigned i = 0; i < A->rows; i++) {
        for (unsigned j = 0; j < A->columns; j++) {
            A->m[i * columns + j] = (double) rand() / RAND_MAX;
        }
    }

    // Allocate the matrix on the GPU
    matrix_t * d_A;
    cudaMalloc((void**) &d_A, sizeof(matrix_t));

    double * d_m;
    cudaMalloc((void**) &d_m, rows * columns * sizeof(double));

    // Copy the matrix from the CPU to the GPU
    cudaMemcpy(d_m, A->m, rows * columns * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_A->m), &d_m, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_A->rows), &rows, sizeof(unsigned), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_A->columns), &columns, sizeof(unsigned), cudaMemcpyHostToDevice);

    // Copy the matrix from the GPU to the CPU
    double * h_A = (double *) malloc(rows * columns * sizeof(double));
    cudaMemcpy(h_A, d_m, rows * columns * sizeof(double), cudaMemcpyDeviceToHost);

    // Print the matrix
    printf("Matrix A:\n");
    for (unsigned i = 0; i < rows; i++) {
        for (unsigned j = 0; j < columns; j++) {
            printf("%.2f ", h_A[i * columns + j]);
        }
        printf("\n");
    }

    // Clean up
    free(A->m);
    free(A);
    free(h_A);
    cudaFree(d_m);
    cudaFree(d_A);

    return 0;
}
   
matrix_t * alloc_matrix_cpu(unsigned rows, unsigned columns)
{
    matrix_t * res = (matrix_t*) malloc( sizeof(matrix_t) );
    res->m = (double *) calloc(columns * rows, sizeof(double));
    res->columns = columns;
    res->rows = rows;
    return res;
}
