
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define N 4

typedef struct {
    unsigned rows;
    unsigned columns;
    double *m;
} matrix_t;

__global__ void matrix_sum_kernel(matrix_t m1, matrix_t m2, matrix_t res)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < m1.rows * m1.columns)
    {
        res.m[idx] = m1.m[idx] + m2.m[idx];
    }
}

void matrix_sum(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert ( (m1->columns == m2->columns)   &&
             (m1->columns == res->columns)  &&
             (m1->rows == m2->rows)         &&
             (m1->rows == res->rows));
    dim3 dimBlock(16, 16);
    dim3 dimGrid((m1->columns + dimBlock.x - 1) / dimBlock.x, (m1->rows + dimBlock.y - 1) / dimBlock.y);
    matrix_sum_kernel<<<dimGrid, dimBlock>>>(*m1, *m2, *res);
    cudaDeviceSynchronize();
}

matrix_t * alloc_matrix_gpu(unsigned rows, unsigned columns)
{
    matrix_t * res;
    cudaMalloc(&res, sizeof(matrix_t));
    cudaMalloc(&(res->m), columns * rows * sizeof(double));
    cudaMemcpy(&res->rows, &rows, sizeof(unsigned), cudaMemcpyHostToDevice);
    cudaMemcpy(&res->columns, &columns, sizeof(unsigned), cudaMemcpyHostToDevice);
    return res;
}

void destroy_matrix(matrix_t *m)
{
    cudaFree(m->m);
    cudaFree(m);
}

int main()
{
    matrix_t *A1 = alloc_matrix(N, N);
    matrix_t *A2 = alloc_matrix(N, N);
    matrix_t *res = alloc_matrix(N, N);

    double h_A1[N][N] = { { 1, 2, 3, 4 },
                         { 5, 6, 7, 8 },
                         { 9, 10, 11, 12 },
                         { 13, 14, 15, 16 } };
    double h_A2[N][N] = { { 16, 15, 14, 13 },
                         { 12, 11, 10, 9 },
                         { 8, 7, 6, 5 },
                         { 4, 3, 2, 1 } };

    cudaMemcpy(A1->m, h_A1, N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(A2->m, h_A2, N * N * sizeof(double), cudaMemcpyHostToDevice);

    matrix_sum(A1, A2, res);

    double h_res[N][N];

    cudaMemcpy(h_res, res->m, N * N * sizeof(double), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%lf ", h_res[i][j]);
        }
        printf("\n");
    }

    destroy_matrix_gpu(A1);
    destroy_matrix_gpu(A2);
    destroy_matrix_gpu(res);

    return 0;
}
