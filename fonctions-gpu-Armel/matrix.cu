#include "matrix.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <iostream>

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

matrix_t * alloc_matrix_gpu(unsigned rows, unsigned columns)
{
    matrix_t * res;
    cudaMalloc(&res, sizeof(matrix_t));
    cudaMalloc(&(res->m), columns * rows * sizeof(double));
    cudaMemcpy(res, &(matrix_t){.rows = rows, .columns = columns}, sizeof(matrix_t), cudaMemcpyHostToDevice);
    return res;
}

void destroy_matrix_gpu(matrix_t *m)
{
    cudaFree(m->m);
    cudaFree(m);
}

__global__ void print_matrix_kernel(matrix_t m, bool is_short)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m.rows && col < m.columns)
    {
        printf("%.2lf ", m.m[col + row * m.columns]);
    }
}

void print_matrix(matrix_t *m, bool is_short)
{
    unsigned lim_rows = 0;
    unsigned lim_col = 0;

    if (is_short)
    {
        lim_rows = MIN(m->rows, 4);
        lim_col = MIN(m->columns, 10);
    }
    else
    {
        lim_rows = m->rows;
        lim_col = m->columns;
    }

    dim3 dimBlock(10, 4);
    dim3 dimGrid((lim_col + dimBlock.x - 1) / dimBlock.x, (lim_rows + dimBlock.y - 1) / dimBlock.y);
    print_matrix_kernel<<<dimGrid, dimBlock>>>(*m, is_short);
    cudaDeviceSynchronize();
    printf("\n");
}

__global__ void hadamard_product_kernel(matrix_t m1, matrix_t m2, matrix_t res)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < m1.rows * m1.columns)
    {
        res.m[idx] = m1.m[idx] * m2.m[idx];
    }
}

void hadamard_product(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert ( (m1->columns == m2->columns)   &&
             (m1->columns == res->columns)  &&
             (m1->rows == m2->rows)         &&
             (m1->rows == res->rows));
    dim3 dimBlock(16, 16);
    dim3 dimGrid((m1->columns + dimBlock.x - 1) / dimBlock.x, (m1->rows + dimBlock.y - 1) / dimBlock.y);
    hadamard_product_kernel<<<dimGrid, dimBlock>>>(*m1, *m2, *res);
    cudaDeviceSynchronize();
}

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

__global__ void matrix_minus_kernel(matrix_t m1, matrix_t m2, matrix_t res)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < m1.rows * m1.columns)
    {
        res.m[idx] = m1.m[idx] - m2.m[idx];
    }
}

void matrix_minus(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert ( (m1->columns == m2->columns)   &&
             (m1->columns == res->columns)  &&
             (m1->rows == m2->rows)         &&
             (m1->rows == res->rows));
    dim3 dimBlock(16, 16);
    dim3 dimGrid((m1->columns + dimBlock.x - 1) / dimBlock.x, (m1->rows + dimBlock.y - 1) / dimBlock.y);
    matrix_minus_kernel<<<dimGrid, dimBlock>>>(*m1, *m2, *res);
    cudaDeviceSynchronize();
}

__global__ void matrix_dot_kernel(matrix_t m1, matrix_t m2, matrix_t res)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < res.rows && col < res.columns)
    {
        double sum = 0;
        for (int k = 0; k < m1.columns; k++)
        {
            sum += m1.m[k + row * m1.columns] * m2.m[col + k * m2.columns];
        }
        res.m[col + row * res.columns] = sum;
    }
}

void matrix_dot(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert ( (m1->columns == m2->rows)  &&
             (m1->rows == res->rows)    &&
             (m2->columns == res->columns));
    dim3 dimBlock(16, 16);
    dim3 dimGrid((res->columns + dimBlock.x - 1) / dimBlock.x, (res->rows + dimBlock.y - 1) / dimBlock.y);
    matrix_dot_kernel<<<dimGrid, dimBlock>>>(*m1, *m2, *res);
    cudaDeviceSynchronize();
}

__global__ void matrix_function_kernel(matrix_t m1, double (*f)(double), matrix_t res)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < m1.rows * m1.columns)
    {
        res.m[idx] = f(m1.m[idx]);
    }
}

void matrix_function(matrix_t *m1, double (*f)(double), matrix_t *res)
{
    assert ( (m1->columns == res->columns) &&             
             (m1->rows == res->rows));
    dim3 dimBlock(16, 16);
    dim3 dimGrid((m1->columns + dimBlock.x - 1) / dimBlock.x, (m1->rows + dimBlock.y - 1) / dimBlock.y);
    matrix_function_kernel<<<dimGrid, dimBlock>>>(*m1, f, *res);
    cudaDeviceSynchronize();
}

__global__ void matrix_transpose_kernel(matrix_t m1, matrix_t res)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m1.rows && col < m1.columns)
    {
        res.m[row + col * m1.rows] = m1.m[col + row * m1.columns];
    }
}

void matrix_transpose(matrix_t *m1, matrix_t *res)
{
    assert ( (m1->columns == res->rows) &&             
             (m1->rows == res->columns));
    dim3 dimBlock(16, 16);
    dim3 dimGrid((m1->columns + dimBlock.x - 1) / dimBlock.x, (m1->rows + dimBlock.y - 1) / dimBlock.y);
    matrix_transpose_kernel<<<dimGrid, dimBlock>>>(*m1, *res);
    cudaDeviceSynchronize();
}

__global__ void matrix_scalar_kernel(matrix_t m1, double s, matrix_t res)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < m1.rows * m1.columns)
    {
        res.m[idx] = m1.m[idx] * s;
    }
}


void matrix_scalar(matrix_t *m1, double s, matrix_t *res)
{
    assert ( (m1->rows == res->rows) &&             
             (m1->columns == res->columns));
    dim3 dimBlock(16, 16);
    dim3 dimGrid((m1->columns + dimBlock.x - 1) / dimBlock.x, (m1->rows + dimBlock.y - 1) / dimBlock.y);
    matrix_scalar_kernel<<<dimGrid, dimBlock>>>(*m1, s, *res);
    cudaDeviceSynchronize();
}

void matrix_memcpy(matrix_t *dest, const matrix_t *src)
{
    assert ( (dest->rows == src->rows)      &&             
             (dest->columns == src->columns));
             
   cudaMemcpy(dest->m, src->m, src->columns * src->rows * sizeof(double), cudaMemcpyDeviceToDevice);    
}