#include "matrix.h"
#include <stdlib.h>
#include <string.h>

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

matrix_t * alloc_matrix(unsigned rows, unsigned columns)
{
    matrix_t * res = (matrix_t*) malloc( sizeof(matrix_t) );
    res->m = (double *) calloc(columns * rows, sizeof(double));
    res->columns = columns;
    res->rows = rows;
    return res;
}

void destroy_matrix(matrix_t *m)
{
    free(m->m);
    free(m);
}

matrix_t *alloc_matrix_GPU(unsigned rows, unsigned columns)
{
    matrix_t *res = (matrix_t *)malloc(sizeof(matrix_t));

    double *m;
    cudaMalloc((double **)&m, columns * rows * sizeof(double));

    res->m = m;
    res->columns = columns;
    res->rows = rows;
    return res;
}

void destroy_matrix_GPU(matrix_t *m)
{
    cudaFree(m->m);
    free(m);
}

/*
__global__ 
void computeMatrixMulGPU
(
   double *A, double *B, double *C,
   int numARows, int numAColumns,
   int numBRows, int numBColumns
)
{
   int row = blockIdx.y * blockDim.y + threadIdx.y;
   int col = blockIdx.x * blockDim.x + threadIdx.x;

   if(row < numARows && col < numBColumns) {
       double sum = 0.0f;
       for (int i = 0; i < numAColumns; ++i) {
           sum += A[row * numAColumns + i] * B[i * numBColumns + col];
       }
       C[row * numBColumns + col] = sum;
   }
}
*/

__global__ 
void computeMatrixMulGPU(
   double *A, double *B, double *C,
   int numARows, int numAColumns,
   int numBRows, int numBColumns)
{
   const int BLOCK_SIZE = 16; // Adjust as needed
   __shared__ double sharedA[BLOCK_SIZE][BLOCK_SIZE];
   __shared__ double sharedB[BLOCK_SIZE][BLOCK_SIZE];

   int bx = blockIdx.x, by = blockIdx.y;
   int tx = threadIdx.x, ty = threadIdx.y;
   int row = by * BLOCK_SIZE + ty;
   int col = bx * BLOCK_SIZE + tx;
   double sum = 0.0;

   for (int m = 0; m < (numAColumns + BLOCK_SIZE - 1) / BLOCK_SIZE; ++m) {
      if (m * BLOCK_SIZE + tx < numAColumns && row < numARows)
         sharedA[ty][tx] = A[row * numAColumns + m * BLOCK_SIZE + tx];
      else
         sharedA[ty][tx] = 0.0;

      if (m * BLOCK_SIZE + ty < numBRows && col < numBColumns)
         sharedB[ty][tx] = B[(m * BLOCK_SIZE + ty) * numBColumns + col];
      else
         sharedB[ty][tx] = 0.0;

      __syncthreads();

      for (int k = 0; k < BLOCK_SIZE; ++k)
         sum += sharedA[ty][k] * sharedB[k][tx];

      __syncthreads();
   }

   if (row < numARows && col < numBColumns)
      C[row * numBColumns + col] = sum;
}


void matrix_dot(const matrix_t *m1, const matrix_t *m2, matrix_t *res) {
    
    
 
   dim3 blockDim(16, 16);
   dim3 gridDim(ceil(((double)m2->columns) / blockDim.x), ceil(((double)m1->rows) / blockDim.y));


    computeMatrixMulGPU<<<gridDim, blockDim>>>(m1->m,m2->m, res->m,m1->rows, m1->columns, m2->rows, m2->columns);
  
}


__global__ void hadamard_product_kernel(
   double *A, double *B, double *C,
   int numARows, int numAColumns,
   int numBRows, int numBColumns
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numARows * numAColumns)
    {
        C[idx] = A[idx] * B[idx];
    }
}

void hadamard_product(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert ( (m1->columns == m2->columns)   &&
             (m1->columns == res->columns)  &&
             (m1->rows == m2->rows)         &&
             (m1->rows == res->rows));


    int threadsPerBlock = 256; 
    int blocksPerGrid = (m1->rows * m1->columns + threadsPerBlock - 1) / threadsPerBlock; 
    hadamard_product_kernel<<<blocksPerGrid,threadsPerBlock>>>(m1->m,m2->m, res->m, m1->rows, m1->columns, m2->rows, m2->columns);
   


}


void print_matrix(matrix_t *m, bool is_short) {
    
    double *h_matrix = (double *)malloc(m->rows * m->columns * sizeof(double));
    cudaMemcpy(h_matrix, m->m, m->rows * m->columns * sizeof(double), cudaMemcpyDeviceToHost);

    unsigned lim_rows = is_short ? MIN(m->rows, 4) : m->rows;
    unsigned lim_col = is_short ? MIN(m->columns, 10) : m->columns;

    for (unsigned int i = 0; i < lim_rows; ++i) {
        for (unsigned int j = 0; j < lim_col; ++j) {
            printf("%.2lf ", h_matrix[i * m->columns + j]);
        }
        if (is_short && lim_col != m->columns) printf("...");
        printf("\n");
    }
    if (is_short && lim_rows != m->rows) printf("...\n");

    free(h_matrix);
}

__global__ void matrix_sum_kernel(double *A, double *B, double *C,
   int numARows, int numAColumns,
   int numBRows, int numBColumns)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numARows * numAColumns)
    {
        C[idx] = A[idx] + B[idx];
    }
}

void matrix_sum(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert ( (m1->columns == m2->columns)   &&
             (m1->columns == res->columns)  &&
             (m1->rows == m2->rows)         &&
             (m1->rows == res->rows));
    
    
    int threadsPerBlock = 256; 
    int blocksPerGrid = (m1->rows * m1->columns + threadsPerBlock - 1) / threadsPerBlock; 
    matrix_sum_kernel<<<blocksPerGrid,threadsPerBlock>>>(m1->m,m2->m, res->m, m1->rows, m1->columns, m2->rows, m2->columns);
   


}

__global__ void matrix_minus_kernel(double *A, double *B, double *C,
   int numARows, int numAColumns,
   int numBRows, int numBColumns)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numARows * numAColumns)
    {
        C[idx] = A[idx] - B[idx];
    }
}

void matrix_minus(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert ( (m1->columns == m2->columns)   &&
             (m1->columns == res->columns)  &&
             (m1->rows == m2->rows)         &&
             (m1->rows == res->rows));
    
    int threadsPerBlock = 256; 
    int blocksPerGrid = (m1->rows * m1->columns + threadsPerBlock - 1) / threadsPerBlock; 
    matrix_minus_kernel<<<blocksPerGrid,threadsPerBlock>>>(m1->m,m2->m, res->m, m1->rows, m1->columns, m2->rows, m2->columns);
   


}


__global__ void matrix_transpose_kernel(double *A, double *C,
   int numARows, int numAColumns)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < numARows && col < numAColumns)
    {
        C[row + col * numARows] = A[col + row * numAColumns];
    }
}


void matrix_transpose(matrix_t *m1, matrix_t *res)
{


    dim3 dimBlock(16, 16);

    
    dim3 dimGrid((m1->columns + dimBlock.x - 1) / dimBlock.x,
                 (m1->rows + dimBlock.y - 1) / dimBlock.y);   

    matrix_transpose_kernel<<<dimGrid,dimBlock>>>(m1->m, res->m, m1->rows, m1->columns);
    
    


}

__global__ void matrix_scalar_kernel(double *A, double *C,
   int numARows, int numAColumn,float s)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numARows * numAColumn)
    {
        C[idx] = A[idx] * s;
    }
}

void matrix_scalar(matrix_t *m1,double s, matrix_t *res)
{



    int threadsPerBlock = 256; 


    int blocksPerGrid = (m1->rows * m1->columns + threadsPerBlock - 1) / threadsPerBlock;  

    matrix_scalar_kernel<<<blocksPerGrid,threadsPerBlock>>>(m1->m,res->m, m1->rows, m1->columns,s);
   
    
    

}

__global__ void matrix_function_kernel(double *A, double *B, bool deriv, int numRows, int numColumns)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < numRows && col < numColumns)
    {
        double x = A[row * numColumns + col];
        double sig = 1 / (1 + exp(-x));
        if (deriv)
        {
            sig = sig * (1 - sig);
        }
        B[row * numColumns + col] = sig;
    }
}

void matrix_function(matrix_t *d_m, bool deriv, matrix_t *d_res)
{
    assert((d_m->columns == d_res->columns) &&
           (d_m->rows == d_res->rows));


    dim3 threadsPerBlock(16, 16);

    dim3 blocksPerGrid((d_m->columns + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (d_m->rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_function_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_m->m, d_res->m, deriv, d_res->rows, d_res->columns);
    
    
}

void matrix_memcpy(matrix_t *dest, const matrix_t *src)
{
    assert((dest->rows == src->rows) &&
           (dest->columns == src->columns));

    memcpy(dest->m, src->m, src->columns * src->rows * sizeof(double));
}

void ones(matrix_t *d_m)
{
    matrix_t *h_m = alloc_matrix(d_m->rows, d_m->columns);
    
    for (int idx = 0; idx < d_m->columns * d_m->rows; idx++)
    {
        h_m->m[idx] = 1.0f;
    }
    
    cudaMemcpy(d_m->m, h_m->m,h_m->columns * h_m->rows * sizeof(double), cudaMemcpyHostToDevice);
    
}

void fill_matrix(matrix_t *m) {
    
    int size = m->rows * m->columns;
    double *temp = (double *)malloc(size * sizeof(double));

    for (int i = 0; i < size; ++i) {
        temp[i] = (double)i; 
    }

    
    cudaMemcpy(m->m, temp, size * sizeof(double), cudaMemcpyHostToDevice);
    
    
    free(temp);
}

/*
int main() {
    
    const unsigned int rows = 4, columns = 4;

    // Create matrices
    matrix_t *m1 = alloc_matrix_GPU(rows, columns);
    matrix_t *m2 = alloc_matrix_GPU(rows, columns);
    matrix_t *result = alloc_matrix_GPU(rows, columns);

    // Initialize matrices with some values
    fill_matrix(m1);
    cudaDeviceSynchronize();
    fill_matrix(m2);
    cudaDeviceSynchronize();

    // Print initial matrices
    printf("Matrix m1:\n");
    print_matrix(m1, false);
    cudaDeviceSynchronize();
    printf("\nMatrix m2:\n");
    print_matrix(m2, false);
    cudaDeviceSynchronize();

    // Test matrix_dot
    matrix_dot(m1, m2, result);
    cudaDeviceSynchronize();
    printf("\nmatrix_dot result:\n");
    print_matrix(result, false);  
    cudaDeviceSynchronize();

    // Test hadamard_product
    hadamard_product(m1, m2, result);
    cudaDeviceSynchronize();
    printf("\nhadamard_product result:\n");
    print_matrix(result, false);
    cudaDeviceSynchronize();

    // Test matrix_sum
    matrix_sum(m1, m2, result);
    cudaDeviceSynchronize();
    printf("\nmatrix_sum result:\n");
    print_matrix(result, false);
    cudaDeviceSynchronize();

    // Test matrix_minus
    matrix_minus(m1, m2, result);
    cudaDeviceSynchronize();
    printf("\nmatrix_minus result:\n");
    print_matrix(result, false);
    cudaDeviceSynchronize();

    // Test matrix_transpose
    matrix_transpose(m1, result);
    cudaDeviceSynchronize();
    printf("\nmatrix_transpose result:\n");
    print_matrix(result, false);
    cudaDeviceSynchronize();

    // Test matrix_scalar
    double scalar = 2.0;
    matrix_scalar(m1, scalar, result);
    cudaDeviceSynchronize();
    printf("\nmatrix_scalar result:\n");
    print_matrix(result, false);
    cudaDeviceSynchronize();

    // Test matrix_function (using sigmoid and its derivative as an example)
    // Implement sigmoid and sigmoid_derivative functions
    matrix_function(m1, false, result);  // Sigmoid
    cudaDeviceSynchronize();
    printf("\nmatrix_function (sigmoid) result:\n");
    print_matrix(result, false);
    cudaDeviceSynchronize();

    matrix_function(m1, true, result);  // Sigmoid derivative
    cudaDeviceSynchronize();
    printf("\nmatrix_function (sigmoid derivative) result:\n");
    print_matrix(result, false);
    cudaDeviceSynchronize();

    // Cleanup
    destroy_matrix_GPU(m1);
    destroy_matrix_GPU(m2);
    destroy_matrix_GPU(result);

    return 0;
}

*/

