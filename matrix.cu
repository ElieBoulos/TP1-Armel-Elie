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
    //printf("free %p %p\n", m, m->m);
    free(m->m);
    free(m);
}

void print_matrix(matrix_t *m, bool is_short){
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

    for (int row = 0; row < lim_rows; row ++)
    {
        for (int col = 0; col < lim_col; col ++)
        {
            printf("%.2lf ", m->m[col + row * m->columns]);
        }
        if (is_short && lim_col != m->columns) printf("...");
        printf("\n");
    }
    if (is_short && lim_rows != m->rows) printf("...\n");
}

void hadamard_product(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert ( (m1->columns == m2->columns)   &&
             (m1->columns == res->columns)  &&
             (m1->rows == m2->rows)         &&
             (m1->rows == res->rows));

    for (int idx = 0; idx < m1->rows * m1->columns; idx ++)
    {
            res->m[idx] = m1->m[idx] * m2->m[idx];
    }
}

void matrix_sum(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert ( (m1->columns == m2->columns)  &&
             (m1->columns == res->columns) &&
             (m1->rows == m2->rows)        &&
             (m1->rows == res->rows));

    for (int idx = 0; idx < m1->rows * m1->columns; idx ++)
    { 
        res->m[idx] = m1->m[idx] + m2->m[idx];
    }
}

void matrix_minus(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert ( (m1->columns == m2->columns)  &&
             (m1->columns == res->columns) &&
             (m1->rows == m2->rows)        &&
             (m1->rows == res->rows));
             
    for (int idx = 0; idx < m1->rows * m1->columns; idx ++)
    {
        res->m[idx] = m1->m[idx] - m2->m[idx];
    }
}

void matrix_dot(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert ( (m1->columns == m2->rows)  &&
             (m1->rows == res->rows)    &&
             (m2->columns == res->columns));

    for (int row = 0; row < m1->rows; row ++)
    {
        for (int col = 0; col < m2->columns; col ++)
        {
            int idx = col + row * m2->columns;
            double var = 0.0;

            for (int ii = 0; ii < m1->columns; ii++)
            {
                var += m1->m[ii + row * m1->columns] * m2->m[col + ii * m2->columns];
            }

            res->m[idx] = var;
        }
    }
}




void matrix_function(matrix_t *m1, double (*f)(double), matrix_t *res)
{
    assert ( (m1->columns == res->columns) &&             
             (m1->rows == res->rows));

    for (int idx = 0; idx < m1->rows * m1->columns; idx ++)
    {
        res->m[idx] = f(m1->m[idx]);
    }
}

void matrix_transpose(matrix_t *m1, matrix_t *res)
{
    assert ( (m1->columns == res->rows) &&             
             (m1->rows == res->columns));
    
    for (int row = 0; row < m1->rows; row++)
    {
        for (int col = 0; col < m1->columns; col ++)
        {
            res->m[row + col * m1->rows] = m1->m[col + row * m1->columns];
        }
    }
}

void matrix_scalar(matrix_t *m1, double s, matrix_t *res)
{
    assert ( (m1->rows == res->rows) &&             
             (m1->columns == res->columns));

    for (int idx = 0; idx < m1->columns*m1->rows; idx ++)
    {
        res->m[idx] = m1->m[idx] * s;
    }
}

void matrix_memcpy(matrix_t *dest, const matrix_t *src)
{
    assert ( (dest->rows == src->rows)      &&             
             (dest->columns == src->columns));

    memcpy(dest->m, src->m, src->columns * src->rows * sizeof(double));     
}


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



void matrix_dot_gpu(const matrix_t *m1, const matrix_t *m2, matrix_t *res) {
    
    

    size_t size_m1 = m1->rows * m1->columns * sizeof(double);
    size_t size_m2 = m2->rows * m2->columns * sizeof(double);
    size_t size_res = res->rows * res->columns * sizeof(double);

   double *deviceA;
   double *deviceB;
   double *deviceC;

   
   cudaMalloc((void **)&deviceA, size_m1);
   cudaMalloc((void **)&deviceB, size_m2);
   cudaMalloc((void **)&deviceC, size_res);
    
    cudaMemset(deviceC, 0, size_res); 
 
    
    cudaMemcpy(deviceA, m1->m, size_m1, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, m2->m, size_m2, cudaMemcpyHostToDevice);
    
 
   dim3 blockDim(16, 16);
   dim3 gridDim(ceil(((double)m2->columns) / blockDim.x), ceil(((double)m1->rows) / blockDim.y));


    computeMatrixMulGPU<<<gridDim, blockDim>>>(deviceA,deviceB, deviceC,m1->rows, m1->columns, m2->rows, m2->columns);
    cudaDeviceSynchronize();

    
    cudaMemcpy(res->m, deviceC, size_res, cudaMemcpyDeviceToHost);
    
                                                   
    
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
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

void hadamard_product_GPU(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert ( (m1->columns == m2->columns)   &&
             (m1->columns == res->columns)  &&
             (m1->rows == m2->rows)         &&
             (m1->rows == res->rows));
    size_t size_m1 = m1->rows * m1->columns * sizeof(double);
    size_t size_m2 = m2->rows * m2->columns * sizeof(double);
    size_t size_res = res->rows * res->columns * sizeof(double);

   double *deviceA;
   double *deviceB;
   double *deviceC;

   
   cudaMalloc((void **)&deviceA, size_m1);
   cudaMalloc((void **)&deviceB, size_m2);
   cudaMalloc((void **)&deviceC, size_res);
    
    cudaMemset(deviceC, 0, size_res); 
 
    
    cudaMemcpy(deviceA, m1->m, size_m1, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, m2->m, size_m2, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256; 
    int blocksPerGrid = (m1->rows * m1->columns + threadsPerBlock - 1) / threadsPerBlock; 
    hadamard_product_kernel<<<blocksPerGrid,threadsPerBlock>>>(deviceA, deviceB, deviceC, m1->rows, m1->columns, m2->rows, m2->columns);
    cudaDeviceSynchronize();

    
    cudaMemcpy(res->m, deviceC, size_res, cudaMemcpyDeviceToHost);
    
                                                   
    
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);


}

__global__ void print_matrix_kernel( double *A,
   int numARows, int numAColumns, bool is_short)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < numARows && col < numAColumns)
    {
        printf("%.2lf ", A[col + row * numAColumns]);
    }
}

void print_matrix_GPU(matrix_t *m, bool is_short)
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
    
    double *deviceA;

    size_t size_m1 = m->rows * m->columns * sizeof(double);
   
    cudaMalloc((void **)&deviceA, size_m1);
    
    cudaMemcpy(deviceA, m->m, size_m1, cudaMemcpyHostToDevice);

    dim3 dimBlock(16, 16);

    dim3 dimGrid((lim_col + dimBlock.x - 1) / dimBlock.x, 
                 (lim_rows + dimBlock.y - 1) / dimBlock.y);

    print_matrix_kernel<<<dimGrid, dimBlock>>>(deviceA,m->rows,m->columns, is_short);
    cudaDeviceSynchronize();
    printf("\n");
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

void matrix_sum_GPU(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert ( (m1->columns == m2->columns)   &&
             (m1->columns == res->columns)  &&
             (m1->rows == m2->rows)         &&
             (m1->rows == res->rows));
    
    size_t size_m1 = m1->rows * m1->columns * sizeof(double);
    size_t size_m2 = m2->rows * m2->columns * sizeof(double);
    size_t size_res = res->rows * res->columns * sizeof(double);

    double *deviceA;
    double *deviceB;
    double *deviceC;

   
   cudaMalloc((void **)&deviceA, size_m1);
   cudaMalloc((void **)&deviceB, size_m2);
   cudaMalloc((void **)&deviceC, size_res);
    
    cudaMemset(deviceC, 0, size_res); 
 
    
    cudaMemcpy(deviceA, m1->m, size_m1, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, m2->m, size_m2, cudaMemcpyHostToDevice);
    
    int threadsPerBlock = 256; 
    int blocksPerGrid = (m1->rows * m1->columns + threadsPerBlock - 1) / threadsPerBlock; 
    matrix_sum_kernel<<<blocksPerGrid,threadsPerBlock>>>(deviceA, deviceB, deviceC, m1->rows, m1->columns, m2->rows, m2->columns);

    cudaDeviceSynchronize();
    
    cudaMemcpy(res->m, deviceC, size_res, cudaMemcpyDeviceToHost);
    
                                                   
    
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);

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

void matrix_minus_GPU(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert ( (m1->columns == m2->columns)   &&
             (m1->columns == res->columns)  &&
             (m1->rows == m2->rows)         &&
             (m1->rows == res->rows));
    
    size_t size_m1 = m1->rows * m1->columns * sizeof(double);
    size_t size_m2 = m2->rows * m2->columns * sizeof(double);
    size_t size_res = res->rows * res->columns * sizeof(double);

    double *deviceA;
    double *deviceB;
    double *deviceC;

   
   cudaMalloc((void **)&deviceA, size_m1);
   cudaMalloc((void **)&deviceB, size_m2);
   cudaMalloc((void **)&deviceC, size_res);
    
    cudaMemset(deviceC, 0, size_res); 
 
    
    cudaMemcpy(deviceA, m1->m, size_m1, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, m2->m, size_m2, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256; 
    int blocksPerGrid = (m1->rows * m1->columns + threadsPerBlock - 1) / threadsPerBlock; 
    matrix_minus_kernel<<<blocksPerGrid,threadsPerBlock>>>(deviceA, deviceB, deviceC, m1->rows, m1->columns, m2->rows, m2->columns);

    
    cudaMemcpy(res->m, deviceC, size_res, cudaMemcpyDeviceToHost);
    
                                                   
    
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);

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


void matrix_transpose_GPU(matrix_t *m1, matrix_t *res)
{


    size_t size_m1 = m1->rows * m1->columns * sizeof(double);
    size_t size_res = res->rows * res->columns * sizeof(double);

    double *deviceA;
    double *deviceC;

   
   cudaMalloc((void **)&deviceA, size_m1);
   cudaMalloc((void **)&deviceC, size_res);
    
 
    
    cudaMemcpy(deviceA, m1->m, size_m1, cudaMemcpyHostToDevice);

    dim3 dimBlock(16, 16);

    
    dim3 dimGrid((m1->columns + dimBlock.x - 1) / dimBlock.x,
                 (m1->rows + dimBlock.y - 1) / dimBlock.y);   

    matrix_transpose_kernel<<<dimGrid,dimBlock>>>(deviceA, deviceC, m1->rows, m1->columns);
    cudaDeviceSynchronize();
    
    cudaMemcpy(res->m, deviceC, size_res, cudaMemcpyDeviceToHost);
    
                                                   
    
    cudaFree(deviceA);
    cudaFree(deviceC);

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

void matrix_scalar_GPU(matrix_t *m1,double s, matrix_t *res)
{


    size_t size_m1 = m1->rows * m1->columns * sizeof(double);
    size_t size_res = res->rows * res->columns * sizeof(double);

    double *deviceA;
    double *deviceC;

   
   cudaMalloc((void **)&deviceA, size_m1);
   cudaMalloc((void **)&deviceC, size_res);
    
 
    
    cudaMemcpy(deviceA, m1->m, size_m1, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256; 


    int blocksPerGrid = (m1->rows * m1->columns + threadsPerBlock - 1) / threadsPerBlock;  

    matrix_scalar_kernel<<<blocksPerGrid,threadsPerBlock>>>(deviceA, deviceC, m1->rows, m1->columns,s);
    cudaDeviceSynchronize();
    
    cudaMemcpy(res->m, deviceC, size_res, cudaMemcpyDeviceToHost);
    
                                                   
    
    cudaFree(deviceA);
    cudaFree(deviceC);

}

//final version



