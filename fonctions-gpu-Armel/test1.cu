#include "ann.h"
#include "matrix.h"


#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))



// allocate the memory on the CPU
matrix_t * alloc_matrix_cpu(unsigned rows, unsigned columns)
{
    matrix_t * res = (matrix_t*) malloc( sizeof(matrix_t) );
    res->m = (double *) calloc(columns * rows, sizeof(double));
    res->columns = columns;
    res->rows = rows;
    return res;
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

__global__ void hadamard_product_kernel(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    int idx = threadIdx .x + blockIdx .x * blockDim .x;
    if (idx < m1->rows * m1->columns) 
        {
            res->m[idx] =  m1->m[idx]* m2->m[idx];
        }
}

void hadamard_product(matrix_t *A1, matrix_t *A2, matrix_t *d_res)
{
    assert((A1->columns == A2->columns) && (A1->rows == A2->rows));

    unsigned rows = A1->rows;
    unsigned columns = A2->columns;

    // Allocate the matrix on the GPU

    matrix_t *d_A1 = NULL;
    cudaMalloc((void **)&d_A1, sizeof(matrix_t));
    if (d_A1 == NULL)
    {
        printf("Error: allocation failed\n");
        return;
    }
    double *d_m1 = NULL;
    cudaMalloc((void **)&d_m1, sizeof(double) * rows * columns);
    if (d_m1 == NULL) 
    {
        printf("Error: allocation failed\n");
        return;
    }
    cudaMemcpy(d_m1, A1->m, sizeof(double) * rows * columns, cudaMemcpyHostToDevice);
    d_A1->rows = rows;
    d_A1->columns = columns;
    d_A1->m = d_m1;

    matrix_t *d_A2 = NULL;
    cudaMalloc((void **)&d_A2, sizeof(matrix_t));
    if (d_A2 == NULL) 
    {
        printf("Error: allocation failed\n");
        return;
    }
    double *d_m2 = NULL;
    cudaMalloc((void **)&d_m2, sizeof(double) * rows * columns);
    if (d_m2 == NULL)
    {  
        printf("Error: allocation failed\n");
        return;
    }
    cudaMemcpy(d_m2, A2->m, sizeof(double) * rows * columns, cudaMemcpyHostToDevice);
    d_A2->rows = rows;
    d_A2->columns = columns;
    d_A2->m = d_m2;

    const int N = A1->columns * A1->rows;
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    hadamard_product_kernel <<<blocksPerGrid, threadsPerBlock>>>(d_A1, d_A2, d_res);

    // Deallocate memory on the GPU
    cudaFree(d_A1);
    cudaFree(d_m1);
    cudaFree(d_A2);
    cudaFree(d_m2);
}

int main()
{
    matrix_t *A1_h;
    matrix_t *A2_h;
    
    A1_h = alloc_matrix_cpu(3, 3);
    A2_h = alloc_matrix_cpu(3, 3);

    for (int row = 0; row < 3; row++)
    {
        for (int col = 0; col < 3; col++)
        {
            A1_h->m[col + row * 3] = col + 1;
            A2_h->m[col + row * 3] = row;
        }
    }

    unsigned rows = A1_h->rows;
    unsigned columns = A1_h->columns;

    matrix_t *d_res;
    cudaMalloc((void**) &d_res, sizeof(matrix_t));
    cudaMalloc((void**) &d_res->m, columns * rows *sizeof(double));

    hadamard_product(A1_h, A2_h, d_res);

    cudaFree(d_res);
    cudaFree(d_res->m);
    return 0;
}