#ifndef MATRIX_H
#define MATRIX_H
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <stdbool.h>

typedef struct
{
    double * m;
    unsigned columns;
    unsigned rows;
}  matrix_t;

matrix_t * alloc_matrix(unsigned rows, unsigned columns);

void destroy_matrix(matrix_t *m);

void print_matrix(matrix_t *m, bool is_short);

void hadamard_product(matrix_t *m1, matrix_t *m2, matrix_t *res);

void matrix_sum(matrix_t *m1, matrix_t *m2, matrix_t *res);

void matrix_minus(matrix_t *m1, matrix_t *m2, matrix_t *res);

void matrix_dot(matrix_t *m1, matrix_t *m2, matrix_t *res);

void matrix_function(matrix_t *m1, double (*f)(double), matrix_t *res);

void matrix_transpose(matrix_t *m1, matrix_t *res);

void matrix_scalar(matrix_t *m1, double s, matrix_t *res);

void matrix_memcpy(matrix_t *dest, const matrix_t *src);

void matrix_dot_gpu(const matrix_t *m1, const matrix_t *m2, matrix_t *res);

void hadamard_product_GPU(matrix_t *m1, matrix_t *m2, matrix_t *res);

void print_matrix_GPU(matrix_t *m, bool is_short);

void matrix_sum_GPU(matrix_t *m1, matrix_t *m2, matrix_t *res);

void matrix_minus_GPU(matrix_t *m1, matrix_t *m2, matrix_t *res);

void matrix_transpose_GPU(matrix_t *m1, matrix_t *res);

void matrix_scalar_GPU(matrix_t *m1,double s, matrix_t *res);

void matrix_memcpy_GPU(matrix_t *dest, const matrix_t *src);

#endif // MATRIX_H
