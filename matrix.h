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

matrix_t *alloc_matrix_GPU(unsigned rows, unsigned columns);

void destroy_matrix_GPU(matrix_t *m);

void print_matrix(matrix_t *m, bool is_short);

void matrix_dot(const matrix_t *m1, const matrix_t *m2, matrix_t *res);

void hadamard_product(matrix_t *m1, matrix_t *m2, matrix_t *res);

void print_matrix(matrix_t *m, bool is_short);

void matrix_sum(matrix_t *m1, matrix_t *m2, matrix_t *res);

void matrix_minus(matrix_t *m1, matrix_t *m2, matrix_t *res);

void matrix_transpose(matrix_t *m1, matrix_t *res);

void matrix_scalar(matrix_t *m1,double s, matrix_t *res);

void matrix_function(matrix_t *d_m, bool prime, matrix_t *d_res);

void matrix_memcpy(matrix_t *dest, const matrix_t *src);

void ones(matrix_t *d_m);

#endif // MATRIX_H
