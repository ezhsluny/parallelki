#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <omp.h>

/* matrix_vector_product_omp: Compute matrix-vector product c[m] = a[m][n] * b[n] */
void matrix_vector_product_omp(double *a, double *b, double *c, int m, int n)
{
    #pragma omp parallel for
        for (int i = 0; i < m; i++) {
            double sum = 0.0;
            for (int j = 0; j < n; j++) {
                sum += a[i * n + j] * b[j];}
            c[i] = sum;
        }
}

void matrix_vector_product(double *a, double *b, double *c, int m, int n)
{
    for (int i = 0; i < m; i++) {
        c[i] = 0.0;
        for (int j = 0; j < n; j++)
            c[i] += a[i * n + j] * b[j];
    }
}

double run_serial(int m, int n)
{
    double *a, *b, *c;
    a = malloc(sizeof(*a) * m * n);
    b = malloc(sizeof(*b) * n);
    c = malloc(sizeof(*c) * m);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++)
            a[i * n + j] = i + j;
    }

    for (int j = 0; j < n; j++)
        b[j] = j;

    double t = omp_get_wtime();
    matrix_vector_product(a, b, c, m, n);
    t = omp_get_wtime() - t;

    // printf("Elapsed time (serial): %.6f sec.\n", t);
    free(a);
    free(b);
    free(c);

    return t;
}

double run_parallel(int m, int n)
{
    double *a, *b, *c;
    a = malloc(sizeof(*a) * m * n);
    b = malloc(sizeof(*b) * n);
    c = malloc(sizeof(*c) * m);

    #pragma omp parallel
    {
        int threadid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        int items_per_thread = m / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? m : lb + items_per_thread;

        for (int i = lb; i < ub; i++) {
            for (int j = 0; j < n; j++) {
                a[i * n + j] = i + j;
            }
            c[i] = 0.0;
        }
    }

    #pragma omp parallel for
    for (int j = 0; j < n; j++)
        b[j] = j;

    double t = omp_get_wtime();
    matrix_vector_product_omp(a, b, c, m, n);
    t = omp_get_wtime() - t;

    // printf("Elapsed time (parallel): %.6f sec.\n", t);
    free(a);
    free(b);
    free(c);

    return t;
}

int main(int argc, char **argv)
{
    int m = 2000, n = 2000;
    printf("Matrix-vector product (c[m] = a[m, n] * b[n]; m = %d, n = %d)\n", m, n);
    printf("Memory used: %" PRIu64 " MiB\n", ((m * n + m + n) * sizeof(double)) >> 20);

    double time_serial;
    double time_parallel;

    omp_set_num_threads(40);

    for (int i = 0; i < 10; i++)
    {
        time_serial += run_serial(m, n);
        time_parallel += run_parallel(m, n);
    }

    printf("Elapsed time (parallel): %.6f sec.\n", time_parallel/10);
    printf("Elapsed time (serial): %.6f sec.\n", time_serial/10);


    return 0;
}