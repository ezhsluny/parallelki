#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <omp.h>
#include <inttypes.h>


using namespace std;

const double EPS = 0.00001;
const int MAX_ITER = 1000;
const int N = 400;
const double T = 0.1;

int continue_iter(double* b, double* x0)
{
    double norm_b = 0.0;
    double norm_x0 = 0.0;

    #pragma omp parallel for
    for (int i = 0; i < N; i++)
    {
        norm_b += b[i] * b[i];
        norm_x0 += x0[i] * x0[i];
    }

    norm_b = sqrt(norm_b);
    norm_x0 = sqrt(norm_x0);

    return (norm_x0 / norm_b > EPS);
}


void solving_parallel(double* a, double* b, double* x)
{
    #pragma omp parallel
    {
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();
        int items_per_thread = N / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (N - 1) : (lb + items_per_thread - 1);
        
        double *x0 = (double*)malloc(N * sizeof(double));
        int iter = 0;

        do
        {
            for (int i = lb; i <= ub; i++)
            {
                double tmp = 0.0;
                for (int j = 0; j < N; j++)
                {
                    tmp += a[i * N + j] * x[j];
                }

                x0[i] = tmp - b[i];
                x[i] -= T * x0[i];
            }
            iter++;
        } while (continue_iter(b, x0) && iter < MAX_ITER);

        free(x0);
    }
}


void solving_parallel_for(double* a, double* b, double* x)
{
    #pragma omp parallel
    {
        double *x0 = (double*)malloc(N * sizeof(double));

        #pragma omp parallel for schedule(guided, 4)
        for (int iteration = 0; iteration < MAX_ITER; iteration++)
        {
            for (int i = 0; i <= N; i++)
            {
                double tmp = 0.0;
                for (int j = 0; j < N; j++)
                {
                    tmp += a[i * N + j] * x[j];
                }

                #pragma omp critical
                {
                x0[i] = tmp - b[i];
                x[i] -= T * x0[i];
                }
            }

            if (!continue_iter(b, x0))
                iteration = MAX_ITER;
        }

        free(x0);
    }
}


double run_parallel()
{
    double *a = (double*)malloc(N * N * sizeof(double));
    double *b = (double*)malloc(N * sizeof(double));
    double *x = (double*)malloc(N * sizeof(double));

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            a[i * N + j] = (i == j) ? 2.0 : 1.0;    
    }

    for (int j = 0; j < N; j++)
    {
        b[j] = N + 1;
        x[j] = 0.0;
    }

    double t = omp_get_wtime();
    solving_parallel(a, b, x);
    t = omp_get_wtime() - t;

    // cout << endl << "Solution for x:" << endl;
    // for (int i = 0; i < N; i++)
    //     cout << x[i] << " ";

    free(a);
    free(b);
    free(x);

    return t;
}

double run_parallel_for()
{
    double *a = (double*)malloc(N * N * sizeof(double));
    double *b = (double*)malloc(N * sizeof(double));
    double *x = (double*)malloc(N * sizeof(double));

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            a[i * N + j] = (i == j) ? 2.0 : 1.0;    
    }

    for (int j = 0; j < N; j++)
    {
        b[j] = N + 1;
        x[j] = 0.0;
    }

    double t = omp_get_wtime();
    solving_parallel_for(a, b, x);
    t = omp_get_wtime() - t;

    // cout << endl << "Solution for x:" << endl;
    // for (int i = 0; i < N; i++)
    //     cout << x[i] << " ";

    free(a);
    free(b);
    free(x);

    return t;
}

int main(int argc, char **argv)
{
    cout << endl << "A system of linear algebraic equations: size A = "\
                 << N << "x" << N  << endl\
                 << "size b = " << N << endl\
                 << "size x = " << N << endl;
    
    omp_set_num_threads(6);
    double tparallel = 0.0;
    double tparallel_for = 0.0;

        // tparallel += run_parallel();
        tparallel_for += run_parallel_for();
    

    cout << endl << "Execution time (parallel): " << tparallel << endl;
    cout << endl << "Execution time (parallel_for): " << tparallel_for << endl;

    return 0;
}