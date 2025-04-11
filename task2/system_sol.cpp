#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <omp.h>
#include <chrono>

using namespace std;

const double EPS = 0.00001;
const int MAX_ITER = 1000;
const int N = 2000;
const double T = 0.1;

void solving_parallel(const vector<double>& a, const vector<double>& b, vector<double>& x) {
    vector<double> x0(N);

    #pragma omp parallel
    {
        for (int itr = 0; itr < MAX_ITER; itr++) 
        {
            double local_diff = 0.0;
            
            int threadid = omp_get_thread_num();
            int nthreads = omp_get_num_threads();
            int items_per_thread = N / nthreads;
            int lb = threadid * items_per_thread;
            int ub = (threadid == nthreads - 1) ? N : lb + items_per_thread;
    
            for (int i = lb; i < ub; i++) 
            {
                double tmp = 0.0;
                for (int j = 0; j < N; j++) 
                {
                    if (i != j)
                    {
                        tmp += a[i*N + j] * x[j];
                    }
                }

                double upd_x = (b[i] - tmp) / a[i*N + i];
                x0[i] = x[i] + T * upd_x;
                local_diff = max(local_diff, abs(x0[i] - x[i]));
            }

            #pragma omp critical
            {
                static double global_diff = 0.0;
                global_diff = max(global_diff, local_diff);
                
                if (global_diff < EPS) {
                    itr = MAX_ITER;
                }
            }

            x = x0;
        }
    }
}

void solving_parallel_for(const vector<double>& a, const vector<double>& b, vector<double>& x) {
    vector<double> x0(N);
    double global_diff = EPS + 1;

    #pragma omp parallel
    {
        for (int itr = 0; itr < MAX_ITER && global_diff >= EPS; itr++) {
            double local_diff = 0.0;

            #pragma omp for schedule(static)
            for (int i = 0; i < N; i++) {
                double tmp = 0.0;
                for (int j = 0; j < N; j++) {
                    if (i != j) tmp += a[i*N + j] * x[j];
                }
                x0[i] = x[i] + T * ((b[i] - tmp) / a[i*N + i]);
                local_diff = max(local_diff, abs(x0[i] - x[i]));
            }

            #pragma omp critical
            {
                global_diff = max(global_diff, local_diff);
            }

            #pragma omp barrier

            #pragma omp for
            for (int i = 0; i < N; i++) {
                x[i] = x0[i];
            }
        }
    }
}

double run_parallel() {
    vector<double> a(0);
    vector<double> b(N, N + 1);
    vector<double> x(N, 0.0);

    a.reserve(N*N);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            a[i*N + j] = (i == j) ? 2.0 : 1.0;
        }
    }

    auto t_start = chrono::steady_clock::now();
    solving_parallel(a, b, x);
    auto t_end = chrono::steady_clock::now();
    chrono::duration<double> exec_time = t_end - t_start;

    for (int i = 0; i < N; i++) {
        if (fabs(x[i] - 1.0) > EPS) {
            cout << "Incorrect solution at x[" << i << "] = " << x[i] << endl;
            break;
        }
    }

    return exec_time.count();
}

double run_parallel_for() {
    vector<double> a(0);
    vector<double> b(N, N + 1);
    vector<double> x(N, 0.0);

    a.reserve(N*N);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            a[i*N + j] = (i == j) ? 2.0 : 1.0;
        }
    }

    auto t_start = chrono::steady_clock::now();
    solving_parallel_for(a, b, x);
    auto t_end = chrono::steady_clock::now();
    chrono::duration<double> exec_time = t_end - t_start;

    for (int i = 0; i < N; i++) {
        if (fabs(x[i] - 1.0) > EPS) {
            cout << "Incorrect solution at x[" << i << "] = " << x[i] << endl;
            break;
        }
    }
    
    return exec_time.count();
}

int main(int argc, char **argv) {   
    cout << endl << "A system of linear algebraic equations: size A = "
         << N << "x" << N  << endl
         << "size b = " << N << endl
         << "size x = " << N << endl;

    int num_thr = stoi(argv[1]);
    omp_set_num_threads(num_thr);
    double tparallel = 0.0;
    double tparallel_for = 0.0;

    for (int i = 0; i < 10; i++) {
        // tparallel += run_parallel();
        tparallel_for += run_parallel_for();
    }

    cout << endl << "Execution time (parallel): " << tparallel/10 << " s" << endl;
    cout << endl << "Execution time (parallel_for): " << tparallel_for/10 << " s" << endl;

    return 0;
}
