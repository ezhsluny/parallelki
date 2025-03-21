#include <iostream>
#include <queue>
#include <future>
#include <thread>
#include <chrono>
#include <cmath>
#include <functional>
#include <mutex>

using namespace std;

void initialize_matrix(double *mat, int r, int c)
{
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            mat[i * c + j] = i + j;
        }
    }
}

void initialize_vector(double *vec, int n)
{
    for (int i = 0; i < n; i++)
    {
        vec[i] = i;
    }
}

void matrix_vector_product(double *mat, double *vec, double *res, int rStart, int rEnd, int c)
{
    for (int i = rStart; i < rEnd; i++) {
        res[i] = 0.0;
        for (int j = 0; j < c; j++)
            res[i] += mat[i * c + j] * vec[j];
    }
}

int main()
{
    int rows = 40'000;
    int cols = 40'000;

    auto t_start = chrono::high_resolution_clock::now();

    double* mat = (double*)malloc(sizeof(*mat) * rows * cols);
    double* vec = (double*)malloc(sizeof(*vec) * cols);
    double* res = (double*)malloc(sizeof(*res) * rows);

    thread matrix_thread_init(initialize_matrix, mat, rows, cols);
    thread vector_thread_init(initialize_vector, vec, cols);

    matrix_thread_init.join();
    vector_thread_init.join();

    int num_threads = 40;
    vector<thread> threads;
    int rows_on_thread = rows / num_threads;

    for (int i = 0; i < num_threads; i++)
    {
        int rStart = i*rows_on_thread;
        int rEnd = (i == num_threads - 1) ? rows : rStart + rows_on_thread;
        threads.emplace_back(matrix_vector_product, mat, vec, res, rStart, rEnd, cols);
    }

    for (auto& thread : threads)
    {
        thread.join();
    }

    auto t_end = chrono::high_resolution_clock::now();

    // for (int i = 0; i < rows; i++)
    // {
    //     cout << res[i] << " ";
    // }
    // cout << endl;

    chrono::duration<double> exec_time = t_end - t_start;
    cout << "Execution time on " << num_threads << " threads: " << exec_time.count();
    cout << endl;

    free(mat);
    free(vec);
    free(res);

    return 0;
}