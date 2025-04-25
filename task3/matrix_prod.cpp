#include <iostream>
#include <vector>
#include <thread>
#include <chrono>

using namespace std;

void initialize_matrix(vector<double>& mat, int r, int c) {
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            mat[i * c + j] = i + j;
        }
    }
}

void initialize_vector(vector<double>& vec, int n) {
    for (int i = 0; i < n; i++) {
        vec[i] = i;
    }
}

void matrix_vector_product(const vector<double>& mat, const vector<double>& vec, 
                          vector<double>& res, int rStart, int rEnd, int c) {
    for (int i = rStart; i < rEnd; i++) {
        res[i] = 0.0;
        for (int j = 0; j < c; j++) {
            res[i] += mat[i * c + j] * vec[j];
        }
    }
}

double run_prod(const int rows, const int cols, const int num_threads)
{
    vector<double> mat(rows * cols);
    vector<double> vec(cols);
    vector<double> res(rows);

    thread matrix_thread_init(initialize_matrix, ref(mat), rows, cols);
    thread vector_thread_init(initialize_vector, ref(vec), cols);

    matrix_thread_init.join();
    vector_thread_init.join();

    auto t_start = chrono::steady_clock::now();

    vector<thread> threads;
    const int rows_per_thread = rows / num_threads;

    for (int i = 0; i < num_threads; i++) {
        int rStart = i * rows_per_thread;
        int rEnd = (i == num_threads - 1) ? rows : rStart + rows_per_thread;
        threads.emplace_back(matrix_vector_product, cref(mat), cref(vec), ref(res), 
                        rStart, rEnd, cols);
    }

    for (auto& thread : threads) {
        thread.join();
    }

    auto t_end = chrono::steady_clock::now();
    chrono::duration<double> exec_time = t_end - t_start;

    return exec_time.count();
}

int main(int argc, char **argv) 
{
    const int num_threads = stoi(argv[1]);
    const int rows = 40'000;
    const int cols = 40'000;

    double exec_time = 0.0;

    for (int i = 0; i < 10; i++)
    {
        exec_time += run_prod(rows, cols, num_threads);
    }
    // Пример вывода результатов (закомментирован)
    // for (int i = 0; i < rows; i++) {
    //     cout << res[i] << " ";
    // }
    // cout << endl;

    cout << "Execution time on " << num_threads << " threads: " 
         << exec_time / 10 << endl;

    return 0;
}