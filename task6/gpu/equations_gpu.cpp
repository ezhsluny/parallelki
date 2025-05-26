#include <iostream>
#include <boost/program_options.hpp>
#include <cmath>
#include <memory>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <openacc.h>
#include <cstring> // для strdup

void saveMatrixToFile(const double* matrix, int NX, const std::string& filename) {
    std::ofstream outputFile(filename);
    if (!outputFile.is_open()) {
        std::cerr << "Не удалось открыть файл " << filename << " для записи." << std::endl;
        return;
    }

    int fieldWidth = 10;

    for (int i = 0; i < NX; ++i) {
        for (int j = 0; j < NX; ++j) {
            outputFile << std::setw(fieldWidth) << std::fixed << std::setprecision(4) << matrix[i * NX + j];
        }
        outputFile << std::endl;
    }

    outputFile.close();
}

void initialize(std::unique_ptr<double[]> &matrix, int NX) {
    matrix[0] = 10.0;
    matrix[NX - 1] = 20.0;
    matrix[(NX - 1) * NX + (NX - 1)] = 30.0;
    matrix[(NX - 1) * NX] = 20.0;

    for (size_t i = 1; i < NX - 1; ++i) {
        matrix[i] = matrix[0] + (i * (matrix[NX - 1] - matrix[0]) / (NX - 1));
        matrix[i * NX] = matrix[0] + (i * (matrix[(NX - 1) * NX] - matrix[0]) / (NX - 1));
        matrix[i * NX + (NX - 1)] = matrix[NX - 1] + (i * (matrix[(NX - 1) * NX + (NX - 1)] - matrix[NX - 1]) / (NX - 1));
        matrix[(NX - 1) * NX + i] = matrix[(NX - 1) * NX] + (i * (matrix[(NX - 1) * NX + (NX - 1)] - matrix[(NX - 1) * NX]) / (NX - 1));
    }
}

void solve_heat_equation(int NX, double EPS, int ITER_MAX, bool profile_mode) {
    double error = 1.0;
    int iter = 0;

    std::unique_ptr<double[]> A(new double[NX * NX]);
    std::unique_ptr<double[]> Anew(new double[NX * NX]);

    initialize(A, NX);
    initialize(Anew, NX);

    auto start = std::chrono::high_resolution_clock::now();
    double* curA = A.get();
    double* prevA = Anew.get();

    #pragma acc data copyin(error, prevA[0:NX * NX], curA[0:NX * NX])
    {
        while (iter < ITER_MAX && error > EPS) {
            #pragma acc parallel loop independent collapse(2) present(curA, prevA)
            for (size_t i = 1; i < NX - 1; ++i) {
                for (size_t j = 1; j < NX - 1; ++j) {
                    curA[i * NX + j] = 0.25 * (prevA[i * NX + j + 1] + prevA[i * NX + j - 1] 
                                     + prevA[(i - 1) * NX + j] + prevA[(i + 1) * NX + j]);
                }
            }

            if ((iter + 1) % 10000 == 0 || profile_mode) {
                error = 0.0;
                #pragma acc update device(error)
                #pragma acc parallel loop independent collapse(2) reduction(max:error) present(curA, prevA)
                for (size_t i = 1; i < NX - 1; ++i) {
                    for (size_t j = 1; j < NX - 1; ++j) {
                        error = fmax(error, fabs(curA[i * NX + j] - prevA[i * NX + j]));
                    }
                }
                #pragma acc update self(error)
                
                if (!profile_mode) {
                    std::cout << "Итерация: " << iter + 1 << " ошибка: " << error << std::endl;
                }
            }

            std::swap(prevA, curA);
            ++iter;
        }
        #pragma acc update self(curA[0:NX * NX])
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    auto timeMs = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();

    if (!profile_mode) {
        std::cout << "Время: " << timeMs << " мс, Ошибка: " << error << ", Итерации: " << iter << std::endl;

        if (NX == 13 || NX == 10) {
            for (size_t i = 0; i < NX; ++i) {
                for (size_t j = 0; j < NX; ++j) {
                    std::cout << A[i * NX + j] << ' ';
                }
                std::cout << std::endl;
            }
        }
    }

    saveMatrixToFile(curA, NX, "matrix.txt");
}

int main(int argc, char const *argv[]) {
    boost::program_options::options_description desc("Options");
    desc.add_options()
        ("eps", boost::program_options::value<double>()->default_value(1e-6), "Точность")
        ("nx", boost::program_options::value<int>()->default_value(1024), "Размер матрицы")
        ("iters", boost::program_options::value<int>()->default_value(1000000), "Количество итераций")
        ("help", "Help message")
        ("profile", "Enable profiling");

    boost::program_options::variables_map vm;
    boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
    boost::program_options::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << "\n";
        exit(1);
    }

    int NX = vm["nx"].as<int>();
    double EPS = vm["eps"].as<double>();
    int ITER_MAX = vm["iters"].as<int>();
    bool profile_mode = vm.count("profile");

    if (profile_mode) {
        ITER_MAX = 1000;
        std::cout << "PROFILING MODE\n";
    }

    acc_set_device_num(3, acc_device_nvidia);

    // Установка переменных окружения для профилирования (исправлено с использованием strdup)
    if (profile_mode) {
        putenv(strdup("NV_ACC_NOTIFY=1"));
        putenv(strdup("NV_ACC_PROFILING=1"));
        putenv(strdup("NV_ACC_TIME=1"));
    }

    solve_heat_equation(NX, EPS, ITER_MAX, profile_mode);

    return 0;
}