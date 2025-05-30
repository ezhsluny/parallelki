#include <iostream>
#include <boost/program_options.hpp>
#include <cmath>
#include <memory>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <cstring>
#include <openacc.h>
#include <cublas_v2.h>

// Initialize the matrix with boundary conditions
void initialize(std::unique_ptr<double[]> &matrix, int NX) {
    // Set corner values
    matrix[0] = 10.0;
    matrix[NX - 1] = 20.0;
    matrix[(NX - 1) * NX + (NX - 1)] = 30.0;
    matrix[(NX - 1) * NX] = 20.0;

    // Linear interpolation for boundary values
    for (size_t i = 1; i < NX - 1; ++i) {
        matrix[i] = matrix[0] + (i * (matrix[NX - 1] - matrix[0]) / (NX - 1)); // Top edge
        matrix[i * NX] = matrix[0] + (i * (matrix[(NX - 1) * NX] - matrix[0]) / (NX - 1)); // Left edge
        matrix[i * NX + (NX - 1)] = matrix[NX - 1] + (i * (matrix[(NX - 1) * NX + (NX - 1)] - matrix[NX - 1]) / (NX - 1)); // Right edge
        matrix[(NX - 1) * NX + i] = matrix[(NX - 1) * NX] + (i * (matrix[(NX - 1) * NX + (NX - 1)] - matrix[(NX - 1) * NX]) / (NX - 1)); // Bottom edge
    }
}

// Function to save the matrix to a file
void saveMatrixToFile(const double* matrix, int size, const std::string& filename) {
    std::ofstream outputFile(filename);
    if (!outputFile.is_open()) {
        std::cerr << "Не удалось открыть файл " << filename << " для записи." << std::endl;
        return;
    }

    int fieldWidth = 10;

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            outputFile << std::setw(fieldWidth) << std::fixed << std::setprecision(4) << matrix[i * size + j];
        }
        outputFile << std::endl;
    }

    outputFile.close();
}

// Function to solve the heat equation
void solve_heat_equation(int NX, double EPS, int ITER_MAX, bool profile_mode) {
    double error = 1.0;
    int iter = 0;

    // Initialize CUDA BLAS
    cublasStatus_t status;
    cublasHandle_t cublasHandle;
    status = cublasCreate(&cublasHandle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "CUBLAS initialization failed\n";
        return;
    }

    // Allocate memory for matrices
    std::unique_ptr<double[]> A(new double[NX * NX]);
    std::unique_ptr<double[]> newA(new double[NX * NX]);

    double alpha = -1.0;
    int maxIndex = 0;

    // Initialize matrices
    initialize(A, NX);
    initialize(newA, NX);

    auto start = std::chrono::high_resolution_clock::now();
    double* curA = A.get();
    double* prevA = newA.get();

    // Copy data to the GPU
    #pragma acc enter data copyin(curA[0:NX*NX], prevA[0:NX*NX], maxIndex, alpha)

    // Iterative solver loop
    while (iter < ITER_MAX && error > EPS) {
        // Update the matrix
        #pragma acc parallel loop collapse(2) present(curA, prevA)
        for (size_t i = 1; i < NX - 1; ++i) {
            for (size_t j = 1; j < NX - 1; ++j) {
                curA[i * NX + j] = 0.25 * (
                    prevA[i * NX + j + 1] +
                    prevA[i * NX + j - 1] +
                    prevA[(i - 1) * NX + j] +
                    prevA[(i + 1) * NX + j]);
            }
        }

        // Periodically check error and copy data between matrices
        if ((iter + 1) % 100 == 0) {
            #pragma acc host_data use_device(curA, prevA)
            {
                // Compute the difference between matrices
                status = cublasDaxpy(cublasHandle, NX * NX, &alpha, curA, 1, prevA, 1);
                if (status != CUBLAS_STATUS_SUCCESS) {
                    std::cerr << "cublasDaxpy failed\n";
                }

                // Find the maximum difference
                status = cublasIdamax(cublasHandle, NX * NX, prevA, 1, &maxIndex);
                if (status != CUBLAS_STATUS_SUCCESS) {
                    std::cerr << "cublasIdamax failed\n";
                }
            }

            // Update error and print progress
            #pragma acc update self(prevA[maxIndex - 1])
            error = fabs(prevA[maxIndex - 1]);
            if (!profile_mode) {
                std::cout << "Итерация: " << iter + 1 << " ошибка: " << error << std::endl;
            }

            // Copy current matrix to previous matrix
            #pragma acc host_data use_device(curA, prevA)
            {
                status = cublasDcopy(cublasHandle, NX * NX, curA, 1, prevA, 1);
                if (status != CUBLAS_STATUS_SUCCESS) {
                    std::cerr << "cublasDcopy failed\n";
                }
            }
        }

        // Swap matrices for next iteration
        std::swap(prevA, curA);
        iter++;
    }

    // Copy final data back to host
    #pragma acc update self(curA[0:NX*NX])
    #pragma acc exit data delete(curA[0:NX*NX], prevA[0:NX*NX], maxIndex, alpha)

    // Cleanup CUDA BLAS
    cublasDestroy(cublasHandle);

    // Calculate and print execution time
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    if (!profile_mode) {
        std::cout << "Время: " << duration << " мс, Ошибка: " << error << ", Итерации: " << iter << std::endl;

        if (NX == 13 || NX == 10) {
            for (size_t i = 0; i < NX; ++i) {
                for (size_t j = 0; j < NX; ++j) {
                    std::cout << curA[i * NX + j] << ' ';
                }
                std::cout << std::endl;
            }
        }
    }

    saveMatrixToFile(curA, NX, "matrix.txt");
}

int main(int argc, char const *argv[]) {
    // Define command line options
    boost::program_options::options_description desc("Options");
    desc.add_options()
        ("eps", boost::program_options::value<double>()->default_value(1e-6), "Точность")
        ("nx", boost::program_options::value<int>()->default_value(1024), "Размер матрицы")
        ("iters", boost::program_options::value<int>()->default_value(1000000), "Количество итераций")
        ("help", "Help message")
        ("profile", "Enable profiling");

    // Parse command line arguments
    boost::program_options::variables_map vm;
    boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
    boost::program_options::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 1;
    }

    // Extract parameters
    int NX = vm["nx"].as<int>();
    double EPS = vm["eps"].as<double>();
    int ITER_MAX = vm["iters"].as<int>();
    bool profile_mode = vm.count("profile");

    if (profile_mode) {
        ITER_MAX = 1000;
        std::cout << "PROFILING MODE\n";
        putenv(strdup("NV_ACC_NOTIFY=1"));
        putenv(strdup("NV_ACC_PROFILING=1"));
        putenv(strdup("NV_ACC_TIME=1"));
    }

    acc_set_device_num(3, acc_device_nvidia);

    solve_heat_equation(NX, EPS, ITER_MAX, profile_mode);

    return 0;
}
