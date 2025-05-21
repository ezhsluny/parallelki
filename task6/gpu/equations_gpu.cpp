#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <iomanip>
#include <boost/program_options.hpp>
#include <chrono>

int N = 256;
double EPS = 1.0e-6;
int ITER_MAX = 1000000;

void print_matrix(double* matrix, int nx, int ny) {
    std::cout << "\nMatrix values:\n";
    for (int i = 0; i < ny; i++) {
        for (int j = 0; j < nx; j++) {
            std::cout << std::fixed << std::setprecision(2) 
                      << std::setw(6) << matrix[i*nx + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}

void initialize(double* matrix, int nx, int ny) {
    double corners[4] = {10.0, 20.0, 30.0, 20.0};
    
    // Initialize all to 0 on host
    for (int i = 0; i < (nx + 2) * (ny + 2); i++) {
        matrix[i] = 0.0;
    }
    
    // Set corner values
    matrix[(nx + 2) + 1] = corners[0];
    matrix[(nx + 2) * 2 - 2] = corners[1];
    matrix[(nx + 2) * (ny + 1) - 2] = corners[2];
    matrix[(nx + 2) * ny + 1] = corners[3];
    
    // Initialize edges
    for (int i = (nx + 2) + 2, j = 1; i < (nx + 2) * 2 - 2; i++, j++) {
        double coef = (double)(j) / (nx - 1);
        matrix[i] = corners[0] * (1.0 - coef) + corners[1] * coef;
        matrix[(nx + 2) * ny + 1 + j] = corners[3] * (1.0 - coef) + corners[2] * coef;
    }

    for (int i = (nx + 2) * 2 - 2 + (nx + 2), j = 1; i < (nx + 2) * (ny + 1) - 2; i+=(nx + 2), j++) {
        double coef = (double)(j) / (ny - 1);
        matrix[i] = corners[1] * (1.0 - coef) + corners[2] * coef;
        matrix[i - nx + 1] = corners[0] * (1.0 - coef) + corners[3] * coef;
    }
}

double solve_heat_equation(double* A, double* Anew, int nx, int ny, double eps, int iter_max) {
    double error = eps + 1.0; // Ensure we enter the loop
    int iter = 0;
    
    #pragma acc enter data copyin(A[0:(nx+2)*(ny+2)], Anew[0:(nx+2)*(ny+2)])
    
    while (error > eps && iter < iter_max) {
        error = 0.0;
        
        // Create device copy of error for reduction
        #pragma acc data copy(error)
        {
            // Compute new values and calculate error in one pass
            #pragma acc parallel loop collapse(2) present(A, Anew) reduction(max:error)
            for (int i = 1; i <= ny; i++) {
                for (int j = 1; j <= nx; j++) {
                    Anew[i*(nx + 2) + j] = 0.25 * (A[i*(nx + 2) + j-1] + A[i*(nx + 2) + j+1] + 
                                             A[(i-1)*(nx + 2) + j] + A[(i+1)*(nx + 2) + j]);
                    error = fmax(error, fabs(Anew[i*(nx + 2) + j] - A[i*(nx + 2) + j]));
                }
            }
            
            // Swap pointers instead of copying data
            // #pragma acc parallel loop collapse(2) present(A, Anew)
            // for (int i = 1; i <= ny; i++) {
            //     for (int j = 1; j <= nx; j++) {
            //         A[i*(nx + 2) + j] = Anew[i*(nx + 2) + j];
            //     }
            // }    
        }

        std::swap(A, Anew);
        
        // if (iter % 100 == 0) {
        //     printf("Iteration %d, error = %0.6f\n", iter, error);
        // }
        
        iter++;
    }
    
    #pragma acc update self(A[0:(nx+2)*(ny+2)])
    #pragma acc exit data delete(A[0:(nx+2)*(ny+2)], Anew[0:(nx+2)*(ny+2)])
    
    return error;
}

int main(int argc, char *argv[]) {
    boost::program_options::options_description desc("Heat Equation Solver Options");
    desc.add_options()
        ("help", "help message")
        ("n", boost::program_options::value<int>()->default_value(256), "grid size")
        ("eps", boost::program_options::value<double>()->default_value(1.0e-6), "precision")
        ("iter", boost::program_options::value<int>()->default_value(1000000), "max iterations")
        ("profile", "enable profiling");
    
    boost::program_options::variables_map vm;
    try {
        boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
        boost::program_options::notify(vm);
        
        if (vm.count("help")) {
            std::cout << desc << "\n";
            return 0;
        }
        
        N = vm["n"].as<int>();
        EPS = vm["eps"].as<double>();
        ITER_MAX = vm["iter"].as<int>();
        
        if (vm.count("profile")) {
            ITER_MAX = 1000;  // Для профилирования
            std::cout << "PROFILING MODE\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    double *A = new double[(N + 2) * (N + 2)];
    double *Anew = new double[(N + 2) * (N + 2)];

    initialize(A, N, N);
    initialize(Anew, N, N);

    std::cout << "Initial matrix (first 10x10):\n";
    print_matrix(A, std::min(N, 13), std::min(N, 13));

    auto start = std::chrono::steady_clock::now();
    double final_error = solve_heat_equation(A, Anew, N, N, EPS, ITER_MAX);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "\nFinal results:" << std::endl;
    std::cout << "Grid size: " << N << " x " << N << std::endl;
    std::cout << "Iterations: " << (ITER_MAX == 100 ? "PROFILING" : std::to_string(ITER_MAX)) << std::endl;
    std::cout << "Final error: " << final_error << std::endl;
    std::cout << "Time elapsed: " << elapsed.count() << " seconds\n" << std::endl;

    print_matrix(A, std::min(N, 13), std::min(N, 13));

    delete[] A;
    delete[] Anew;
    
    return 0;
}