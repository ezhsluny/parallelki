#include <iostream>
#include <cmath>
#include <memory>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <boost/program_options.hpp>

// Function to initialize the matrix with boundary conditions
void initMatrix(std::unique_ptr<double[]>& matrix, int NX) {
    for (size_t i = 0; i < NX * NX; i++) {
        matrix[i] = 0;
    }

    // Set boundary values
    matrix[0] = 10.0;
    matrix[NX - 1] = 20.0;
    matrix[(NX - 1) * NX + (NX - 1)] = 30.0;
    matrix[(NX - 1) * NX] = 20.0;

    // Initialize the edges of the matrix with linear interpolation
    for (size_t i = 1; i < NX - 1; ++i) {
        matrix[i] = matrix[0] + (i * (matrix[NX - 1] - matrix[0]) / (NX - 1)); // Top edge
        matrix[i * NX] = matrix[0] + (i * (matrix[(NX - 1) * NX] - matrix[0]) / (NX - 1)); // Left edge
        matrix[i * NX + (NX - 1)] = matrix[NX - 1] + (i * (matrix[(NX - 1) * NX + (NX - 1)] - matrix[NX - 1]) / (NX - 1)); // Right edge
        matrix[(NX - 1) * NX + i] = matrix[(NX - 1) * NX] + (i * (matrix[(NX - 1) * NX + (NX - 1)] - matrix[(NX - 1) * NX]) / (NX - 1)); // Bottom edge
    }
}

// Function to save the matrix to a file
void saveMatrixToFile(const double* matrix, int N, const std::string& filename) {
    std::ofstream outputFile(filename);
    if (!outputFile.is_open()) {
        std::cerr << "Unable to open file " << filename << " for writing." << std::endl;
        return;
    }

    int fieldWidth = 10;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            outputFile << std::setw(fieldWidth) << std::fixed << std::setprecision(4) << matrix[i * N + j];
        }
        outputFile << std::endl;
    }
    outputFile.close();
}

// CUDA kernel for performing one iteration of the Jacobi method
__global__ void computeOneIteration(double* prevmatrix, double* curmatrix, int size) {
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;

    // Perform the Jacobi iteration if within bounds
    if (i < size - 1 && j < size - 1) {
        curmatrix[i * size + j] = 0.25 * (prevmatrix[i * size + j + 1] + prevmatrix[i * size + j - 1] +
                                        prevmatrix[(i - 1) * size + j] + prevmatrix[(i + 1) * size + j]);
    }
}

// CUDA kernel for calculating the error using CUB BlockReduce
__global__ void calculateError(double* prevmatrix, double* curmatrix, double* block_max_errors, int size) {
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;

    // Use CUB's BlockReduce to find the maximum error in each block
    typedef cub::BlockReduce<double, 32> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    // Calculate the error for the current thread
    double thread_error = 0.0;
    if (i < size - 1 && j < size - 1) {
        thread_error = fabs(curmatrix[i * size + j] - prevmatrix[i * size + j]);
    }

    // Reduce to find the maximum error in the block
    double block_max_error = BlockReduce(temp_storage).Reduce(thread_error, cub::Max());

    // Store the maximum error for the block
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        block_max_errors[blockIdx.y * gridDim.x + blockIdx.x] = block_max_error;
    }
}

int main(int argc, char** argv) {
    boost::program_options::options_description desc("Options");
    desc.add_options()
        ("eps", boost::program_options::value<double>()->default_value(1e-6), "Accuracy")
        ("nx", boost::program_options::value<int>()->default_value(256), "Grid size (N x N)")
        ("iters", boost::program_options::value<int>()->default_value(1000000), "Max iterations")
        ("help", "Show help message");

    boost::program_options::variables_map vm;
    boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
    boost::program_options::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 1;
    }

    // Extract parameters from command line
    const double EPS = vm["eps"].as<double>();
    const int N = vm["nx"].as<int>();
    const int max_iter = vm["iters"].as<int>();

    // Allocate host memory for matrices
    std::unique_ptr<double[]> A(new double[N * N]);
    std::unique_ptr<double[]> Anew(new double[N * N]);

    // Initialize matrices
    initMatrix(A, N);
    initMatrix(Anew, N);

    // Allocate device memory
    double *d_A, *d_Anew, *d_block_errors;
    cudaMalloc(&d_A, N * N * sizeof(double));
    cudaMalloc(&d_Anew, N * N * sizeof(double));

    // Determine block error storage size
    dim3 block_size(32, 32);
    dim3 grid_size((N + block_size.x - 1) / block_size.x,
                   (N + block_size.y - 1) / block_size.y);
    const int num_blocks = grid_size.x * grid_size.y;
    cudaMalloc(&d_block_errors, num_blocks * sizeof(double));

    // Copy initial data to device
    cudaMemcpy(d_A, A.get(), N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Anew, Anew.get(), N * N * sizeof(double), cudaMemcpyHostToDevice);

    // Create CUDA stream and graph
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaGraph_t graph;
    cudaGraphExec_t graph_instance;
    bool graph_created = false;

    // Main iteration loop
    double error = 1.0;
    int iter = 0;
    auto start = std::chrono::high_resolution_clock::now();

    while (error > EPS && iter < max_iter) {
        if (!graph_created) {
            // Begin capturing CUDA graph
            cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

            for (int i = 0; i < 1000; i++) {
                computeOneIteration<<<grid_size, block_size, 0, stream>>>(d_Anew, d_A, N);
                std::swap(d_Anew, d_A);
            }

            // Capture error calculation
            calculateError<<<grid_size, block_size, 0, stream>>>(d_Anew, d_A, d_block_errors, N);

            // End capturing and instantiate the graph
            cudaStreamEndCapture(stream, &graph);
            cudaGraphInstantiate(&graph_instance, graph, NULL, NULL, 0);
            graph_created = true;
        }

        // Launch the graph
        cudaGraphLaunch(graph_instance, stream);
        cudaStreamSynchronize(stream);

        // Find maximum error across blocks
        std::unique_ptr<double[]> block_errors(new double[num_blocks]);
        cudaMemcpy(block_errors.get(), d_block_errors, num_blocks * sizeof(double), cudaMemcpyDeviceToHost);

        error = 0.0;
        for (int i = 0; i < num_blocks; i++) {
            if (block_errors[i] > error) {
                error = block_errors[i];
            }
        }

        iter += 1000;
        std::cout << "Iteration: " << iter << ", Error: " << error << std::endl;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // Copy final result back to host
    cudaMemcpy(A.get(), d_A, N * N * sizeof(double), cudaMemcpyDeviceToHost);

    std::cout << "Final results - Iterations: " << iter << ", Error: " << error << std::endl;
    std::cout << "Time: " << duration << " ms" << std::endl;

    saveMatrixToFile(A.get(), N, "matrix.txt");

    if (graph_created) {
        cudaGraphExecDestroy(graph_instance);
        cudaGraphDestroy(graph);
    }
    cudaStreamDestroy(stream);
    cudaFree(d_A);
    cudaFree(d_Anew);
    cudaFree(d_block_errors);

    return 0;
}
