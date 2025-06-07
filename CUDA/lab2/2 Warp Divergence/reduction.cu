#include <cstdio>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <iostream>
#include <cmath> 

#include <cuda_runtime.h>
#include "/content/helper_timer.h"

#include "/content/reduction.h"

struct BenchmarkResult {
    float elapsed_time_ms;
    float bandwidth_GBs;
};

// Function prototypes
void run_reduction(int (*reduce)(float*, float*, int, int),
                   float *d_outPtr, float *d_inPtr, int size, int n_threads);
BenchmarkResult run_benchmark(int (*reduce)(float*, float*, int, int),
                              float *d_outPtr, float *d_inPtr, int size);
void init_input(float* data, int size);
float get_cpu_result(float *data, int size);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[])
{
    float *h_inPtr = nullptr;
    float *d_inPtr = nullptr, *d_outPtr = nullptr;

    // Define the different input sizes to test
    std::vector<unsigned int> sizes_to_test = {1 << 16, 1 << 17, 1 << 18, 1 << 19, 1 << 20, 1 << 22, 1 << 24, 1 << 25, 1 << 26, 1 << 27, 1 << 28};

    float result_host, result_gpu;

    srand(2019);

    // Setup CSV file for logging results
    std::ofstream csv_file;
    csv_file.open("reduction_multikernel_benchmark.csv");
    if (!csv_file.is_open()) {
        std::cerr << "Failed to open CSV file for writing." << std::endl;
        return 1;
    }
    // Write CSV header
    csv_file << "Size,Time_ms,Bandwidth_GBs,Host_Result,Device_Result,Difference\n";

    // Loop over all specified sizes
    for (unsigned int size : sizes_to_test) {
        printf("\nTesting size: %u\n", size);

        // Free memory from previous iteration, if any
        if (h_inPtr) free(h_inPtr);
        if (d_inPtr) cudaFree(d_inPtr);
        if (d_outPtr) cudaFree(d_outPtr);

        // Allocate host memory
        h_inPtr = (float*)malloc(size * sizeof(float));
        if (!h_inPtr) {
            std::cerr << "Failed to allocate host memory for size " << size << std::endl;
            continue; // Skip to next size
        }

        // Data initialization with random values and obtain host answer
        init_input(h_inPtr, size);
        result_host = get_cpu_result(h_inPtr, size);

        // Allocate device memory and check for errors
        cudaError_t err;
        err = cudaMalloc((void**)&d_inPtr, size * sizeof(float));
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate d_inPtr: " << cudaGetErrorString(err) << std::endl;
            free(h_inPtr); h_inPtr = nullptr;
            continue;
        }
        err = cudaMalloc((void**)&d_outPtr, size * sizeof(float));
         if (err != cudaSuccess) {
            std::cerr << "Failed to allocate d_outPtr: " << cudaGetErrorString(err) << std::endl;
            cudaFree(d_inPtr); d_inPtr = nullptr;
            free(h_inPtr); h_inPtr = nullptr;
            continue;
        }

        // Copy data from host to device and check for errors
        err = cudaMemcpy(d_inPtr, h_inPtr, size * sizeof(float), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::cerr << "cudaMemcpy H2D failed: " << cudaGetErrorString(err) << std::endl;
            cudaFree(d_outPtr); d_outPtr = nullptr;
            cudaFree(d_inPtr); d_inPtr = nullptr;
            free(h_inPtr); h_inPtr = nullptr;
            continue;
        }

        // Run the benchmark on the GPU
        BenchmarkResult bench_res = run_benchmark(reduction, d_outPtr, d_inPtr, size);

        // Copy the final result from device to host and check for errors
        err = cudaMemcpy(&result_gpu, d_outPtr, sizeof(float), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::cerr << "cudaMemcpy D2H failed: " << cudaGetErrorString(err) << std::endl;
        }

        // Print comparison and log results
        printf("host: %f, device: %f, diff: %e\n", result_host, result_gpu, fabsf(result_host - result_gpu));
        
        csv_file << size << "," << bench_res.elapsed_time_ms << "," << bench_res.bandwidth_GBs
                 << "," << result_host << "," << result_gpu << "," << fabsf(result_host - result_gpu) << "\n";
    }

    // Final cleanup
    if (h_inPtr) free(h_inPtr);
    if (d_inPtr) cudaFree(d_inPtr);
    if (d_outPtr) cudaFree(d_outPtr);

    csv_file.close();
    printf("\nBenchmark results saved to reduction_multikernel_benchmark.csv\n");

    return 0;
}

void run_reduction(int (*reduce)(float*, float*, int, int),
                   float *d_outPtr, float *d_inPtr, int size, int n_threads)
{
    // This loop executes the reduction kernel multiple times until size is 1.
    // The kernel itself returns the new size of the partially-reduced array.
    while (size > 1) {
        size = reduce(d_outPtr, d_inPtr, size, n_threads);
    }
}

BenchmarkResult run_benchmark(int (*reduce)(float*, float*, int, int),
                              float *d_outPtr, float *d_inPtr, int size)
{
    int num_threads = 256;
    int test_iter = 100;

    // Warm-up: Copy data and run the full multi-pass reduction once
    cudaMemcpy(d_outPtr, d_inPtr, size * sizeof(float), cudaMemcpyDeviceToDevice);
    run_reduction(reduce, d_outPtr, d_outPtr, size, num_threads); // In-place reduction
    cudaDeviceSynchronize();

    // Initialize timer
    StopWatchInterface *timer;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    ////////
    // Operation body
    ////////
    for (int i = 0; i < test_iter; i++) {
        // We must copy the original data back to the output buffer before each timed run
        cudaMemcpy(d_outPtr, d_inPtr, size * sizeof(float), cudaMemcpyDeviceToDevice);
        // Perform the full multi-pass reduction
        run_reduction(reduce, d_outPtr, d_outPtr, size, num_threads);
    }

    // Getting elapsed time
    cudaDeviceSynchronize();
    sdkStopTimer(&timer);

    // Compute and print the performance
    float elapsed_time_ms = sdkGetTimerValue(&timer) / (float)test_iter;
    float bandwidth = (float)size * sizeof(float) / elapsed_time_ms / 1e6;
    printf("  Time= %.3f msec, Bandwidth= %f GB/s\n", elapsed_time_ms, bandwidth);

    sdkDeleteTimer(&timer);
    
    // Return results in the struct
    return {elapsed_time_ms, bandwidth};
}

void init_input(float *data, int size)
{
    for (int i = 0; i < size; i++) {
        // Keep the numbers small so we don't get truncation error in the sum
        data[i] = (rand() & 0xFF) / (float)RAND_MAX;
    }
}

float get_cpu_result(float *data, int size)
{
    double result = 0.0; // Use double for host accumulation to improve precision
    for (int i = 0; i < size; i++)
        result += data[i];

    return (float)result;
}