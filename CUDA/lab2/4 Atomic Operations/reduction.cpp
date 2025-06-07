#include <cstdio>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <iostream>
#include <cmath> // For fabsf

#include <cuda_runtime.h>
#include "/content/helper_timer.h"

#include "/content/reduction.h"

struct BenchmarkResult {
    float elapsed_time_ms;
    float bandwidth_GBs;
};

// Function Prototypes
void run_reduction(void (*reduce_func)(float*, float*, int, int),
                   float *d_outPtr, float *d_inPtr, int size, int n_threads);
BenchmarkResult run_benchmark(void (*reduce_func)(float*, float*, int, int),
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
    csv_file.open("reduction_atomic_benchmark.csv");
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
        
        // Data initialization with random values
        init_input(h_inPtr, size);

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

        // Run the benchmark on the GPU using the atomic_reduction kernel
        BenchmarkResult bench_res = run_benchmark(atomic_reduction, d_outPtr, d_inPtr, size);

        // The result of an atomic reduction is usually placed in a single memory location.
        // We assume it's the first element of the output buffer.
        err = cudaMemcpy(&result_gpu, d_outPtr, sizeof(float), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::cerr << "cudaMemcpy D2H failed: " << cudaGetErrorString(err) << std::endl;
        }

        // Get all sum from CPU for verification
        result_host = get_cpu_result(h_inPtr, size);
        printf("host: %f, device: %f, diff: %e\n", result_host, result_gpu, fabsf(result_host - result_gpu));
        
        // Log results to the CSV file
        csv_file << size << "," << bench_res.elapsed_time_ms << "," << bench_res.bandwidth_GBs
                 << "," << result_host << "," << result_gpu << "," << fabsf(result_host - result_gpu) << "\n";
    }
    
    // Final cleanup
    if (h_inPtr) free(h_inPtr);
    if (d_inPtr) cudaFree(d_inPtr);
    if (d_outPtr) cudaFree(d_outPtr);

    csv_file.close();
    printf("\nBenchmark results saved to reduction_atomic_benchmark.csv\n");

    return 0;
}

void run_reduction(void (*reduce_func)(float*, float*, int, int),
                   float *d_outPtr, float *d_inPtr, int size, int n_threads)
{
    // Simple wrapper that calls the provided reduction kernel once
    reduce_func(d_outPtr, d_inPtr, size, n_threads);
}

BenchmarkResult run_benchmark(void (*reduce_func)(float*, float*, int, int),
                              float *d_outPtr, float *d_inPtr, int size)
{
    int num_threads = 256;
    int test_iter = 100;

    // Warm-up run to ensure caches are warm and GPU is clocked up
    run_reduction(reduce_func, d_outPtr, d_inPtr, size, num_threads);
    cudaDeviceSynchronize();

    // Initialize timer
    StopWatchInterface *timer;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    ////////
    // Operation body
    ////////
    for (int i = 0; i < test_iter; i++) {
        // Since the reduction is in-place on d_outPtr, we must restore
        // the original data from d_inPtr before each timed iteration.
        // For atomic reduction, this also resets the output location to 0.
        cudaMemset(d_outPtr, 0, sizeof(float));
        run_reduction(reduce_func, d_outPtr, d_inPtr, size, num_threads);
    }

    // Getting elapsed time
    cudaDeviceSynchronize();
    sdkStopTimer(&timer);

    // Compute and print the performance
    float elapsed_time_ms = sdkGetTimerValue(&timer) / (float)test_iter;
    float bandwidth = (float)size * sizeof(float) / elapsed_time_ms / 1e6;
    printf("  Time= %.3f msec, Bandwidth= %f GB/s\n", elapsed_time_ms, bandwidth);

    sdkDeleteTimer(&timer);

    return {elapsed_time_ms, bandwidth};
}

void init_input(float *data, int size)
{
    for (int i = 0; i < size; i++)
    {
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