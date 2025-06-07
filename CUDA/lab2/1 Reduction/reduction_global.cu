%%cuda_group_save --name "reduction_global.cu" --group "reduction_example"

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <iostream>

#include <cuda_runtime.h>
#include "/content/helper_timer.h"

#include "/content/reduction.h"


struct BenchmarkResult {
    float elapsed_time_ms;
    float bandwidth_GBs;
};

BenchmarkResult run_benchmark(void (*reduce)(float *, float *, int, int),
                              float *d_outPtr, float *d_inPtr, int size);
void init_input(float *data, int size);
float get_cpu_result(float *data, int size);

int main(int argc, char *argv[])
{
    float *h_inPtr = nullptr;
    float *d_inPtr = nullptr, *d_outPtr = nullptr;

    std::vector<unsigned int> sizes_to_test = {1 << 16, 1 << 17, 1 << 18, 1 << 19, 1 << 20, 1 << 22, 1 << 24, 1 << 25, 1 << 26, 1 << 27, 1 << 28};

    float result_host, result_gpu;

    srand(2019);

    std::ofstream csv_file;
    csv_file.open("reduction_benchmark_global.csv");
    if (!csv_file.is_open()) {
        std::cerr << "Failed to open CSV file for writing." << std::endl;
        return 1;
    }
    csv_file << "Size,Time_ms,Bandwidth_GBs,Host_Result,Device_Result,Difference\n";

    for (unsigned int size : sizes_to_test) {
        printf("\nTesting size: %u\n", size);

        if (h_inPtr) free(h_inPtr);
        if (d_inPtr) cudaFree(d_inPtr);
        if (d_outPtr) cudaFree(d_outPtr);

        h_inPtr = (float *)malloc(size * sizeof(float));
        if (!h_inPtr) {
            std::cerr << "Failed to allocate host memory for size " << size << std::endl;
            continue;
        }

        init_input(h_inPtr, size);

        cudaError_t err;
        err = cudaMalloc((void **)&d_inPtr, size * sizeof(float));
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate d_inPtr: " << cudaGetErrorString(err) << std::endl;
            free(h_inPtr); h_inPtr = nullptr;
            continue;
        }
        err = cudaMalloc((void **)&d_outPtr, size * sizeof(float));
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate d_outPtr: " << cudaGetErrorString(err) << std::endl;
            cudaFree(d_inPtr); d_inPtr = nullptr;
            free(h_inPtr); h_inPtr = nullptr;
            continue;
        }

        err = cudaMemcpy(d_inPtr, h_inPtr, size * sizeof(float), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::cerr << "cudaMemcpy H2D failed: " << cudaGetErrorString(err) << std::endl;
            cudaFree(d_outPtr); d_outPtr = nullptr;
            cudaFree(d_inPtr); d_inPtr = nullptr;
            free(h_inPtr); h_inPtr = nullptr;
            continue;
        }


        BenchmarkResult bench_res = run_benchmark(global_reduction, d_outPtr, d_inPtr, size);

        err = cudaMemcpy(&result_gpu, d_outPtr, sizeof(float), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::cerr << "cudaMemcpy D2H failed: " << cudaGetErrorString(err) << std::endl;
        }


        result_host = get_cpu_result(h_inPtr, size);
        printf("host: %f, device %f, diff: %e\n", result_host, result_gpu, fabsf(result_host - result_gpu));

        csv_file << size << "," << bench_res.elapsed_time_ms << "," << bench_res.bandwidth_GBs
                 << "," << result_host << "," << result_gpu << "," << fabsf(result_host - result_gpu) << "\n";

    }

    // Final cleanup
    if (h_inPtr) free(h_inPtr);
    if (d_inPtr) cudaFree(d_inPtr);
    if (d_outPtr) cudaFree(d_outPtr);

    csv_file.close();
    printf("\nBenchmark results saved to reduction_benchmark_global.csv\n");

    return 0;
}

BenchmarkResult run_benchmark(void (*reduce)(float *, float *, int, int),
                              float *d_outPtr, float *d_inPtr, int size)
{
    int num_threads = 256;
    int test_iter = 100;

    cudaMemcpy(d_outPtr, d_inPtr, size * sizeof(float), cudaMemcpyDeviceToDevice);
    reduce(d_outPtr, d_outPtr, num_threads, size);
    cudaDeviceSynchronize();


    // initialize timer
    StopWatchInterface *timer;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    ////////
    // Operation body
    ////////
    for (int i = 0; i < test_iter; i++)
    {
        cudaMemcpy(d_outPtr, d_inPtr, size * sizeof(float), cudaMemcpyDeviceToDevice);
        reduce(d_outPtr, d_outPtr, num_threads, size);
    }

    cudaDeviceSynchronize();
    sdkStopTimer(&timer);

    float elapsed_time_msed = sdkGetTimerValue(&timer) / (float)test_iter;
    float bandwidth = (size * sizeof(float) / elapsed_time_msed / 1e6);

    printf("  Kernel Time= %.3f msec, Effective Bandwidth= %f GB/s\n", elapsed_time_msed, bandwidth);

    sdkDeleteTimer(&timer);

    return {elapsed_time_msed, bandwidth};
}

void init_input(float *data, int size)
{
    for (int i = 0; i < size; i++)
    {
        data[i] = (rand() & 0xFF) / (float)RAND_MAX;
    }
}

float get_cpu_result(float *data, int size)
{
    double result = 0.f;
    for (int i = 0; i < size; i++)
        result += data[i];

    return (float)result;
}