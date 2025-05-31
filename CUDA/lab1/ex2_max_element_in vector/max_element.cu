%%cuda
#include <cstdio>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>
#include <algorithm>

using namespace std;

struct GpuTimer
{
      cudaEvent_t start;
      cudaEvent_t stop;
 
      GpuTimer()
      {
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
      }
 
      ~GpuTimer()
      {
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
      }
 
      void Start()
      {
            cudaEventRecord(start, 0);
      }
 
      void Stop()
      {
            cudaEventRecord(stop, 0);
      }
 
      float Elapsed()
      {
            float elapsed;
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsed, start, stop);
            return elapsed;
      }
};

#define MAX_N 268435456      

__global__ void maxi(int* a, int* b, int n, int threads_per_block)
{
    int block = threads_per_block * blockIdx.x;
    int max_val = a[block]; 

    for (int i = block; i < min(threads_per_block + block, n); i++) {
        if (max_val < a[i]) {
            max_val = a[i];
        }
    }
    b[blockIdx.x] = max_val;
}


int host_max(int *a, int n) {
    int max_val = a[0];
    for(int i = 1; i < n; i++) {
        if(a[i] > max_val) {
            max_val = a[i];
        }
    }
    return max_val;
}


void fill_array(int *data, int n) {
    srand(42); 
    for(int i = 0; i < n; i++) {
        data[i] = rand() % (n * 10); 
    }
}


int gpu_max_reduction(int *d_array, int n, int threads_per_block, GpuTimer &timer) {
    int *d_temp;
    int current_n = n;
    
    int max_blocks = (n + threads_per_block - 1) / threads_per_block;
    cudaMalloc(&d_temp, max_blocks * sizeof(int));
    
    timer.Start();
    
    while (current_n > 1) {
        int grids = (current_n + threads_per_block - 1) / threads_per_block;
        
        maxi<<<grids, 1>>>(d_array, d_temp, current_n, threads_per_block);
        cudaDeviceSynchronize();
        
        current_n = grids;
        
        cudaMemcpy(d_array, d_temp, current_n * sizeof(int), cudaMemcpyDeviceToDevice);
    }
    
    timer.Stop();
    
    int result;
    cudaMemcpy(&result, d_array, sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaFree(d_temp);
    
    return result;
}

int main()
{
    FILE *csv_file = fopen("max_element_results.csv", "w");
    if (!csv_file) {
        printf("Error: Could not create CSV file\n");
        return 1;
    }
    
    
    fprintf(csv_file, "N,Threads_Per_Block,Num_Blocks,GPU_Time_ms,CPU_Time_ms,GPU_Result,CPU_Result,Results_Match\n");
    
    printf("Starting max element performance analysis...\n");
    printf("Array sizes: 2048 to %d\n", MAX_N);
    printf("Block sizes: 64 to 1024 threads\n");
    
    
    for(int power = 11; power <= 28; power++) {
        int N = 1 << power;  
        
        printf("\nTesting N = %d (2^%d)\n", N, power);
        
        
        for(int threads_per_block = 64; threads_per_block <= 1024; threads_per_block *= 2) {
            int *a;
            int *d_a;
            GpuTimer gpu_timer;
            
            int size = N * sizeof(int);
            int no_of_blocks = (N + threads_per_block - 1) / threads_per_block;
            
            
            a = (int *)malloc(size);
            if (!a) {
                printf("Error: Host memory allocation failed for N=%d\n", N);
                continue;
            }
            
            
            fill_array(a, N);
            
            
            cudaError_t err = cudaMalloc((void **)&d_a, size);
            if (err != cudaSuccess) {
                printf("Error: Device memory allocation failed for N=%d\n", N);
                free(a);
                continue;
            }
            
            
            cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
            
            
            int gpu_result = gpu_max_reduction(d_a, N, threads_per_block, gpu_timer);
            float gpu_time = gpu_timer.Elapsed();
            
            
            clock_t cpu_start = clock();
            int cpu_result = host_max(a, N);
            clock_t cpu_end = clock();
            float cpu_time = ((float)(cpu_end - cpu_start) / CLOCKS_PER_SEC) * 1000.0; 
            
            
            bool results_match = (gpu_result == cpu_result);
            
            
            fprintf(csv_file, "%d,%d,%d,%.6f,%.6f,%d,%d,%s\n", 
                    N, threads_per_block, no_of_blocks, 
                    gpu_time, cpu_time, gpu_result, cpu_result,
                    results_match ? "true" : "false");
            
            printf("  Threads/Block: %4d, Blocks: %8d, GPU: %8.3f ms, CPU: %8.3f ms, GPU_Max: %d, CPU_Max: %d, Match: %s\n",
                   threads_per_block, no_of_blocks, gpu_time, cpu_time, 
                   gpu_result, cpu_result, results_match ? "YES" : "NO");
            
            
            free(a);
            cudaFree(d_a);
        }
    }
    
    fclose(csv_file);
    printf("\nResults saved to max_element_results.csv\n");
    
    return 0;
}