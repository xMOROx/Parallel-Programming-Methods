#include<stdlib.h>
#include<math.h>
#include<stdbool.h>

#include<stdio.h>
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

void host_add(int *a, int *b, int *c, int n) {
	for(int idx=0;idx<n;idx++)
		c[idx] = a[idx] + b[idx];
}

__global__ void device_add(int *a, int *b, int *c, int n) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
        if (index < n)  
            c[index] = a[index] + b[index];
}



void fill_array(int *data, int n) {
	for(int idx=0;idx<n;idx++)
		data[idx] = idx;
}


bool verify_results(int *gpu_result, int *cpu_result, int n) {
	for(int idx=0; idx<n; idx++) {
		if(gpu_result[idx] != cpu_result[idx]) {
			printf("Mismatch at index %d: GPU=%d, CPU=%d\n", idx, gpu_result[idx], cpu_result[idx]);
			return false;
		}
	}
	return true;
}

void print_output(int *a, int *b, int*c, int n) {
	for(int idx=0;idx<n;idx++)
		printf("\n %d + %d  = %d",  a[idx] , b[idx], c[idx]);
}
int main(void) {
	FILE *csv_file = fopen("vector_addition_results.csv", "w");
	if (!csv_file) {
		printf("Error: Could not create CSV file\n");
		return 1;
	}
	
	
	fprintf(csv_file, "N,Threads_Per_Block,Num_Blocks,GPU_Time_ms,CPU_Time_ms,Results_Match\n");
	
	printf("Starting performance analysis...\n");
	printf("Array sizes: 2048 to %d\n", MAX_N);
	printf("Block sizes: 64 to 1024 threads\n");
	
	
	for(int power = 11; power <= 28; power++) {
		int N = 1 << power;  
		
		printf("\nTesting N = %d (2^%d)\n", N, power);
		
		
		for(int threads_per_block = 64; threads_per_block <= 1024; threads_per_block *= 2) {
			int *a, *b, *c_gpu, *c_cpu;
			int *d_a, *d_b, *d_c; 
			GpuTimer gpu_timer;
			
			int size = N * sizeof(int);
			int no_of_blocks = (N + threads_per_block - 1) / threads_per_block;  
			
			
			a = (int *)malloc(size);
			b = (int *)malloc(size);
			c_gpu = (int *)malloc(size);
			c_cpu = (int *)malloc(size);
			
			if (!a || !b || !c_gpu || !c_cpu) {
				printf("Error: Host memory allocation failed for N=%d\n", N);
				continue;
			}
			
			
			fill_array(a, N);
			fill_array(b, N);
			
			
			cudaError_t err1 = cudaMalloc((void **)&d_a, size);
			cudaError_t err2 = cudaMalloc((void **)&d_b, size);
			cudaError_t err3 = cudaMalloc((void **)&d_c, size);
			
			if (err1 != cudaSuccess || err2 != cudaSuccess || err3 != cudaSuccess) {
				printf("Error: Device memory allocation failed for N=%d\n", N);
				free(a); free(b); free(c_gpu); free(c_cpu);
				continue;
			}
			
			
			cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
			cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
			
			
			gpu_timer.Start();
			device_add<<<no_of_blocks, threads_per_block>>>(d_a, d_b, d_c, N);
			cudaDeviceSynchronize();  
			gpu_timer.Stop();
			float gpu_time = gpu_timer.Elapsed();
			
			
			cudaMemcpy(c_gpu, d_c, size, cudaMemcpyDeviceToHost);
			
			
			clock_t cpu_start = clock();
			host_add(a, b, c_cpu, N);
			clock_t cpu_end = clock();
			float cpu_time = ((float)(cpu_end - cpu_start) / CLOCKS_PER_SEC) * 1000.0; 
			
			
			bool results_match = verify_results(c_gpu, c_cpu, N);
			
			
			fprintf(csv_file, "%d,%d,%d,%.6f,%.6f,%s\n", 
					N, threads_per_block, no_of_blocks, 
					gpu_time, cpu_time, 
					results_match ? "true" : "false");
			
			printf("  Threads/Block: %4d, Blocks: %8d, GPU: %8.3f ms, CPU: %8.3f ms, Match: %s\n",
				   threads_per_block, no_of_blocks, gpu_time, cpu_time, 
				   results_match ? "YES" : "NO");
			
			
			free(a); free(b); free(c_gpu); free(c_cpu);
			cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
		}
	}
	
	fclose(csv_file);
	printf("\nResults saved to vector_addition_results.csv\n");
	
	return 0;
}
