%%cuda
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<stdbool.h>
#include<time.h>

#define MAX_N 4096      
#define MAX_BLOCK_SIZE 32 

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

__global__ void matrix_transpose_naive(int *input, int *output, int N) {

	int indexX = threadIdx.x + blockIdx.x * blockDim.x;
	int indexY = threadIdx.y + blockIdx.y * blockDim.y;
	
	if (indexX < N && indexY < N) {
		int index = indexY * N + indexX;
		int transposedIndex = indexX * N + indexY;

		
		output[transposedIndex] = input[index];
	}
}

__global__ void matrix_transpose_shared(int *input, int *output, int N, int BLOCK_SIZE) {

	
	extern __shared__ int sharedMemory[];
	
	
	
	
	int indexX = threadIdx.x + blockIdx.x * blockDim.x;
	int indexY = threadIdx.y + blockIdx.y * blockDim.y;

	
	int tindexX = threadIdx.x + blockIdx.y * blockDim.x;
	int tindexY = threadIdx.y + blockIdx.x * blockDim.y;

	
	int localIndexX = threadIdx.x;
	int localIndexY = threadIdx.y;

	if (indexX < N && indexY < N) {
		int index = indexY * N + indexX;
		int localIndex = localIndexY * BLOCK_SIZE + localIndexX;
		
		
		sharedMemory[localIndex] = input[index];
	}

	__syncthreads();

	if (tindexX < N && tindexY < N) {
		int transposedIndex = tindexY * N + tindexX;
		int transposedLocalIndex = localIndexX * BLOCK_SIZE + localIndexY;
		
		
		output[transposedIndex] = sharedMemory[transposedLocalIndex];
	}
}


void fill_array(int *data, int N) {
	for(int idx=0;idx<(N*N);idx++)
		data[idx] = idx;
}


void cpu_transpose(int *input, int *output, int N) {
	for(int i = 0; i < N; i++) {
		for(int j = 0; j < N; j++) {
			output[j * N + i] = input[i * N + j];
		}
	}
}


bool verify_results(int *gpu_result, int *cpu_result, int N) {
	for(int idx = 0; idx < N*N; idx++) {
		if(gpu_result[idx] != cpu_result[idx]) {
			printf("Mismatch at index %d: GPU=%d, CPU=%d\n", idx, gpu_result[idx], cpu_result[idx]);
			return false;
		}
	}
	return true;
}

void print_output(int *a, int *b, int N) {
	printf("\n Original Matrix::\n");
	for(int idx=0;idx<(N*N);idx++) {
		if(idx%N == 0)
			printf("\n");
		printf(" %d ",  a[idx]);
	}
	printf("\n Transposed Matrix::\n");
	for(int idx=0;idx<(N*N);idx++) {
		if(idx%N == 0)
			printf("\n");
		printf(" %d ",  b[idx]);
	}
}
int main(void) {
	FILE *csv_file = fopen("matrix_transpose_results.csv", "w");
	if (!csv_file) {
		printf("Error: Could not create CSV file\n");
		return 1;
	}
	
	
	fprintf(csv_file, "N,Block_Size,Num_Blocks,Kernel_Type,GPU_Time_ms,CPU_Time_ms,Results_Match\n");
	
	printf("Starting matrix transpose performance analysis...\n");
	printf("Matrix sizes: 256x256 to %dx%d\n", MAX_N, MAX_N);
	printf("Block sizes: 8x8 to %dx%d\n", MAX_BLOCK_SIZE, MAX_BLOCK_SIZE);
	
	
	for(int power = 8; power <= 12; power++) {
		int N = 1 << power;  
		if (N > MAX_N) break;
		
		printf("\nTesting N = %dx%d (2^%d)\n", N, N, power);
		
		
		for(int block_size = 8; block_size <= MAX_BLOCK_SIZE; block_size *= 2) {
			
			printf("  Block size: %dx%d\n", block_size, block_size);
			
			int *a, *b_naive, *b_shared, *b_cpu;
			int *d_a, *d_b_naive, *d_b_shared;
			GpuTimer gpu_timer;
			
			int size = N * N * sizeof(int);
			int grid_size = (N + block_size - 1) / block_size;
			int num_blocks = grid_size * grid_size;
			
			
			a = (int *)malloc(size);
			b_naive = (int *)malloc(size);
			b_shared = (int *)malloc(size);
			b_cpu = (int *)malloc(size);
			
			if (!a || !b_naive || !b_shared || !b_cpu) {
				printf("Error: Host memory allocation failed for N=%d\n", N);
				continue;
			}
			
			
			fill_array(a, N);
			
			
			cudaError_t err1 = cudaMalloc((void **)&d_a, size);
			cudaError_t err2 = cudaMalloc((void **)&d_b_naive, size);
			cudaError_t err3 = cudaMalloc((void **)&d_b_shared, size);
			
			if (err1 != cudaSuccess || err2 != cudaSuccess || err3 != cudaSuccess) {
				printf("Error: Device memory allocation failed for N=%d\n", N);
				free(a); free(b_naive); free(b_shared); free(b_cpu);
				continue;
			}
			
			
			cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
			
			dim3 blockSize(block_size, block_size, 1);
			dim3 gridSize(grid_size, grid_size, 1);
			
			
			gpu_timer.Start();
			matrix_transpose_naive<<<gridSize, blockSize>>>(d_a, d_b_naive, N);
			cudaDeviceSynchronize();
			gpu_timer.Stop();
			float naive_time = gpu_timer.Elapsed();
			
			
			cudaMemcpy(b_naive, d_b_naive, size, cudaMemcpyDeviceToHost);
			
			
			int shared_mem_size = block_size * block_size * sizeof(int);
			gpu_timer.Start();
			matrix_transpose_shared<<<gridSize, blockSize, shared_mem_size>>>(d_a, d_b_shared, N, block_size);
			cudaDeviceSynchronize();
			gpu_timer.Stop();
			float shared_time = gpu_timer.Elapsed();
			
			
			cudaMemcpy(b_shared, d_b_shared, size, cudaMemcpyDeviceToHost);
			
			
			clock_t cpu_start = clock();
			cpu_transpose(a, b_cpu, N);
			clock_t cpu_end = clock();
			float cpu_time = ((float)(cpu_end - cpu_start) / CLOCKS_PER_SEC) * 1000.0; 
			
			
			bool naive_match = verify_results(b_naive, b_cpu, N);
			bool shared_match = verify_results(b_shared, b_cpu, N);
			
			
			fprintf(csv_file, "%d,%d,%d,Naive,%.6f,%.6f,%s\n", 
					N, block_size, num_blocks, naive_time, cpu_time, 
					naive_match ? "true" : "false");
			
			fprintf(csv_file, "%d,%d,%d,Shared,%.6f,%.6f,%s\n", 
					N, block_size, num_blocks, shared_time, cpu_time, 
					shared_match ? "true" : "false");
			
			printf("    Naive:  %8.3f ms, CPU: %8.3f ms, Match: %s\n",
				   naive_time, cpu_time, naive_match ? "YES" : "NO");
			printf("    Shared: %8.3f ms, CPU: %8.3f ms, Match: %s\n",
				   shared_time, cpu_time, shared_match ? "YES" : "NO");
			printf("    Speedup (Naive vs Shared): %.2fx\n", naive_time / shared_time);
			
			
			free(a); free(b_naive); free(b_shared); free(b_cpu);
			cudaFree(d_a); cudaFree(d_b_naive); cudaFree(d_b_shared);
		}
	}
	
	fclose(csv_file);
	printf("\nResults saved to matrix_transpose_results.csv\n");
	
	return 0;
}
