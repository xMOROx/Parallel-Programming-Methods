#include <iostream>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include "kernel.h"
#include "kernel.cu"
#include "dev_array.h"
#include <math.h>
#include "helper_timer.h"
#include <fstream>

using namespace std;

int main()
{
    int matrix_sizes[] = {128, 256, 512, 1024};

    int num_sizes = 4;
    
    // Create CSV file
    ofstream csv_file("matrix_multiplication_results.csv");
    csv_file << "Method,Matrix_Size,Time_ms,Bandwidth_GB_s,Error" << endl;

    for (int size_idx = 0; size_idx < num_sizes; size_idx++) {
        int N = matrix_sizes[size_idx];
        int SIZE = N * N;
        
        cout << "Testing matrix size: " << N << "x" << N << endl;

        // Allocate memory on the host
        vector<float> h_A(SIZE);
        vector<float> h_B(SIZE);
        vector<float> h_C(SIZE);

        // Initialize matrices on the host
        for (int i=0; i<N; i++){
            for (int j=0; j<N; j++){
                h_A[i*N+j] = sin(i);
                h_B[i*N+j] = cos(j);
            }
        }

        // Allocate memory on the device
        dev_array<float> d_A(SIZE);
        dev_array<float> d_B(SIZE);
        dev_array<float> d_C(SIZE);

        // Create timer
        StopWatchInterface *timer = NULL;
        sdkCreateTimer(&timer);

        // Start timing
        sdkStartTimer(&timer);

        d_A.set(&h_A[0], SIZE);
        d_B.set(&h_B[0], SIZE);

        matrixMultiplication(d_A.getData(), d_B.getData(), d_C.getData(), N);
        cudaDeviceSynchronize();

        d_C.get(&h_C[0], SIZE);
        cudaDeviceSynchronize();

        // Stop timing
        sdkStopTimer(&timer);
        float gpu_time = sdkGetTimerValue(&timer);

        // Calculate bandwidth (read A, read B, write C = 3 * N^2 * sizeof(float) bytes)
        float bytes_transferred = 3.0f * N * N * sizeof(float);
        float bandwidth_gb_s = (bytes_transferred / (gpu_time / 1000.0f)) / (1024.0f * 1024.0f * 1024.0f);

        float *cpu_C;
        cpu_C=new float[SIZE];

        // Now do the matrix multiplication on the CPU
        float sum;
        for (int row=0; row<N; row++){
            for (int col=0; col<N; col++){
                sum = 0.f;
                for (int n=0; n<N; n++){
                    sum += h_A[row*N+n]*h_B[n*N+col];
                }
                cpu_C[row*N+col] = sum;
            }
        }

        double err = 0;
        // Check the result and make sure it is correct
        for (int ROW=0; ROW < N; ROW++){
            for (int COL=0; COL < N; COL++){
                err += cpu_C[ROW * N + COL] - h_C[ROW * N + COL];
            }
        }

        cout << "Matrix size: " << N << "x" << N << ", Time: " << gpu_time << " ms, Bandwidth: " << bandwidth_gb_s << " GB/s, Error: " << err << endl;
        
        // Write to CSV
        csv_file << "simplest," << N << "," << gpu_time << "," << bandwidth_gb_s << "," << err << endl;

        // Clean up
        delete[] cpu_C;
        sdkDeleteTimer(&timer);
    }

    csv_file.close();
    cout << "Results saved to matrixs_simplest_multiplication_results.csv" << endl;

    return 0;
}
