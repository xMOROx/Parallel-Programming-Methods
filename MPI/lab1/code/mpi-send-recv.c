#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define N 10000

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int message_sizes[] = {1, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 65536, 262144, 1048576, 4194304, 8388608, 16777216, 33554432};
    int num_sizes = sizeof(message_sizes) / sizeof(int);

    char *buffer;
    double start, end, total_time;

    if (rank == 0) {
        printf("------------------------------------------------------------\n");
        printf("Size [B]\tThroughput [Mbit/s]\tTime [s]\n");
        printf("------------------------------------------------------------\n");
    }

    for (int s = 0; s < num_sizes; s++) {
        int size = message_sizes[s];
        buffer = (char *)malloc(size);

        if (rank == 0) {
            start = MPI_Wtime();
        }

        if (rank == 0) {
            for (int i = 0; i < N; i++) {
                MPI_Send(buffer, size, MPI_BYTE, 1, 0, MPI_COMM_WORLD);
                MPI_Recv(buffer, size, MPI_BYTE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        } else if (rank == 1) {
            for (int i = 0; i < N; i++) {
                MPI_Recv(buffer, size, MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(buffer, size, MPI_BYTE, 0, 0, MPI_COMM_WORLD);
            }
        }
        
        if (rank == 0) {
            end = MPI_Wtime();
            total_time = (end - start) / (2.0 * N); 

            double throughput = (size * 8) / (total_time * 1e6);

            FILE *fp;
            if (s == 0) {
                fp = fopen("results.csv", "w");
                fprintf(fp, "Size(B),Throughput(Mbit/s),Time(s)\n");
            } else {
                fp = fopen("results.csv", "a");
            }
            fprintf(fp, "%d,%.2f,%.12f\n", size, throughput, total_time);
            fclose(fp);
            
            printf("%d\t\t%.2f\t\t%.12fs\n", size, throughput, total_time);

            if (size == 1) {
                printf("\nDelay for the message 1B: %.12f s\n", total_time );
            }
        }

        free(buffer);
    }

    MPI_Finalize();
    return 0;
}
