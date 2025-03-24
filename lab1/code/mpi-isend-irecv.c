#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 10000

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int message_sizes[] = {1,       64,      128,     256,      512,     1024,
                         2048,    4096,    8192,    16384,    65536,   262144,
                         1048576, 4194304, 8388608, 16777216, 33554432};
  int num_sizes = sizeof(message_sizes) / sizeof(int);

  char *buffer_send, *buffer_recv;
  double start, end, total_time;

  if (rank == 0) {
    printf("------------------------------------------------------------\n");
    printf("Size [B]\tThroughput [Mbit/s]\tTime [s]\n");
    printf("------------------------------------------------------------\n");
  }

  for (int s = 0; s < num_sizes; s++) {
    int size = message_sizes[s];
    buffer_send = (char *)malloc(size);
    buffer_recv = (char *)malloc(size);

    if (rank == 0) {
      start = MPI_Wtime();
    }

    if (rank == 0) {
      MPI_Request send_request, recv_request;
      MPI_Status send_status, recv_status;

      for (int i = 0; i < N; i++) {
        MPI_Isend(buffer_send, size, MPI_BYTE, 1, 0, MPI_COMM_WORLD,
                  &send_request);
        MPI_Irecv(buffer_recv, size, MPI_BYTE, 1, 0, MPI_COMM_WORLD,
                  &recv_request);

        MPI_Wait(&send_request, &send_status);
        MPI_Wait(&recv_request, &recv_status);
      }
    } else if (rank == 1) {
      MPI_Request send_request, recv_request;
      MPI_Status send_status, recv_status;

      for (int i = 0; i < N; i++) {
        MPI_Irecv(buffer_recv, size, MPI_BYTE, 0, 0, MPI_COMM_WORLD,
                  &recv_request);
        MPI_Isend(buffer_send, size, MPI_BYTE, 0, 0, MPI_COMM_WORLD,
                  &send_request);

        MPI_Wait(&recv_request, &recv_status);
        MPI_Wait(&send_request, &send_status);
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
      printf("%d\t\t%.2f\t\t%.12f\n", size, throughput, total_time);

      if (size == 1) {
        printf("\nDelay for the message 1B: %.12f s\n", total_time);
      }
    }

    free(buffer_send);
    free(buffer_recv);
  }

  MPI_Finalize();
  return 0;
}
