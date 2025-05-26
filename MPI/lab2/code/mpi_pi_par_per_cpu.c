#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    unsigned long long local_points, local_points_inside = 0, total_points_inside = 0;
    unsigned long long i;
    double x, y, pi_estimate;
    double start_time, end_time;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (argc != 2) {
        if (rank == 0) {
            printf("Usage: %s <number of points per process>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }
    
    local_points = strtoull(argv[1], NULL, 10);
    if (local_points == 0) {
        if (rank == 0) {
            printf("Error: Number of points per process must be positive\n");
        }
        MPI_Finalize();
        return 1;
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    srand(time(NULL) + rank);
    
    start_time = MPI_Wtime();
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    for (i = 0; i < local_points; i++) {
        x = (2.0 * rand() / RAND_MAX) - 1.0;
        y = (2.0 * rand() / RAND_MAX) - 1.0;
        
        if (x*x + y*y <= 1.0) {
            local_points_inside++;
        }
    }
    
    MPI_Reduce(&local_points_inside, &total_points_inside, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    
    end_time = MPI_Wtime();
    
    if (rank == 0) {
        unsigned long long total_points = local_points * size;
        pi_estimate = 4.0 * total_points_inside / (double)total_points;
        printf("Number of processes: %d\n", size);
        printf("Points per process: %llu\n", local_points);
        printf("Total number of points (points per process * processes): %llu\n", total_points);
        printf("Total points inside circle: %llu\n", total_points_inside);
        printf("Estimate of pi: %f\n", pi_estimate);
        printf("Execution time: %f seconds\n", end_time - start_time);
    }
    
    MPI_Finalize();
    return 0;
}

