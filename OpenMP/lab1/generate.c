#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#define TO_STRING(x) #x
#define SCHEDULE_STR(x) TO_STRING(x)


typedef struct {
    long long size;
    int *data;
} vector_t;

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "[ERROR] Usage: %s <num_threads> <array_size>\n", argv[0]);
        fprintf(stderr, "[INFO] To set chunk size, compile with -DCHUNK_SIZE=<size>, e.g., -DCHUNK_SIZE=1024\n");
        fprintf(stderr, "[INFO] To set schedule type, compile with -DSCHEDULE_TYPE=<type>, e.g., -DSCHEDULE_TYPE=static\n");
        fprintf(stderr, "[INFO] To enable nowait, compile with -DNOWAIT\n");
        return 1;
    }
    int n_threads = atoi(argv[1]);
    long long m_size = atoll(argv[2]);
    if (n_threads <= 0 || m_size <= 0) {
        fprintf(stderr, "[ERROR] Number of threads and array size must be positive.\n");
        return 1;
    }
    FILE *csv_file = NULL;
    const char *csv_filename = "results.csv";
    csv_file = fopen(csv_filename, "w");
    if (csv_file == NULL) {
        fprintf(stderr, "[ERROR] Could not open file %s for writing.\n", csv_filename);
        return 1;
    } else {
        fprintf(csv_file, "p,n,time\n");
        fflush(csv_file);
    }
    vector_t vector;
    vector.size = m_size;
    vector.data = (int *)malloc(m_size * sizeof(int));
    if (vector.data == NULL) {
        fprintf(stderr, "[ERROR] Failed to allocate memory for the array (size %lld).\n", m_size);
        if (csv_file != NULL) fclose(csv_file);
        return 1;
    }
    omp_set_num_threads(n_threads);
    double start_time, end_time, elapsed_time;
    start_time = omp_get_wtime();
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        unsigned short xsubi[3];
        xsubi[0] = (unsigned short)(tid * 11 + 17);
        xsubi[1] = (unsigned short)(tid * 31 + 37);
        xsubi[2] = (unsigned short)(tid * 51 + 57);
        #ifdef CHUNK_SIZE
            #ifdef NOWAIT
            #pragma omp for schedule(SCHEDULE_TYPE, CHUNK_SIZE) nowait
            #else
            #pragma omp for schedule(SCHEDULE_TYPE, CHUNK_SIZE)
            #endif
        #else
            #pragma omp for
        #endif
        for (long long i = 0; i < vector.size; ++i) {
            vector.data[i] = nrand48(xsubi);
        }
    }
    end_time = omp_get_wtime();
    elapsed_time = end_time - start_time;
    printf("[INFO] Successfully filled array of size %lld with random numbers using %d threads.\n", vector.size, n_threads);
    #ifdef CHUNK_SIZE
        printf("[INFO] Used schedule: %s with chunk size = %d", SCHEDULE_STR(SCHEDULE_TYPE), CHUNK_SIZE);
        #ifdef NOWAIT
            printf(" (with nowait)\n");
        #else
            printf(" (without nowait)\n");
        #endif
    #else
        printf("[INFO] Used default OpenMP schedule (CHUNK_SIZE macro was not defined).\n");
    #endif
    printf("[INFO] Parallel section execution time: %f seconds.\n", elapsed_time);
    if (csv_file != NULL) {
        fprintf(csv_file, "%d,%lld,%f\n", n_threads, vector.size, elapsed_time);
        fflush(csv_file);
        fclose(csv_file);
        printf("[INFO] Result saved to file: %s\n", csv_filename);
    }
    printf("[DEBUG] First few elements:\n");
    long long print_count = (vector.size < 10) ? vector.size : 10;
    for (long long i = 0; i < print_count; ++i) {
        printf("[DEBUG] data[%lld] = %d\n", i, vector.data[i]);
    }
    if (vector.size > 10) {
        printf("[DEBUG] ...\n");
    }
    free(vector.data);
    return 0;
}

