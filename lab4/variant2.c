#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "buckets.h"

#ifndef CHUNK
#define CHUNK 40000
#endif

#ifndef BUCKETS_NUMBER
#define BUCKETS_NUMBER 1024000
#endif

#ifndef CAPACITY_MULTIPLIER
#define CAPACITY_MULTIPLIER 2.0
#endif

// Comparison function for qsort – ascending order
int compare_li_t(const void *a, const void *b) {
    li_t arg1 = *(const li_t*)a;
    li_t arg2 = *(const li_t*)b;
    return (arg1 > arg2) - (arg1 < arg2);
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "[ERROR] Usage: %s <num_threads> <num_elements>\n", argv[0]);
        fprintf(stderr, "[INFO] You can override defaults with: -DCHUNK=%d -DBUCKETS_NUMBER=%d -DCAPACITY_MULTIPLIER=%.1f\n", CHUNK, BUCKETS_NUMBER, CAPACITY_MULTIPLIER);
        return 1;
    }

    int n_threads = atoi(argv[1]);
    size_t num_elements = atoll(argv[2]);
    if (n_threads <= 0 || num_elements <= 0) {
        fprintf(stderr, "[ERROR] Number of threads and elements must be positive.\n");
        return 1;
    }

    omp_set_num_threads(n_threads);
    int num_buckets = BUCKETS_NUMBER;

    size_t initial_capacity = (num_elements / num_buckets) + 1;
    initial_capacity = (size_t)(initial_capacity * CAPACITY_MULTIPLIER);
    printf("[INFO] Initial capacity of each bucket: %zu\n", initial_capacity);
    bucket_collection_t bucket_collection;
    bucket_collection_init(&bucket_collection, num_buckets, initial_capacity);

    li_t *initial_array = (li_t *)malloc(num_elements * sizeof(li_t));
    if (initial_array == NULL) {
        fprintf(stderr, "[ERROR] Memory allocation failed for initial array.\n");
        exit(EXIT_FAILURE);
    }

    double overall_start = omp_get_wtime();
    double t_phase, time_gen, time_dist, time_sort, time_copy;

    // Phase (a): Number generation
    t_phase = omp_get_wtime();
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        unsigned short state[3] = {
            (unsigned short)(tid * 11 + 17),
            (unsigned short)(tid * 31 + 37),
            (unsigned short)(tid * 51 + 57)
        };
        #pragma omp for schedule(guided, CHUNK)
        for (size_t i = 0; i < num_elements; i++) {
            initial_array[i] = nrand48(state) % MAX_RANGE;
        }
    }
    time_gen = omp_get_wtime() - t_phase;

    // Phase (b): Distribution
    t_phase = omp_get_wtime();
    #pragma omp parallel for schedule(guided, CHUNK)
    for (size_t i = 0; i < num_elements; i++) {
        li_t value = initial_array[i];
        int bucket_index = (int)(((size_t)value * num_buckets) / MAX_RANGE);
        if (bucket_index >= num_buckets) bucket_index = num_buckets - 1;
        bucket_add(&bucket_collection.buckets[bucket_index], value);
    }
    time_dist = omp_get_wtime() - t_phase;

    // Phase (c): Sorting
    t_phase = omp_get_wtime();
    #pragma omp parallel for schedule(guided, CHUNK)
    for (int b = 0; b < num_buckets; b++) {
        size_t bucket_size = bucket_get_size(&bucket_collection.buckets[b]);
        qsort(bucket_collection.buckets[b].data, bucket_size, sizeof(li_t), compare_li_t);
    }
    time_sort = omp_get_wtime() - t_phase;

    // Phase (d): Copying
    t_phase = omp_get_wtime();
    size_t *offsets = (size_t *)malloc(num_buckets * sizeof(size_t));
    if (offsets == NULL) {
        fprintf(stderr, "[ERROR] Memory allocation failed for offsets.\n");
        exit(EXIT_FAILURE);
    }
    offsets[0] = 0;
    for (int b = 1; b < num_buckets; b++) {
        offsets[b] = offsets[b - 1] + bucket_get_size(&bucket_collection.buckets[b - 1]);
    }
    #pragma omp parallel for schedule(guided, CHUNK)
    for (int b = 0; b < num_buckets; b++) {
        size_t bucket_size = bucket_get_size(&bucket_collection.buckets[b]);
        memcpy(&initial_array[offsets[b]], bucket_collection.buckets[b].data, bucket_size * sizeof(li_t));
    }
    time_copy = omp_get_wtime() - t_phase;

    double overall_end = omp_get_wtime();
    double time_overall = overall_end - overall_start;

    double check_is_sorted_start, check_is_sorted_end;

    check_is_sorted_start = omp_get_wtime();
    // Validate sorted array
    int is_sorted = 1;
    for (size_t i = 1; i < num_elements; i++) {
        if (initial_array[i] < initial_array[i - 1]) {
            printf("[DEBUG] initial_array[%d] = %d, initial_array[%d] = %d\n", i - 1, initial_array[i - 1], i, initial_array[i]);
            is_sorted = 0;
            break;
        }
    }
    check_is_sorted_end = omp_get_wtime();

    // Print results
    printf("[INFO] Time - number generation:    %f seconds\n", time_gen);
    printf("[INFO] Time - distribution:         %f seconds\n", time_dist);
    printf("[INFO] Time - bucket sorting:       %f seconds\n", time_sort);
    printf("[INFO] Time - copying buckets:      %f seconds\n", time_copy);
    printf("[INFO] Time - check sorted:         %f seconds\n", check_is_sorted_end - check_is_sorted_start);
    printf("[INFO] Total execution time:        %f seconds\n", time_overall);
    printf("[INFO] Number of buckets:           %d\n", num_buckets);
    printf("[INFO] Number of elements:          %lld\n", num_elements);
    printf("[INFO] Number of threads:           %d\n", n_threads);
    printf("[INFO] Array is sorted:             %s\n", is_sorted ? "YES" : "NO");

    // Optional preview
    int print_count = (num_elements < 10) ? num_elements : 10;
    printf("[DEBUG] First %d elements of the sorted array:\n", print_count);
    for (int i = 0; i < print_count; i++) {
        printf("initial_array[%d] = %d\n", i, initial_array[i]);
    }

    // Save to CSV
    FILE *fp = fopen("results.csv", "a");
    if (fp == NULL) {
        fprintf(stderr, "[ERROR] Could not open results.csv for writing.\n");
    } else {
        fprintf(fp, "Threads,Elements,Buckets,GenTime,DistTime,SortTime,CopyTime,TotalTime,IsSorted\n");
        fprintf(fp, "%d,%lld,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%s\n",
                n_threads, num_elements, num_buckets,
                time_gen, time_dist, time_sort, time_copy, time_overall,
                is_sorted ? "YES" : "NO");
        fclose(fp);
    }

    // Cleanup
    free(offsets);
    free(initial_array);
    bucket_collection_destroy(&bucket_collection);

    return 0;
}
