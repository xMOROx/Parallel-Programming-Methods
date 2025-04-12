#ifndef BUCKETS_H
#define BUCKETS_H

#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

#define MAX_RANGE 1000000000 // Maksymalny zakres generowanych wartości

typedef int li_t; // Typ przechowywanych danych

// --------------------
// Struktura kubełka (bucket_t)
// --------------------
typedef struct bucket {
    li_t *data;       // Tablica przechowująca elementy
    size_t capacity;  // Maksymalna pojemność – brak mechanizmu realokacji!
    size_t next;      // Liczba zapisanych elementów (indeks następnego zapisu)
} bucket_t;

// Inicjalizuje kubełek, alokując pamięć o zadanej pojemności.
static inline void bucket_init(bucket_t *bucket, size_t initial_capacity) {
    bucket->data = (li_t *)malloc(initial_capacity * sizeof(li_t));
    if (bucket->data == NULL) {
        perror("bucket_init: malloc failed");
        exit(EXIT_FAILURE);
    }
    bucket->capacity = initial_capacity;
    bucket->next = 0;
}

// Resetuje kubełek – umożliwia ponowne użycie zaalokowanej pamięci.
static inline void bucket_reset(bucket_t *bucket) {
    bucket->next = 0;
}

// Dodaje nową wartość do kubełka.
// Jeśli liczba elementów przekroczy przydzieloną pojemność, kończymy działanie z błędem.
static inline void bucket_add(bucket_t *bucket, li_t value) {
    size_t index;
    #pragma omp atomic capture
    index = bucket->next++;
    
    if (index >= bucket->capacity) {
        fprintf(stderr, "Bucket capacity exceeded: index = %zu, capacity = %zu\n", index, bucket->capacity);
        exit(EXIT_FAILURE);
    }
    bucket->data[index] = value;
}

// Zwraca liczbę zapisanych elementów w kubełku.
static inline size_t bucket_get_size(const bucket_t *bucket) {
    size_t current_size;
    #pragma omp atomic read
    current_size = bucket->next;
    return current_size;
}

// Zwalnia pamięć przydzieloną dla kubełka.
static inline void bucket_destroy(bucket_t *bucket) {
    free(bucket->data);
    bucket->data = NULL;
    bucket->capacity = 0;
    bucket->next = 0;
}

// ----------------------------
// Struktura kolekcji kubełków (bucket_collection_t)
// ----------------------------
typedef struct bucket_collection {
    bucket_t *buckets;
    size_t num_buckets;
} bucket_collection_t;

// Inicjalizuje kolekcję kubełków – każdy kubełek otrzymuje przydział pamięci o zadanej początkowej pojemności.
static inline void bucket_collection_init(bucket_collection_t *collection, size_t num_buckets, size_t initial_capacity) {
    collection->buckets = (bucket_t *)malloc(num_buckets * sizeof(bucket_t));
    if (collection->buckets == NULL) {
        perror("bucket_collection_init: malloc failed");
        exit(EXIT_FAILURE);
    }
    collection->num_buckets = num_buckets;
    for (size_t i = 0; i < num_buckets; i++) {
        bucket_init(&collection->buckets[i], initial_capacity);
    }
}

// Zwalnia pamięć kolekcji kubełków, wywołując bucket_destroy dla każdego kubełka.
static inline void bucket_collection_destroy(bucket_collection_t *collection) {
    for (size_t i = 0; i < collection->num_buckets; i++) {
        bucket_destroy(&collection->buckets[i]);
    }
    free(collection->buckets);
    collection->buckets = NULL;
    collection->num_buckets = 0;
}

#endif // BUCKETS_H
