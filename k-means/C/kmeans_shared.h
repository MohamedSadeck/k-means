#ifndef KMEANS_SHARED_H
#define KMEANS_SHARED_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Constants for the k-means algorithm
#define N 9        // Reduced to 9 points
#define K 3        // Keep 3 clusters
#define DIM 2      // Keep 2 dimensions
#define MAX_ITERS 100
#define EPSILON 1e-4

// Pre-generated data arrays - 3 clear clusters with 3 points each
static double data[N][DIM] = {
    {0.0, 0.0},  // Cluster 1 - bottom left
    {0.1, 0.1},
    {0.0, 0.1},
    
    {1.0, 1.0},  // Cluster 2 - top right
    {0.9, 1.0},
    {1.0, 0.9},
    
    {0.0, 1.0},  // Cluster 3 - top left
    {0.1, 0.9},
    {0.0, 0.9}
};

// Initial centroids - deliberately placed far from actual clusters
static double initial_centroids[K][DIM] = {
    {0.5, 0.5},  // Center
    {0.2, 0.8},  // Middle-top
    {0.8, 0.2}   // Middle-bottom
};

// Initialize data with fixed values (no random generation needed)
static void init_data() {
    // Do nothing - data is now statically initialized
}

// Allocate a 2D array
static double** allocate_2d(int rows, int cols) {
    double** arr = (double**)malloc(rows * sizeof(double*));
    for(int i = 0; i < rows; i++) {
        arr[i] = (double*)malloc(cols * sizeof(double));
    }
    return arr;
}

// Free a 2D array
static void free_2d(double** arr, int rows) {
    for(int i = 0; i < rows; i++) {
        free(arr[i]);
    }
    free(arr);
}

// Copy data to array
static void copy_data(double** dest) {
    for(int i = 0; i < N; i++) {
        for(int d = 0; d < DIM; d++) {
            dest[i][d] = data[i][d];
        }
    }
}

// Copy initial centroids
static void copy_initial_centroids(double** dest) {
    for(int i = 0; i < K; i++) {
        for(int d = 0; d < DIM; d++) {
            dest[i][d] = initial_centroids[i][d];
        }
    }
}

#endif
