#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "kmeans_shared.h"

// Modify compute_distances to match parallel version
void compute_distances(double** X, double** centroids, double** distances,
                       int n, int k, int dim) {
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < k; j++) {
            double sum = 0.0;
            for(int d = 0; d < dim; d++) {
                double diff = X[i][d] - centroids[j][d];
                sum += diff * diff;
            }
            // Store in row-major order like parallel version
            distances[i][j] = sqrt(sum);
        }
    }
}

// Assign cluster labels
void assign_clusters(double** distances, int* labels, int n, int k) {
    for(int i = 0; i < n; i++) {
        double minDist = distances[i][0];
        int minIndex = 0;
        for(int j = 1; j < k; j++) {
            if(distances[i][j] < minDist) {
                minDist = distances[i][j];
                minIndex = j;
            }
        }
        labels[i] = minIndex;
    }
}

// Modify update_centroids to match parallel version's logic
void update_centroids(double** X, int* labels, double** centroids,
                      int n, int k, int dim) {
    double** sums = allocate_2d(k, dim);
    int* counts = (int*)calloc(k, sizeof(int));
    
    // First zero out the sums
    for(int i = 0; i < k; i++) {
        for(int d = 0; d < dim; d++) {
            sums[i][d] = 0.0;
        }
    }
    
    // Accumulate sums like parallel version
    for(int i = 0; i < n; i++) {
        int cluster = labels[i];
        counts[cluster]++;
        for(int d = 0; d < dim; d++) {
            sums[cluster][d] += X[i][d];
        }
    }
    
    // Update centroids same as parallel
    for(int i = 0; i < k; i++) {
        if(counts[i] > 0) {
            for(int d = 0; d < dim; d++) {
                centroids[i][d] = sums[i][d] / counts[i];
            }
        }
    }
    
    free_2d(sums, k);
    free(counts);
}

// K-means function
void k_means(double** X, int n, int k, int dim, int max_iters, int* labels, double** centroids) {
    double** distances = allocate_2d(n, k);
    double epsilon = 1e-4;  // Match parallel version epsilon

    // Print initial centroids
    printf("Initial centroids:\n");
    for (int i = 0; i < k; i++) {
        printf("Centroid %d: ", i);
        for (int d = 0; d < dim; d++) {
            printf("%.2f ", centroids[i][d]);
        }
        printf("\n");
    }

    for(int iter = 0; iter < max_iters; iter++) {
        double old_centroids[k][dim];
        for(int i = 0; i < k; i++) {
            for(int d = 0; d < dim; d++) {
                old_centroids[i][d] = centroids[i][d];
            }
        }
        
        compute_distances(X, centroids, distances, n, k, dim);
        assign_clusters(distances, labels, n, k);
        update_centroids(X, labels, centroids, n, k, dim);

        // Print iteration centroids
        printf("Iteration %d centroids:\n", iter + 1);
        for (int i = 0; i < k; i++) {
            printf("Centroid %d: ", i);
            for (int d = 0; d < dim; d++) {
                printf("%.2f ", centroids[i][d]);
            }
            printf("\n");
        }

        // Check convergence using max change like parallel version
        double max_change = 0.0;
        for(int i = 0; i < k; i++) {
            for(int d = 0; d < dim; d++) {
                double diff = centroids[i][d] - old_centroids[i][d];
                if(fabs(diff) > max_change) {
                    max_change = fabs(diff);
                }
            }
        }
        if(max_change < epsilon) break;
    }

    // Print final centroids
    printf("Final centroids:\n");
    for (int i = 0; i < k; i++) {
        printf("Centroid %d: ", i);
        for (int d = 0; d < dim; d++) {
            printf("%.2f ", centroids[i][d]);
        }
        printf("\n");
    }

    free_2d(distances, n);
}

// Main function
int main() {
    double** X = allocate_2d(N, DIM);
    int* labels = (int*)malloc(N * sizeof(int));
    double** centroids = allocate_2d(K, DIM);

    // Initialize data using shared function
    init_data();
    copy_data(X);
    copy_initial_centroids(centroids);

    // Run k-means
    k_means(X, N, K, DIM, MAX_ITERS, labels, centroids);

    // Clean up
    free_2d(centroids, K);
    free_2d(X, N);
    free(labels);
    return 0;
}
