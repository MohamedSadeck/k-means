#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// Allocate a 2D array of doubles
double** allocate_2d(int rows, int cols) {
    double** arr = (double**)malloc(rows * sizeof(double*));
    for(int i = 0; i < rows; i++) {
        arr[i] = (double*)malloc(cols * sizeof(double));
    }
    return arr;
}

// Free a 2D array
void free_2d(double** arr, int rows) {
    for(int i = 0; i < rows; i++) {
        free(arr[i]);
    }
    free(arr);
}

// Generate some sample data for clustering
void generate_data(double** X, int total, int dim) {
    // For simplicity, create 3 clusters with random offsets
    int cluster_size = total / 3;
    srand(42);
    for(int i = 0; i < total; i++) {
        int c = i / cluster_size;
        double offsetX = (c == 0) ? 3.0 : ((c == 1) ? -2.0 : 0.0);
        double offsetY = (c == 0) ? 3.0 : ((c == 1) ? -2.0 : 0.0);
        X[i][0] = ((double)rand() / RAND_MAX) * 4 - 2 + offsetX;
        X[i][1] = ((double)rand() / RAND_MAX) * 4 - 2 + offsetY;
    }
}

// Pick random centroids
void initialize_centroids(double** X, double** centroids, int n, int k, int dim) {
    for(int i = 0; i < k; i++) {
        int idx = rand() % n;
        for(int d = 0; d < dim; d++) {
            centroids[i][d] = X[idx][d];
        }
    }
}

// Compute distances from each point to each centroid
void compute_distances(double** X, double** centroids, double** distances,
                       int n, int k, int dim) {
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < k; j++) {
            double sum = 0.0;
            for(int d = 0; d < dim; d++) {
                double diff = X[i][d] - centroids[j][d];
                sum += diff * diff;
            }
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

// Update centroids based on assigned clusters
void update_centroids(double** X, int* labels, double** centroids,
                      int n, int k, int dim) {
    int* counts = (int*)calloc(k, sizeof(int));
    for(int i = 0; i < k; i++) {
        for(int d = 0; d < dim; d++) {
            centroids[i][d] = 0.0;
        }
    }
    for(int i = 0; i < n; i++) {
        counts[labels[i]]++;
        for(int d = 0; d < dim; d++) {
            centroids[labels[i]][d] += X[i][d];
        }
    }
    for(int i = 0; i < k; i++) {
        if(counts[i] > 0) {
            for(int d = 0; d < dim; d++) {
                centroids[i][d] /= counts[i];
            }
        }
    }
    free(counts);
}

// K-means function
void k_means(double** X, int n, int k, int dim, int max_iters, int* labels) {
    double** centroids = allocate_2d(k, dim);
    double** distances = allocate_2d(n, k);
    initialize_centroids(X, centroids, n, k, dim);
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
        int converged = 1;
        for(int i = 0; i < k; i++) {
            for(int d = 0; d < dim; d++) {
                if(fabs(centroids[i][d] - old_centroids[i][d]) > 1e-6) {
                    converged = 0;
                    break;
                }
            }
            if(!converged) break;
        }
        if(converged) break;
    }
    // Print final centroids
    for(int i = 0; i < k; i++) {
        printf("Centroid %d: ", i);
        for(int d = 0; d < dim; d++) {
            printf("%.4f ", centroids[i][d]);
        }
        printf("\n");
    }
    free_2d(centroids, k);
    free_2d(distances, n);
}

// Main function
int main() {
    int n = 600;
    int dim = 2;
    int k = 3;
    int max_iters = 100;
    double** X = allocate_2d(n, dim);
    int* labels = (int*)malloc(n * sizeof(int));
    // Generate sample data
    generate_data(X, n, dim);
    // Run k-means
    k_means(X, n, k, dim, max_iters, labels);
    // Clean up
    free_2d(X, n);
    free(labels);
    return 0;
}
