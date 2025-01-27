#include "kmeans_shared.h"
#include <mpi.h>

// Function to perform k-means clustering in parallel
void k_means_parallel(double** X, int n, int k, int dim, int max_iters, 
                     int* labels, double** centroids, int rank, int size) {
    int* counts = (int*)malloc(k * sizeof(int));
    double** new_centroids = (double**)malloc(k * sizeof(double*));
    for (int i = 0; i < k; i++) {
        new_centroids[i] = (double*)malloc(dim * sizeof(double));
    }
    
    double epsilon = 1e-4;  // Match sequential version
    
    if (rank == 0) {
        printf("Initial centroids:\n");
        for (int i = 0; i < k; i++) {
            printf("Centroid %d: ", i);
            for (int d = 0; d < dim; d++) {
                printf("%.2f ", centroids[i][d]);
            }
            printf("\n");
        }
    }

    for(int iter = 0; iter < max_iters; iter++) {
        // Save old centroids for convergence check
        double old_centroids[k][dim];
        for(int i = 0; i < k; i++) {
            for(int d = 0; d < dim; d++) {
                old_centroids[i][d] = centroids[i][d];
            }
        }

        // Initialize new centroids and counts
        for (int i = 0; i < k; i++) {
            counts[i] = 0;
            for (int d = 0; d < dim; d++) {
                new_centroids[i][d] = 0.0;
            }
        }

        // Assign points to the nearest centroid
        for (int i = rank; i < n; i += size) {
            int nearest = 0;
            double min_dist = INFINITY;
            for (int j = 0; j < k; j++) {
                double dist = 0.0;
                for (int d = 0; d < dim; d++) {
                    dist += (X[i][d] - centroids[j][d]) * (X[i][d] - centroids[j][d]);
                }
                if (dist < min_dist) {
                    min_dist = dist;
                    nearest = j;
                }
            }
            labels[i] = nearest;
            counts[nearest]++;
            for (int d = 0; d < dim; d++) {
                new_centroids[nearest][d] += X[i][d];
            }
        }

        // Reduce new centroids and counts across all processes
        MPI_Allreduce(MPI_IN_PLACE, counts, k, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        for (int i = 0; i < k; i++) {
            MPI_Allreduce(MPI_IN_PLACE, new_centroids[i], dim, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        }

        // Update centroids
        for (int i = 0; i < k; i++) {
            for (int d = 0; d < dim; d++) {
                if (counts[i] > 0) {
                    centroids[i][d] = new_centroids[i][d] / counts[i];
                }
            }
        }

        if (rank == 0) {
            printf("Iteration %d centroids:\n", iter + 1);
            for (int i = 0; i < k; i++) {
                printf("Centroid %d: ", i);
                for (int d = 0; d < dim; d++) {
                    printf("%.2f ", centroids[i][d]);
                }
                printf("\n");
            }
        }

        // Check convergence using max change
        double local_max_change = 0.0;
        double global_max_change = 0.0;
        
        for(int i = 0; i < k; i++) {
            for(int d = 0; d < dim; d++) {
                double diff = centroids[i][d] - old_centroids[i][d];
                if(fabs(diff) > local_max_change) {
                    local_max_change = fabs(diff);
                }
            }
        }
        
        MPI_Allreduce(&local_max_change, &global_max_change, 1, MPI_DOUBLE, 
                      MPI_MAX, MPI_COMM_WORLD);
        
        if(global_max_change < epsilon) break;
    }

    if (rank == 0) {
        printf("Final centroids:\n");
        for (int i = 0; i < k; i++) {
            printf("Centroid %d: ", i);
            for (int d = 0; d < dim; d++) {
                printf("%.2f ", centroids[i][d]);
            }
            printf("\n");
        }
    }

    for (int i = 0; i < k; i++) {
        free(new_centroids[i]);
    }
    free(new_centroids);
    free(counts);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double** X = allocate_2d(N, DIM);
    double** centroids = allocate_2d(K, DIM);
    int* labels = (int*)malloc(N * sizeof(int));

    if (rank == 0) {
        // Initialize data using shared function
        init_data();
        copy_data(X);
        copy_initial_centroids(centroids);
    }

    // Broadcast data to all processes
    for (int i = 0; i < N; i++) {
        MPI_Bcast(X[i], DIM, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    for (int i = 0; i < K; i++) {
        MPI_Bcast(centroids[i], DIM, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // Run parallel k-means
    k_means_parallel(X, N, K, DIM, MAX_ITERS, labels, centroids, rank, size);

    // Cleanup
    free_2d(X, N);
    free_2d(centroids, K);
    free(labels);

    MPI_Finalize();
    return 0;
}
