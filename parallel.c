#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Allocate a 2D array dynamically
double** allocate_2d(int rows, int cols) {
    double** array = (double**)malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++) {
        array[i] = (double*)malloc(cols * sizeof(double));
    }
    return array;
}

// Free a 2D array
void free_2d(double** array, int rows) {
    for (int i = 0; i < rows; i++) {
        free(array[i]);
    }
    free(array);
}

void compute_distances(double** X, double** centroids, double* distances,
                       int local_count, int k, int dim) {
    for (int i = 0; i < local_count; i++) {
        for (int j = 0; j < k; j++) {
            double sum = 0.0;
            for (int d = 0; d < dim; d++) {
                double diff = X[i][d] - centroids[j][d];
                sum += diff * diff;
            }
            distances[i * k + j] = sqrt(sum);
        }
    }
}

void assign_clusters(double* distances, int* labels_local,
                     int local_count, int k) {
    for (int i = 0; i < local_count; i++) {
        double min_distance = distances[i * k];
        labels_local[i] = 0;
        for (int j = 1; j < k; j++) {
            if (distances[i * k + j] < min_distance) {
                min_distance = distances[i * k + j];
                labels_local[i] = j;
            }
        }
    }
}

void update_centroids(double** sums, int* counts, double** centroids,
                      int k, int dim) {
    for (int i = 0; i < k; i++) {
        if (counts[i] > 0) {
            for (int d = 0; d < dim; d++) {
                centroids[i][d] = sums[i][d] / counts[i];
            }
        }
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = 600, dim = 2, k = 3, max_iters = 100;
    double** X = NULL;
    int* labels = NULL;

    if (rank == 0) {
        X = allocate_2d(n, dim);
        labels = (int*)malloc(n * sizeof(int));

        // Generate random data for X
        for (int i = 0; i < n; i++) {
            for (int d = 0; d < dim; d++) {
                X[i][d] = (double)rand() / RAND_MAX;
            }
        }
    }

    int base_count = n / size;
    int remainder = n % size;
    int* counts = (int*)malloc(size * sizeof(int));
    int* displs = (int*)malloc(size * sizeof(int));
    for (int i = 0; i < size; i++) {
        counts[i] = base_count + (i < remainder ? 1 : 0);
    }
    displs[0] = 0;
    for (int i = 1; i < size; i++) {
        displs[i] = displs[i - 1] + counts[i - 1];
    }

    double** X_local = allocate_2d(counts[rank], dim);
    double* sendbuf = NULL;
    double* recvbuf = (double*)malloc(counts[rank] * dim * sizeof(double));
    if (rank == 0) {
        sendbuf = (double*)malloc(n * dim * sizeof(double));
        for (int i = 0; i < n; i++) {
            for (int d = 0; d < dim; d++) {
                sendbuf[i * dim + d] = X[i][d];
            }
        }
    }

    int* sendcounts = (int*)malloc(size * sizeof(int));
    int* senddispls = (int*)malloc(size * sizeof(int));
    for (int i = 0; i < size; i++) {
        sendcounts[i] = counts[i] * dim;
        senddispls[i] = displs[i] * dim;
    }
    MPI_Scatterv(sendbuf, sendcounts, senddispls, MPI_DOUBLE,
                 recvbuf, counts[rank] * dim, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    for (int i = 0; i < counts[rank]; i++) {
        for (int d = 0; d < dim; d++) {
            X_local[i][d] = recvbuf[i * dim + d];
        }
    }

    double** centroids = allocate_2d(k, dim);
    if (rank == 0) {
        for (int i = 0; i < k; i++) {
            for (int d = 0; d < dim; d++) {
                centroids[i][d] = X[i][d];
            }
        }
        printf("Initial centroids:\n");
        for (int i = 0; i < k; i++) {
            printf("Centroid %d: ", i);
            for (int d = 0; d < dim; d++) {
                printf("%.2f ", centroids[i][d]);
            }
            printf("\n");
        }
    }

    for (int i = 0; i < k; i++) {
        MPI_Bcast(centroids[i], dim, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    double* distances = (double*)malloc(counts[rank] * k * sizeof(double));
    int* labels_local = (int*)malloc(counts[rank] * sizeof(int));
    for (int iter = 0; iter < max_iters; iter++) {
        compute_distances(X_local, centroids, distances,
                          counts[rank], k, dim);
        assign_clusters(distances, labels_local, counts[rank], k);

        double** partial_sums = allocate_2d(k, dim);
        int* local_counts = (int*)calloc(k, sizeof(int));
        for (int i = 0; i < counts[rank]; i++) {
            int cluster = labels_local[i];
            local_counts[cluster]++;
            for (int d = 0; d < dim; d++) {
                partial_sums[cluster][d] += X_local[i][d];
            }
        }

        double** global_sums = allocate_2d(k, dim);
        int* global_counts = (int*)calloc(k, sizeof(int));
        for (int i = 0; i < k; i++) {
            MPI_Reduce(partial_sums[i], global_sums[i], dim, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Reduce(&local_counts[i], &global_counts[i], 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        }

        if (rank == 0) {
            update_centroids(global_sums, global_counts, centroids, k, dim);
            printf("Iteration %d centroids:\n", iter + 1);
            for (int i = 0; i < k; i++) {
                printf("Centroid %d: ", i);
                for (int d = 0; d < dim; d++) {
                    printf("%.2f ", centroids[i][d]);
                }
                printf("\n");
            }
        }

        for (int i = 0; i < k; i++) {
            MPI_Bcast(centroids[i], dim, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }
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

    free_2d(X_local, counts[rank]);
    free_2d(centroids, k);
    free(distances);
    free(labels_local);
    free(recvbuf);
    if (rank == 0) {
        free_2d(X, n);
        free(labels);
        free(sendbuf);
    }

    MPI_Finalize();
    return 0;
}
