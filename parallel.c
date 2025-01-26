#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// ...existing code for allocate_2d, free_2d, or generate_data...
// (Refer to sequential.c for those helpers if needed)

void compute_distances(double** X, double** centroids, double* distances,
                       int local_count, int k, int dim) {
    // ...existing code logic adapted...
    for(int i = 0; i < local_count; i++) {
        for(int j = 0; j < k; j++) {
            double sum = 0.0;
            // ...existing code to compute distance...
            distances[i*k + j] = sqrt(sum);
        }
    }
}

void assign_clusters(double* distances, int* labels_local,
                     int local_count, int k) {
    // ...existing code logic adapted...
}

void update_centroids(double** sums, int* counts, double** centroids,
                      int k, int dim) {
    for(int i = 0; i < k; i++) {
        if(counts[i] > 0) {
            for(int d = 0; d < dim; d++) {
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

    if(rank == 0) {
        // ...existing code to allocate and generate data...
        // e.g. X = allocate_2d(n, dim); generate_data(X, n, dim);
        labels = (int*)malloc(n * sizeof(int));
    }

    // Distribute data counts
    int base_count = n / size;
    int remainder = n % size;
    int* counts = (int*)malloc(size * sizeof(int));
    int* displs = (int*)malloc(size * sizeof(int));
    for(int i = 0; i < size; i++) {
        counts[i] = base_count + (i < remainder ? 1 : 0);
    }
    displs[0] = 0;
    for(int i = 1; i < size; i++) {
        displs[i] = displs[i-1] + counts[i-1];
    }

    // Allocate local data
    double** X_local = NULL;
    int local_count = counts[rank];
    // ...allocate X_local with local_count rows...

    // Prepare contiguous buffer for scatter/gather
    double* sendbuf = NULL;
    double* recvbuf = (double*)malloc(local_count * dim * sizeof(double));
    if(rank == 0) {
        sendbuf = (double*)malloc(n * dim * sizeof(double));
        // Flatten X into sendbuf
        // ...existing code to copy X into sendbuf...
    }

    // Scatterv data
    int* sendcounts = (int*)malloc(size*sizeof(int));
    int* senddispls = (int*)malloc(size*sizeof(int));
    for(int i = 0; i < size; i++) {
        sendcounts[i] = counts[i]*dim;
        senddispls[i] = displs[i]*dim;
    }
    MPI_Scatterv(sendbuf, sendcounts, senddispls, MPI_DOUBLE,
                 recvbuf, local_count*dim, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    // Rebuild X_local from recvbuf
    // ...existing code to map recvbuf to X_local...

    // Initialize centroids on rank 0
    double** centroids = NULL;
    if(rank == 0) {
        centroids = /* allocate_2d(k, dim) */;
        // ...existing code to pick random initial centroids...
    } 
    else {
        // ...allocate for centroids...
    }
    // Broadcast centroids
    for(int i = 0; i < k; i++) {
        MPI_Bcast(centroids[i], dim, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // Iteration loop
    double* distances = (double*)malloc(local_count*k*sizeof(double));
    int* labels_local = (int*)malloc(local_count*sizeof(int));
    for(int iter = 0; iter < max_iters; iter++) {
        // Store old centroids for convergence
        double* old = (double*)malloc(k*dim*sizeof(double));
        // ...copy centroids to old...

        compute_distances(X_local, centroids, distances,
                          local_count, k, dim);
        assign_clusters(distances, labels_local, local_count, k);

        // Partial sums
        double** partial_sums = /* allocate_2d(k, dim) */;
        int* local_counts = (int*)calloc(k, sizeof(int));
        // ...accumulate partial sums for each cluster...

        // Reduce sums and counts on rank 0
        double** global_sums = NULL;
        int* global_counts = NULL;
        if(rank == 0) {
            global_sums = /* allocate_2d(k, dim) */;
            global_counts = (int*)calloc(k, sizeof(int));
        }
        for(int i = 0; i < k; i++) {
            MPI_Reduce(partial_sums[i], (rank==0?global_sums[i]:NULL),
                       dim, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Reduce(&local_counts[i], (rank==0?&global_counts[i]:NULL),
                       1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        }

        // Rank 0 updates centroids
        if(rank == 0) {
            update_centroids(global_sums, global_counts, centroids, k, dim);
            // Check convergence
            int converged = 1;
            // ...compare centroids with old...
            if(converged) {
                // ...free local memory...
                // broadcast break signal if desired
            }
        }

        // Broadcast updated centroids
        for(int i = 0; i < k; i++) {
            MPI_Bcast(centroids[i], dim, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }
        // ...free local temporary allocations...
    }

    // Gather labels
    int* all_labels = NULL;
    if(rank == 0) {
        all_labels = (int*)malloc(n*sizeof(int));
    }
    MPI_Gatherv(labels_local, local_count, MPI_INT,
                all_labels, counts, displs, MPI_INT,
                0, MPI_COMM_WORLD);

    if(rank == 0) {
        // ...print final centroids, or handle results...
        // free_2d(X); free(labels);
        free(all_labels);
    }
    // ...cleanup local arrays, etc...

    MPI_Finalize();
    return 0;
}
