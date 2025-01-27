from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import time

def kmeans_plusplus_initialization(X, k):
    centroids = np.empty((k, X.shape[1]))
    centroids[0] = X[np.random.choice(X.shape[0])]
    
    for i in range(1, k):
        distances = np.min([np.linalg.norm(X - centroid, axis=1) for centroid in centroids[:i]], axis=0)
        probabilities = distances / np.sum(distances)
        centroids[i] = X[np.random.choice(X.shape[0], p=probabilities)]
    
    return centroids

def plot_iteration(X, centroids, labels, iteration):
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o', alpha=0.5)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x')
    plt.title(f'Iteration {iteration}')
    plt.show()

def parallel_k_means(X, k, max_iters=100):
    start_time = time.time()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Master initializes data and centroids using K-Means++ initialization
    if rank == 0:
        centroids = kmeans_plusplus_initialization(X, k)
        print(f"[Rank {rank}] Initial centroids:\n", centroids)
    else:
        X = None
        centroids = None

    # Broadcast data dimensions and centroids
    centroids = comm.bcast(centroids, root=0)

    # Scatter the data among processes
    num_points = 0 if rank != 0 else X.shape[0]
    num_points = comm.bcast(num_points, root=0)
    local_data_size = num_points // size
    remainder = num_points % size
    counts = [local_data_size + 1 if p < remainder else local_data_size for p in range(size)]
    starts = [sum(counts[:p]) for p in range(size)]
    local_data = np.zeros((counts[rank], X.shape[1]), dtype='float64')
    if rank == 0:
        for p in range(size):
            if p != 0:
                comm.Send(X[starts[p]:starts[p] + counts[p]], dest=p)
            else:
                local_data = X[starts[p]:starts[p] + counts[p]]
    else:
        comm.Recv(local_data, source=0)

    for iteration in range(max_iters):
        # Compute local assignments
        distances = np.array([np.linalg.norm(local_data - centroid, axis=1) for centroid in centroids])
        labels_local = np.argmin(distances, axis=0)

        # Compute local sums and counts
        local_sums = np.zeros_like(centroids)
        local_counts = np.zeros(k, dtype=int)
        for i in range(k):
            local_sums[i] = np.sum(local_data[labels_local == i], axis=0)
            local_counts[i] = np.sum(labels_local == i)

        # Gather global sums and counts
        global_sums = np.zeros_like(centroids)
        global_counts = np.zeros(k, dtype=int)
        comm.Reduce(local_sums, global_sums, op=MPI.SUM, root=0)
        comm.Reduce(local_counts, global_counts, op=MPI.SUM, root=0)

        # Master updates centroids
        if rank == 0:
            old_centroids = centroids.copy()
            for i in range(k):
                if global_counts[i] > 0:
                    centroids[i] = global_sums[i] / global_counts[i]
            print(f"[Rank {rank}] Iteration {iteration} updated centroids:\n", centroids)
            if np.allclose(centroids, old_centroids, rtol=1e-6):
                break

            # Plot the current iteration
            plot_iteration(X, centroids, np.concatenate(comm.gather(labels_local, root=0)), iteration)

            # Prompt to hop to the next iteration
            input("Press Enter to continue to the next iteration...")

        # Broadcast new centroids
        centroids = comm.bcast(centroids, root=0)

    # Gather labels from each process
    all_labels = None
    comm.Barrier()
    labels_local = labels_local.astype(np.int32)  # ensure int32
    if rank == 0:
        all_labels = np.empty(num_points, dtype=np.int32)
    comm.Gather(labels_local, all_labels, root=0)

    if rank == 0:
        print(f"Total time: {time.time() - start_time} seconds")
        return all_labels, centroids

if __name__ == "_main_":
    # Example usage
    X = np.random.rand(100, 2)  # Replace with actual data loading
    k = 3
    labels, centroids = parallel_k_means(X, k)
    if MPI.COMM_WORLD.Get_rank() == 0:
        plt.scatter(X[:, 0], X[:, 1], c=labels)
        plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', c='red')
        plt.show()