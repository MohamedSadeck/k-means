from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt

def parallel_k_means(X, k, max_iters=100):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Master initializes data and centroids
    if rank == 0:
        # ...code to generate or load data X if needed...
        centroids = X[np.random.choice(X.shape[0], k, replace=False)]
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
    local_data = np.zeros((counts[rank], 2), dtype='float64')  # adjust shape
    if rank == 0:
        for p in range(size):
            if p != 0:
                comm.Send(X[starts[p]:starts[p]+counts[p]], dest=p, tag=77)
        local_data = X[starts[0]:starts[0]+counts[0]]
    else:
        comm.Recv(local_data, source=0, tag=77)

    for iteration in range(max_iters):
        # Compute cluster assignments locally
        distances = np.zeros((local_data.shape[0], k))
        for i, c in enumerate(centroids):
            distances[:, i] = np.linalg.norm(local_data - c, axis=1)
        labels_local = np.argmin(distances, axis=1)

        # Partial sums for each cluster
        partial_sums = np.zeros((k, local_data.shape[1]))
        counts_local = np.zeros(k, dtype=int)
        for idx, label in enumerate(labels_local):
            partial_sums[label] += local_data[idx]
            counts_local[label] += 1

        # Reduce to get global sums and counts
        global_sums = np.zeros_like(partial_sums)
        global_counts = np.zeros_like(counts_local)
        comm.Reduce(partial_sums, global_sums, op=MPI.SUM, root=0)
        comm.Reduce(counts_local, global_counts, op=MPI.SUM, root=0)

        # Master updates centroids
        if rank == 0:
            for i in range(k):
                if global_counts[i] > 0:
                    centroids[i] = global_sums[i] / global_counts[i]
            print(f"[Rank {rank}] Iteration {iteration} updated centroids:\n", centroids)

        # Broadcast new centroids
        centroids = comm.bcast(centroids, root=0)

    # Gather labels from each process
    all_labels = None
    comm.Barrier()
    labels_local = labels_local.astype(np.int32)  # ensure int32
    if rank == 0:
        all_labels = np.empty(num_points, dtype=np.int32)
    comm.Gatherv(labels_local, (all_labels, counts, starts, MPI.INT), root=0)

    if rank == 0:
        print("[Rank 0] Final centroids:\n", centroids)
        print("[Rank 0] Final labels:\n", all_labels)
        plt.scatter(X[:, 0], X[:, 1], c=all_labels)
        plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x')
        plt.title("Final clustering result")
        plt.show()
    return centroids

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank == 0:
        np.random.seed(42)
        X = np.vstack((np.random.randn(100, 2) + np.array([4, 4]),
                       np.random.randn(100, 2) + np.array([-4, -4]),
                       np.random.randn(100, 2)))
    else:
        X = None

    final_centroids = parallel_k_means(X, k=3, max_iters=10)