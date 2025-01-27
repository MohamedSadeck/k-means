from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import time

def plot_centroid_selection(data, centroids):
    """Plot the centroid selection process during k-means++ initialization."""
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], marker='.', color='gray', label='data points')
    if len(centroids) > 1:
        plt.scatter(centroids[:-1, 0], centroids[:-1, 1],
                   color='black', label='previously selected centroids')
    plt.scatter(centroids[-1, 0], centroids[-1, 1],
               color='red', label='next centroid')
    plt.title(f'Selecting centroid {len(centroids)}')
    plt.legend()
    plt.draw()
    plt.pause(1)
    plt.close()

def distance(p1, p2):
    """Compute Euclidean distance between two points."""
    return np.sqrt(np.sum((p1 - p2)**2))

def kmeans_plusplus_initialization(X, k):
    """Initialize centroids using k-means++ algorithm."""
    centroids = []
    # Choose first centroid randomly
    centroids.append(X[np.random.randint(X.shape[0])])
    plot_centroid_selection(X, np.array(centroids))

    # Choose remaining k-1 centroids
    for c_id in range(k - 1):
        # Calculate distances from points to nearest centroid
        dist = []
        for i in range(X.shape[0]):
            point = X[i]
            d = float('inf')
            
            # Find minimum distance to existing centroids
            for j in range(len(centroids)):
                temp_dist = distance(point, centroids[j])
                d = min(d, temp_dist)
            dist.append(d)

        # Select point with maximum distance as next centroid
        dist = np.array(dist)
        next_centroid = X[np.argmax(dist)]
        centroids.append(next_centroid)
        plot_centroid_selection(X, np.array(centroids))

    return np.array(centroids)

def plot_iteration(X, centroids, labels, iteration):
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o', alpha=0.5)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x')
    plt.title(f'Iteration {iteration}')
    plt.show()

snapshots = []
current_index = 0

def on_key(event):
    global current_index
    if event.key == 'right':
        current_index = min(current_index + 1, len(snapshots) - 1)
    elif event.key == 'left':
        current_index = max(current_index - 1, 0)
    
    plt.clf()
    X, labels, centroids = snapshots[current_index]
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.5)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200)
    plt.title(f"Iteration {current_index}")
    plt.draw()

def show_snapshots():
    fig, _ = plt.subplots(figsize=(10, 6))
    fig.canvas.mpl_connect('key_press_event', on_key)
    if snapshots:
        X, labels, centroids = snapshots[0]
        plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.5)
        plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200)
        plt.title("Iteration 0")
    plt.show()

def parallel_k_means(X, k, max_iters=100):
    start_time = time.time()
    global snapshots
    snapshots = []  # Reset snapshots

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Master initializes data and broadcasts shapes
    if rank == 0:
        data_shape = X.shape
        centroids = kmeans_plusplus_initialization(X, k)
        print(f"[Rank {rank}] Starting K-means++ clustering with {k} clusters")
        print(f"[Rank {rank}] Data shape: {data_shape}")
    else:
        data_shape = None
        X = None
        centroids = None

    # Broadcast shapes and centroids
    data_shape = comm.bcast(data_shape, root=0)
    centroids = comm.bcast(centroids, root=0)

    # Scatter the data among processes
    local_data_size = data_shape[0] // size
    remainder = data_shape[0] % size
    counts = [local_data_size + 1 if p < remainder else local_data_size for p in range(size)]
    displacements = [sum(counts[:p]) for p in range(size)]
    
    local_data = np.zeros((counts[rank], data_shape[1]), dtype=np.float64)
    
    if rank == 0:
        for p in range(size):
            if p != 0:
                comm.Send(X[displacements[p]:displacements[p] + counts[p]].astype(np.float64), dest=p)
            else:
                local_data = X[displacements[p]:displacements[p] + counts[p]].astype(np.float64)
    else:
        comm.Recv(local_data, source=0)

    converged = False
    for iteration in range(max_iters):
        # Compute local assignments
        distances = np.array([np.linalg.norm(local_data - centroid, axis=1) for centroid in centroids])
        labels_local = np.argmin(distances, axis=0)

        # Compute local sums and counts
        local_sums = np.zeros_like(centroids, dtype=np.float64)
        local_counts = np.zeros(k, dtype=np.int32)
        for i in range(k):
            mask = labels_local == i
            local_sums[i] = np.sum(local_data[mask], axis=0)
            local_counts[i] = np.sum(mask)

        # Gather global sums and counts
        global_sums = np.zeros_like(centroids, dtype=np.float64)
        global_counts = np.zeros(k, dtype=np.int32)
        comm.Reduce(local_sums, global_sums, op=MPI.SUM, root=0)
        comm.Reduce(local_counts, global_counts, op=MPI.SUM, root=0)

        # Master updates centroids
        if rank == 0:
            old_centroids = centroids.copy()
            for i in range(k):
                if global_counts[i] > 0:
                    centroids[i] = global_sums[i] / global_counts[i]
            
            converged = np.allclose(centroids, old_centroids, rtol=1e-6)
            
            # Gather all labels for visualization
            all_labels = np.zeros(data_shape[0], dtype=np.int32)
            displacements = [sum(counts[:p]) for p in range(size)]
            
            # First, copy local labels
            all_labels[displacements[0]:displacements[0] + counts[0]] = labels_local
            
            # Receive labels from other processes
            for p in range(1, size):
                recv_labels = comm.recv(source=p, tag=100)
                all_labels[displacements[p]:displacements[p] + counts[p]] = recv_labels
            
            # Save snapshot
            snapshots.append((X.copy(), all_labels.copy(), centroids.copy()))
            
            # Print progress
            print(f"[Rank {rank}] Iteration {iteration}: Inertia = {np.sum(np.min(distances, axis=0)):.2f}")
        else:
            # Send labels to root
            comm.send(labels_local, dest=0, tag=100)

        # Broadcast convergence status and new centroids
        converged = comm.bcast(converged, root=0)
        centroids = comm.bcast(centroids, root=0)
        
        if converged:
            if rank == 0:
                print(f"[Rank {rank}] Converged after {iteration + 1} iterations!")
            break

    if rank == 0:
        print(f"Total time: {time.time() - start_time:.2f} seconds")
        print("\nUse left/right arrow keys to navigate through iterations")
        print("Close the plot window to see the final result")
        show_snapshots()
        
        # Show final result
        plt.figure(figsize=(10, 6))
        X, labels, centroids = snapshots[-1]
        plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
        plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', c='red', s=200, linewidth=3)
        plt.title('Final Clustering Result')
        plt.show()
        
        return labels, centroids
    return None

if __name__ == "__main__":  # Fixed syntax
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        # Generate sample data using the provided distribution parameters
        np.random.seed(42)
        mean_01 = np.array([0.0, 0.0])
        cov_01 = np.array([[1, 0.3], [0.3, 1]])
        dist_01 = np.random.multivariate_normal(mean_01, cov_01, 100)

        mean_02 = np.array([6.0, 7.0])
        cov_02 = np.array([[1.5, 0.3], [0.3, 1]])
        dist_02 = np.random.multivariate_normal(mean_02, cov_02, 100)

        mean_03 = np.array([7.0, -5.0])
        cov_03 = np.array([[1.2, 0.5], [0.5, 1]])
        dist_03 = np.random.multivariate_normal(mean_03, cov_01, 100)

        mean_04 = np.array([2.0, -7.0])
        cov_04 = np.array([[1.2, 0.5], [0.5, 1.3]])
        dist_04 = np.random.multivariate_normal(mean_04, cov_01, 100)

        X = np.vstack((dist_01, dist_02, dist_03, dist_04))
        np.random.shuffle(X)
    else:
        X = None
    
    k = 4
    result = parallel_k_means(X, k)
    
    if rank == 0:
        labels, centroids = result
        plt.figure(figsize=(10, 6))
        plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
        plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', c='red', s=200, linewidth=3)
        plt.title('Final Clustering Result')
        plt.show()