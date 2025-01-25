import numpy as np
import time
import matplotlib.pyplot as plt

def initialize_centroids(X, k):
    # Randomly select k data points as initial centroids
    indices = np.random.choice(X.shape[0], k, replace=False)
    return X[indices]

def compute_distances(X, centroids):
    # Compute the Euclidean distance between each data point and each centroid
    distances = np.zeros((X.shape[0], len(centroids)))
    for i, centroid in enumerate(centroids):
        distances[:, i] = np.linalg.norm(X - centroid, axis=1)
    return distances

def assign_clusters(distances):
    # Assign each data point to the nearest centroid
    return np.argmin(distances, axis=1)

def update_centroids(X, labels, k):
    # Compute the new centroids as the mean of the data points in each cluster
    centroids = np.zeros((k, X.shape[1]))
    for i in range(k):
        centroids[i] = X[labels == i].mean(axis=0)
    return centroids

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
    plt.scatter(X[:,0], X[:,1], c=labels)
    plt.scatter(centroids[:,0], centroids[:,1], c='red', marker='x')
    plt.title(f"Iteration {current_index}")
    plt.draw()

def show_snapshots():
    fig, _ = plt.subplots()
    fig.canvas.mpl_connect('key_press_event', on_key)
    if snapshots:
        X, labels, centroids = snapshots[0]
        plt.scatter(X[:,0], X[:,1], c=labels)
        plt.scatter(centroids[:,0], centroids[:,1], c='red', marker='x')
        plt.title("Iteration 0")
    plt.show()

def k_means(X, k, max_iters=100):
    start_time = time.time()
    # Initialize centroids
    centroids = initialize_centroids(X, k)
    
    for i in range(max_iters):
        old_centroids = centroids
        
        # Compute distances and assign clusters
        distances = compute_distances(X, centroids)
        labels = assign_clusters(distances)
        
        # Update centroids
        centroids = update_centroids(X, labels, k)
        
        snapshots.append((X.copy(), labels.copy(), centroids.copy()))
        
        # Check for convergence
        if np.all(centroids == old_centroids):
            break
    
    print(f"Elapsed time: {time.time() - start_time:.4f} seconds")
    return centroids, labels

# Example usage
if __name__ == "__main__":
    # Generate some sample data
    np.random.seed(42)
    X = np.vstack((
        np.random.randn(400, 2)*2 + np.array([3, 3]),
        np.random.randn(400, 2)*2 + np.array([-2, -2]),
        np.random.randn(400, 2)*2
    ))

    # Perform k-means clustering
    k = 3
    centroids, labels = k_means(X, k)

    print("Final centroids:\n", centroids)
    print("Labels:\n", labels)
    
    show_snapshots()