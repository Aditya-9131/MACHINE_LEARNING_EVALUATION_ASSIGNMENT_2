import numpy as np
import matplotlib.pyplot as plt
data = np.array([
    [500, 10, 50],
    [1500, 15, 100],
    [200, 5, 40],
    [800, 20, 40],
    [300, 8, 37.5],
    [600, 12, 50],
    [1000, 10, 100],
    [450, 9, 50],
    [1200, 14, 85.71],
    [900, 7, 128.57]
])
def manhattan_distance(point1, point2):
    return np.sum(np.abs(point1 - point2))
def assign_clusters(data, medoids):
    clusters = {}
    for idx, point in enumerate(data):
        distances = [manhattan_distance(point, medoid) for medoid in medoids]
        nearest_medoid = np.argmin(distances)
        if nearest_medoid in clusters:
            clusters[nearest_medoid].append(idx)
        else:
            clusters[nearest_medoid] = [idx]
    return clusters
def update_medoids(clusters, data):
    new_medoids = []
    for cluster in clusters.values():
        cluster_points = data[cluster]
        distance_matrix = np.array([[manhattan_distance(p1, p2) for p2 in cluster_points] for p1 in cluster_points])
        total_distances = np.sum(distance_matrix, axis=1)
        new_medoids.append(data[cluster[np.argmin(total_distances)]])
    return np.array(new_medoids)

def k_medoids(data, k, max_iterations=100):
    medoids_indices = np.random.choice(len(data), k, replace=False)
    medoids = data[medoids_indices]
    
    for iteration in range(max_iterations):
        clusters = assign_clusters(data, medoids)
        new_medoids = update_medoids(clusters, data)
        if np.array_equal(new_medoids, medoids):
            break
        medoids = new_medoids
    
    return medoids, clusters
def calculate_sse(data, medoids, clusters):
    sse = 0
    for medoid_idx, cluster in clusters.items():
        for idx in cluster:
            sse += np.sum((data[idx] - medoids[medoid_idx]) ** 2)
    return sse
k = 3
medoids, clusters = k_medoids(data, k)
sse = calculate_sse(data, medoids, clusters)

print("Final Medoids:")
print(medoids)
print("\nCluster Assignment:")
for medoid_idx, cluster in clusters.items():
    print(f"Medoid {medoid_idx + 1}: Customers {cluster}")

print(f"\nSum of Squared Errors (SSE): {sse}")

def plot_clusters(data, medoids, clusters):
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']  # Assign different colors to each cluster
    plt.figure(figsize=(8, 6))

    for medoid_idx, cluster in clusters.items():
        cluster_points = data[cluster]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors[medoid_idx], label=f'Cluster {medoid_idx+1}')
    
    plt.scatter(medoids[:, 0], medoids[:, 1], c='black', marker='X', s=200, label='Medoids')
    plt.xlabel('Total Spending (in USD)')
    plt.ylabel('Number of Transactions')
    plt.title('K-Medoids Clustering of Customers')
    plt.legend()
    plt.grid(True)
    plt.show()
plot_clusters(data, medoids, clusters)
