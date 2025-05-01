import pandas as pd
import numpy as np
import os
from math import sqrt
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import random
import multiprocessing as mp
from scipy.spatial.distance import cdist
from itertools import repeat

# Randomly generating initial centroids
def choose_centroids(n, k):
    random.seed(42)  # for reproducibility
    return sorted(random.sample(range(n), k))

def distance(centroid, datapoint):
    return sqrt(np.square(centroid - datapoint).sum())

# Sequential K-Means implementation
def Kmeans(df, cidx, n, k):
    iterations = 0
    time_log = []
    centroids = df.loc[cidx].copy().astype(float)  # Ensure centroids are of type float
    clusters = np.zeros(n, dtype=int)

    while iterations < 100:
        start = timer()
        iterations += 1
        old_clusters = clusters.copy()

        for i in range(n):
            dist = [distance(centroids.iloc[j], df.iloc[i]) for j in range(k)]
            clusters[i] = np.argmin(dist)

        time_log.append(timer() - start)

        if np.array_equal(clusters, old_clusters):
            break

        for j in range(k):
            cluster_points = df.iloc[clusters == j]
            if not cluster_points.empty:
                # Ensure that the new centroid is a float
                centroids.iloc[j] = cluster_points.mean(axis=0).astype(float)

    return np.mean(time_log), clusters


# Parallel worker function
def find_cluster(data_chunk, centroids):
    distances = cdist(data_chunk, centroids, 'euclidean')
    return np.argmin(distances, axis=1)

# Parallel K-Means using multiprocessing
def mpKmeans(df, cidx, n, k, cpus):
    iterations = 0
    time_log = []
    centroids = df.iloc[cidx].copy().values  # Corrected to use iloc
    clusters = np.zeros(n, dtype=int)

    while iterations < 100:
        start = timer()
        iterations += 1
        old_clusters = clusters.copy()

        chunks = np.array_split(df.values, cpus)
        with mp.Pool(processes=cpus) as pool:
            results = pool.starmap(find_cluster, zip(chunks, repeat(centroids)))
        clusters = np.concatenate(results)

        time_log.append(timer() - start)

        for j in range(k):
            cluster_points = df.values[clusters == j]
            if cluster_points.size > 0:
                centroids[j] = np.mean(cluster_points, axis=0)

        if np.array_equal(clusters, old_clusters):
            break

    return np.mean(time_log), clusters

if __name__ == "__main__":  # Ensure this is the entry point
    df = pd.read_csv('data/Wholesale customers data.csv')
#    df = pd.read_csv('data/winequality-white.csv',sep=';')
#    df = df[df.columns[:8]]
    k_values = [3, 4, 5, 6, 7, 8]
    n = df.shape[0]
    print("Number of Datapoints =", n)
    print("Number of Attributes =", df.shape[1])

    time_kmeans = []
    time_mp = []
    iter_kmeans = []
    iter_mp = []
    diff_clusters = []

    for k in k_values:
        print(f"\nNumber of Clusters: {k}")
        cidx = choose_centroids(n, k)

        # Sequential KMeans
        start = timer()
        t_km, clus_km = Kmeans(df, cidx, n, k)
        total_km = timer() - start
        time_kmeans.append(total_km)
        iter_kmeans.append(t_km)
        print(f"KMeans:\tTotal Time: {total_km:.4f}\tMean Iteration Time: {t_km:.4f}")

        # Parallel KMeans
        cpus = mp.cpu_count()
        start = timer()
        t_mp, clus_mp = mpKmeans(df, cidx, n, k, cpus)
        total_mp = timer() - start
        time_mp.append(total_mp)
        iter_mp.append(t_mp)
        print(f"MP-KMeans:\tTotal Time: {total_mp:.4f}\tMean Iteration Time: {t_mp:.4f}")

        # Cluster difference
        diff = np.sum(clus_km != clus_mp)
        diff_clusters.append(diff)
        if diff != 0:
            print(f"Cluster Differences: {diff}")

    # Create 'result' directory if it doesn't exist
    os.makedirs('result', exist_ok=True)

    # Plot total execution time
    plt.plot(k_values, time_kmeans, 'r-', label="Sequential")
    plt.plot(k_values, time_mp, 'g-', label="Parallel")
    plt.title("Total Execution Time Comparison")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Execution Time (s)")
    plt.legend()
    plt.grid(True)
    plt.savefig('result/execution_time_comparison_sort_dataset.png')
    # plt.savefig('result/execution_time_comparison_large_dataset.png')
    plt.close()

    # Plot mean iteration time
    plt.plot(k_values, iter_kmeans, 'r--', label="Sequential")
    plt.plot(k_values, iter_mp, 'g--', label="Parallel")
    plt.title("Mean Iteration Time Comparison")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Avg Iteration Time (s)")
    plt.legend()
    plt.grid(True)
    plt.savefig('result/mean_iteration_time_comparison_sort_dataset.png')
    # plt.savefig('result/execution_time_comparison_large_dataset.png')
    plt.close()
