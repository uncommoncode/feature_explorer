import sklearn.cluster
import sklearn.metrics

def landmark_kmeans(points, count):
    kmeans = sklearn.cluster.MiniBatchKMeans(count)
    kmeans.fit(points)
    landmark_indices, _ = sklearn.metrics.pairwise_distances_argmin_min(kmeans.cluster_centers_, points)
    return landmark_indices

LANDMARKS = {
    "kmeans": landmark_kmeans,
}

DEFAULT_LANDMARK = "kmeans"
