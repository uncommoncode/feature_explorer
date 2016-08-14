import sklearn.manifold
import sklearn.decomposition
import sklearn.discriminant_analysis
import sklearn.ensemble
import scipy.spatial.distance

def project_tsne_cossim(points, components=2):
    similarities = scipy.spatial.distance.squareform(
                    scipy.spatial.distance.pdist(points, "cosine")
                    )
    tsne = sklearn.manifold.TSNE(n_components=components, n_iter=1000, perplexity=30, metric="precomputed")
    return tsne.fit_transform(similarities)

def project_pca(points, components=2):
    pca = sklearn.decomposition.PCA(n_components=components)
    return pca.fit_transform(points)

def project_spectral_embedding(points, components=2):
    embedding = sklearn.manifold.SpectralEmbedding(n_components=2)
    return embedding.fit_transform(project_pca(points, 50))

def project_rf(points, components=2):
    hasher = sklearn.ensemble.RandomTreesEmbedding(n_estimators=10,
                                                   random_state=0,
                                                   max_depth=5)
    points_transformed = hasher.fit_transform(project_pca(points, 50))
    pca = sklearn.decomposition.TruncatedSVD(n_components=components)
    return pca.fit_transform(points_transformed)

PROJECTIONS = {
    "tsne_cossim": project_tsne_cossim,
    "pca": project_pca,
    "rf": project_rf,
    "spectral_embedding": project_spectral_embedding,
}

DEFAULT_PROJECTION = "tsne_cossim"
