from sklearn.cluster import KMeans


class Clusterer:
    def __init__(self, num_clusters):
        self.num_clusters = num_clusters

    def cluster_embeddings(self, embeddings):
        # Verifique se os embeddings não estão vazios
        if embeddings.size == 0:
            raise ValueError("Nenhum embedding válido encontrado para clusterização.")

        # Verifique se embeddings é um array 1D e, se necessário, redimensione-o
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(-1, 1)

        # Prossiga com a clusterização usando KMeans
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=0)
        try:
            kmeans.fit(embeddings)
        except ValueError as e:
            raise ValueError(f"Erro durante a clusterização: {e}")

        # Inicialize clusters com base nos rótulos
        clusters = [[] for _ in range(self.num_clusters)]
        for idx, label in enumerate(kmeans.labels_):
            clusters[label].append(idx)

        return clusters
