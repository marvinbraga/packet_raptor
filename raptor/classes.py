import numpy as np

from raptor.clusterers import Clusterer
from raptor.embedders import Embedder
from raptor.nodes import SummaryNode
from raptor.summarizers import Summarizer
from raptor.text_chunkers import TextChunker


class Raptor:
    def __init__(self, chunker, embedder, clusterer, summarizer):
        self.chunker = chunker
        self.embedder = embedder
        self.clusterer = clusterer
        self.summarizer = summarizer

    def build_tree(self, text):
        chunks = self.chunker.split_text(text)
        labels = self.clusterer.cluster_embeddings(self.embedder)

        root = SummaryNode("Root")
        for cluster_id in set(labels):
            cluster_text = ' '.join([chunks[i] for i in range(len(chunks)) if labels[i] == cluster_id])
            summary = self.summarizer.summarize_text(cluster_text)
            root.add_child(SummaryNode(summary))

        return root

    def query_tree(self, query, root):
        embeddings = self.embedder.generate_embeddings([child.get_text() for child in root.get_children()])
        query_embedding = self.embedder.generate_embeddings([query])[0]

        similarities = [np.dot(query_embedding, emb) for emb in embeddings]
        best_summary_idx = np.argmax(similarities)

        return root.get_children()[best_summary_idx].get_text()


if __name__ == "__main__":
    with open('doc.md', 'r', encoding="utf-8") as f:
        text = f.readlines()

    embedder = Embedder().generate_embeddings()
    chunker = TextChunker(embedder)
    clusterer = Clusterer(num_clusters=5)
    summarizer = Summarizer()

    raptor = Raptor(chunker, embedder, clusterer, summarizer)
    tree = raptor.build_tree(text)

    query = "Qual é o tema central da história?"
    answer = raptor.query_tree(query, tree)
    print(answer)
