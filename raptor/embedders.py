from langchain_community.embeddings import HuggingFaceEmbeddings


class Embedder:
    def __init__(self):
        self.model = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

    def generate_embeddings(self):
        return self.model
