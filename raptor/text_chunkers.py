from langchain_experimental.text_splitter import SemanticChunker


class TextChunker:
    def __init__(self, embedding):
        self.embedding = embedding
        self.tokenizer = SemanticChunker(self.embedding)

    def split_text(self, text):
        tokens = self.tokenizer.create_documents(text)
        return tokens
