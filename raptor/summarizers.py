from dotenv import load_dotenv, find_dotenv
from langchain.chains.summarize import load_summarize_chain
from langchain_openai import ChatOpenAI

load_dotenv(find_dotenv())


class Summarizer:
    def __init__(self, model_name="gpt-3.5-turbo"):
        self.llm = ChatOpenAI(
            temperature=0,
            model_name=model_name,
        )
        self.chain = load_summarize_chain(self.llm, chain_type="stuff")

    def summarize_text(self, text):
        summary = self.chain.invoke({"input": text})
        return summary["output_text"]
