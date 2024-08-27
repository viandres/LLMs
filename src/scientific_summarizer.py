import os
import logging
from typing import List
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers.bm25 import BM25Retriever
from langchain_openai import AzureChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document, HumanMessage
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Load environment variables from .env file
load_dotenv()

# Setting up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Retrieve environment variables directly
API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
DEPLOYMENT_NAME = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')

if not API_KEY or not ENDPOINT or not DEPLOYMENT_NAME:
    raise ValueError("API key, endpoint, and deployment name must be set in environment variables")

class BM25Vectorizer:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.vectorizer = TfidfVectorizer(norm=None, smooth_idf=False)
        self.average_document_length = None
        self.doc_len = None
        self.doc_vectors = None

    def fit(self, documents: List[str]):
        tfidf_matrix = self.vectorizer.fit_transform(documents)
        self.doc_vectors = np.asarray(tfidf_matrix.todense())
        self.doc_len = np.array([len(doc.split()) for doc in documents])
        self.average_document_length = np.mean(self.doc_len)

    def transform_query(self, query: str):
        query_vector = self.vectorizer.transform([query])
        return np.asarray(query_vector.todense())[0]

    def get_top_n(self, query_vector, docs: List[Document], n=10):
        scores = self.calculate_scores(query_vector)
        top_n_indices = np.argsort(scores)[-n:][::-1]
        return [docs[i] for i in top_n_indices]

    def calculate_scores(self, query_vector):
        scores = []
        for i, doc_vector in enumerate(self.doc_vectors):
            score = self._score(query_vector, doc_vector, i)
            scores.append(score)
        return np.array(scores)

    def _score(self, query_vector, doc_vector, doc_index):
        score = 0.0
        for idx, q in enumerate(query_vector):
            if q == 0:
                continue
            df = np.sum(self.doc_vectors[:, idx] > 0)
            idf = np.log((len(self.doc_vectors) - df + 0.5) / (df + 0.5) + 1)
            tf = doc_vector[idx]
            denominator = tf + self.k1 * (1 - self.b + self.b * self.doc_len[doc_index] / self.average_document_length)
            score += idf * (tf * (self.k1 + 1)) / denominator
        return score

class DocumentProcessor:
    """Handles loading and processing of documents for summarization."""

    def __init__(self, pdf_path: Path):
        self.pdf_path = pdf_path

    def load_documents(self) -> List[Document]:
        logger.info(f"Loading documents from {self.pdf_path}")
        loader = PyPDFLoader(str(self.pdf_path))
        documents = loader.load()
        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        logger.info("Splitting documents into chunks")
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        return splitter.split_documents(documents)

class ScientificSummarizer:
    """Responsible for retrieving document chunks and generating summaries."""

    def __init__(self):
        self.llm = self.initialize_llm()

    def initialize_llm(self) -> AzureChatOpenAI:
        logger.info("Initializing Azure OpenAI Chat LLM")
        return AzureChatOpenAI(
            api_key=API_KEY,
            azure_deployment=DEPLOYMENT_NAME,
            azure_endpoint=ENDPOINT,
            model="gpt-4",  # Ensure this matches your deployment's model name
        )

    def retrieve_relevant_chunks(self, chunks: List[Document], topic: str) -> List[Document]:
        logger.info(f"Retrieving relevant chunks for topic: {topic}")
        documents_content = [chunk.page_content for chunk in chunks]

        vectorizer = BM25Vectorizer()
        vectorizer.fit(documents_content)

        retriever = BM25Retriever(docs=chunks, vectorizer=vectorizer)
        return retriever.invoke(topic)  # Use the invoke method to retrieve relevant documents

    def generate_summaries_with_references(self, chunks: List[Document]) -> List[str]:
        logger.info("Generating summaries with references")
        referenced_summaries = []
        for chunk in chunks:
            message = HumanMessage(content=f"Summarize the following text:\n{chunk.page_content}\nInclude references to the source.")
            response = self.llm.invoke([message])  # Use invoke method instead of __call__
            summary = response.content  # Correctly access the content of the AIMessage
            reference = f"Source: {chunk.metadata.get('source', 'Unknown Source')}"
            summary_with_reference = f"{summary}\n\nReference: {reference}"
            referenced_summaries.append(summary_with_reference)
        return referenced_summaries

def write_summaries_to_markdown(summaries: List[str], output_file: Path) -> None:
    logger.info(f"Writing summaries to Markdown file: {output_file}")
    with output_file.open('w', encoding='utf-8') as md_file:
        md_file.write("# Summarized Content with References\n\n")
        for i, summary in enumerate(summaries, start=1):
            md_file.write(f"## Summary {i}\n\n")
            md_file.write(summary)
            md_file.write("\n\n---\n\n")

def main():
    # Build the absolute path to the PDF file using pathlib
    base_dir = Path(__file__).resolve().parent.parent
    pdf_path = base_dir / 'data' / '2024_Visum_Blaue_Karte_DE.pdf'

    if not pdf_path.exists():
        logger.error(f"PDF file does not exist: {pdf_path}")
        return

    # Initialize document processor
    document_processor = DocumentProcessor(pdf_path=pdf_path)

    # Load and process documents
    documents = document_processor.load_documents()
    chunks = document_processor.split_documents(documents)

    # Initialize scientific summarizer
    summarizer = ScientificSummarizer()

    # Retrieve relevant chunks and generate summaries
    topic = "Getting Visa"
    relevant_chunks = summarizer.retrieve_relevant_chunks(chunks, topic)
    summaries = summarizer.generate_summaries_with_references(relevant_chunks)

    # Write the summaries to a Markdown file
    output_file = base_dir / 'summarized_content.md'
    write_summaries_to_markdown(summaries, output_file)
    logger.info(f"Summaries written to {output_file}")

if __name__ == "__main__":
    main()
