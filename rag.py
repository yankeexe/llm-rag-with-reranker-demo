import os
import tempfile


#https://stackoverflow.com/questions/76958817/streamlit-your-system-has-an-unsupported-version-of-sqlite3-chroma-requires-sq
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import json
import chromadb
import ollama

from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from streamlit.runtime.uploaded_file_manager import UploadedFile

class Rag:
    # https://www.sbert.net/docs/cross_encoder/pretrained_models.html
    cross_encoder="cross-encoder/ms-marco-MiniLM-L-6-v2"
    chromadbpath="demo-rag"

    #https://docs.trychroma.com/guides#changing-the-distance-function
    embedding_distance_function="cosine"
    embedding="nomic-embed-text:latest"
    model="llama3.2:3b"


    original_system_prompt = """
    You are an AI assistant tasked with providing detailed answers based solely on the given context. Your goal is to analyze the information provided and formulate a comprehensive, well-structured response to the question.

    context will be passed as "Context:"
    user question will be passed as "Question:"

    To answer the question:
    1. Thoroughly analyze the context, identifying key information relevant to the question.
    2. Organize your thoughts and plan your response to ensure a logical flow of information.
    3. Formulate a detailed answer that directly addresses the question, using only the information provided in the context.
    4. Ensure your answer is comprehensive, covering all relevant aspects found in the context.
    5. If the context doesn't contain sufficient information to fully answer the question, state this clearly in your response.

    Format your response as follows:
    1. Use clear, concise language.
    2. Organize your answer into paragraphs for readability.
    3. Use bullet points or numbered lists where appropriate to break down complex information.
    4. If relevant, include any headings or subheadings to structure your response.
    5. Ensure proper grammar, punctuation, and spelling throughout your answer.

    Important: Base your entire response solely on the information provided in the context. Do not include any external knowledge or assumptions not present in the given text.
    """


    def process_document(self,uploaded_file: UploadedFile) -> list[Document]:
        """Processes an uploaded PDF file by converting it to text chunks.

        Takes an uploaded PDF file, saves it temporarily, loads and splits the content
        into text chunks using recursive character splitting.

        Args:
            uploaded_file: A Streamlit UploadedFile object containing the PDF file

        Returns:
            A list of Document objects containing the chunked text from the PDF

        Raises:
            IOError: If there are issues reading/writing the temporary file
        """
        # Store uploaded file as a temp file
        temp_file = tempfile.NamedTemporaryFile("wb", suffix=".pdf", delete=False)
        temp_file.write(uploaded_file.read())

        loader = PyMuPDFLoader(temp_file.name)
        docs = loader.load()
        os.unlink(temp_file.name)  # Delete temp file

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=100,
            # bruno: just use dots as separator
            #separators=["\n\n", "\n", ".", "?", "!", " ", ""],
            separators=["."],
        )
        return text_splitter.split_documents(docs)


    def get_vector_collection(self) -> chromadb.Collection:
        """Gets or creates a ChromaDB collection for vector storage.

        Creates an Ollama embedding function using the nomic-embed-text model and initializes
        a persistent ChromaDB client. Returns a collection that can be used to store and
        query document embeddings.

        Returns:
            chromadb.Collection: A ChromaDB collection configured with the Ollama embedding
                function and cosine similarity space.
        """
        ollama_ef = OllamaEmbeddingFunction(
            url="http://localhost:11434/api/embeddings",
            model_name=self.embedding
        )
        print("using distance function:",self.embedding_distance_function)

        chroma_client = chromadb.PersistentClient(path="./"+self.chromadbpath+"-chroma")
        return chroma_client.get_or_create_collection(
            name="rag_app",
            embedding_function=ollama_ef,
            metadata={"hnsw:space": self.embedding_distance_function},
        )


    def add_to_vector_collection(self,all_splits: list[Document], file_name: str):
        """Adds document splits to a vector collection for semantic search.

        Takes a list of document splits and adds them to a ChromaDB vector collection
        along with their metadata and unique IDs based on the filename.

        Args:
            all_splits: List of Document objects containing text chunks and metadata
            file_name: String identifier used to generate unique IDs for the chunks

        Returns:
            None. Displays a success message via Streamlit when complete.

        Raises:
            ChromaDBError: If there are issues upserting documents to the collection
        """
        collection = self.get_vector_collection()
        documents, metadatas, ids = [], [], []

        for idx, split in enumerate(all_splits):
            documents.append(split.page_content)
            metadatas.append(split.metadata)
            ids.append(f"{file_name}_{idx}")

        collection.upsert(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )
        


    def query_collection(self,prompt: str, n_results: int = 10):
        """Queries the vector collection with a given prompt to retrieve relevant documents.

        Args:
            prompt: The search query text to find relevant documents.
            n_results: Maximum number of results to return. Defaults to 10.

        Returns:
            dict: Query results containing documents, distances and metadata from the collection.

        Raises:
            ChromaDBError: If there are issues querying the collection.
        """
        collection = self.get_vector_collection()
        results = collection.query(query_texts=[prompt], n_results=n_results)
        return results


    def call_llm(self,context: str, prompt: str,system_prompt:str):
        """Calls the language model with context and prompt to generate a response.

        Uses Ollama to stream responses from a language model by providing context and a
        question prompt. The model uses a system prompt to format and ground its responses appropriately.

        Args:
            context: String containing the relevant context for answering the question
            prompt: String containing the user's question

        Yields:
            String chunks of the generated response as they become available from the model

        Raises:
            OllamaError: If there are issues communicating with the Ollama API
        """
        response = ollama.chat(
            model=self.model,
            stream=True,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": f"Context: {context}, Question: {prompt}",
                },
            ],
        )
        for chunk in response:
            if chunk["done"] is False:
                yield chunk["message"]["content"]
            else:
                break


    def re_rank_cross_encoders(self,prompt:str,documents: list[str]) -> tuple[str, list[int]]:
        """Re-ranks documents using a cross-encoder model for more accurate relevance scoring.

        Uses the MS MARCO MiniLM cross-encoder model to re-rank the input documents based on
        their relevance to the query prompt. Returns the concatenated text of the top 3 most
        relevant documents along with their indices.

        Args:
            documents: List of document strings to be re-ranked.

        Returns:
            tuple: A tuple containing:
                - relevant_text (str): Concatenated text from the top 3 ranked documents
                - relevant_text_ids (list[int]): List of indices for the top ranked documents

        Raises:
            ValueError: If documents list is empty
            RuntimeError: If cross-encoder model fails to load or rank documents
        """
        relevant_text = ""
        relevant_text_ids = []
        if self.cross_encoder != "NONE":
            encoder_model = CrossEncoder(self.cross_encoder)
            ranks = encoder_model.rank(prompt, documents, top_k=3)
            for rank in ranks:
                relevant_text += documents[rank["corpus_id"]]
                relevant_text_ids.append(rank["corpus_id"])
        else:
            for i in range(0,3):
                relevant_text += documents[i]
                relevant_text_ids.append(i)
                
        return relevant_text, relevant_text_ids

    # UI

    #system_prompt = original_system_prompt
    def splitAndStore(self,uploaded_file):
        normalizedFileName= uploaded_file.name.translate(
                    str.maketrans({"-": "_", ".": "_", " ": "_"})
                )
        all_splits = self.process_document(uploaded_file)

        # for debug: dump the splits as json
        strs=list(map(lambda x:x.page_content,all_splits))
        with open(normalizedFileName+"-splits.json", "w") as save_file:  
            json.dump(strs, save_file, indent = 6)  


        self.add_to_vector_collection(all_splits, normalizedFileName)
    