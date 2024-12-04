
# This is just the UI 
from rag import *
import streamlit as st

st.set_page_config(page_title="RAG Question Answer")

print("new Rag...")
rag=Rag()




# Question and Answer Area
st.header("🗣️ RAG Question Answer")


# Document Upload Area
with st.sidebar:
   

    rag.chromadbpath = st.text_input("db (this let you experiment with loading different documents):","demo-rag")

    rag.embedding = st.selectbox(
    "embedding (https://ollama.com/search?c=embedding)",
    ("nomic-embed-text:latest",
     "mxbai-embed-large"
    ))


    rag.embedding_distance_function = st.selectbox(
    "embeddings distance function (https://docs.trychroma.com/guides#changing-the-distance-function)",
    ("cosine", "l2", "ip"),
    )

    uploaded_file = st.file_uploader(
        "Upload PDF ", type=["pdf"], accept_multiple_files=False
    )

    process = st.button(
        "⚡️ Process",
    )
    if uploaded_file and process:
        rag.splitAndStore(uploaded_file)
        st.success("Data added to the vector store!")





rag.model=st.selectbox(
    "LLM model (https://ollama.com/search)",(
    "llama3.2:3b",
    "qwen2"
    ))

rag.cross_encoder = st.selectbox(
    """cross encoder is used to rerank documents before feeding to LLM  (https://www.sbert.net/docs/cross_encoder/pretrained_models.html)""",
   ("cross-encoder/ms-marco-MiniLM-L-6-v2",
    "NONE",
    "cross-encoder/ms-marco-TinyBERT-L-2-v2",
    "cross-encoder/ms-marco-MiniLM-L-2-v2",
    "cross-encoder/ms-marco-MiniLM-L-4-v2",
#    "cross-encoder/ms-marco-MiniLM-L-6-v2",
     "cross-encoder/ms-marco-MiniLM-L-12-v2",
    "cross-encoder/qnli-distilroberta-base",
    "cross-encoder/qnli-electra-base",
    "cross-encoder/stsb-TinyBERT-L-4",
    "cross-encoder/stsb-distilroberta-base",
    "cross-encoder/stsb-roberta-base",
    "cross-encoder/stsb-roberta-large",
    "cross-encoder/quora-distilroberta-base",
    "cross-encoder/quora-roberta-base",
    "cross-encoder/quora-roberta-large",
    "cross-encoder/nli-deberta-v3-base",
    "cross-encoder/nli-deberta-base",
    "cross-encoder/nli-deberta-v3-xsmall",
    "cross-encoder/nli-deberta-v3-small",
    "cross-encoder/nli-roberta-base",
    "cross-encoder/nli-MiniLM2-L6-H768",
    "cross-encoder/nli-distilroberta-base",
    "BAAI/bge-reranker-base",
    "BAAI/bge-reranker-large",
    "BAAI/bge-reranker-v2-m3",
    "BAAI/bge-reranker-v2-gemma",
    "BAAI/bge-reranker-v2-minicpm-layerwise",
    "jinaai/jina-reranker-v1-tiny-en",
    "jinaai/jina-reranker-v1-turbo-en",
    "mixedbread-ai/mxbai-rerank-xsmall-v1",
    "mixedbread-ai/mxbai-rerank-base-v1",
    "mixedbread-ai/mxbai-rerank-large-v1",
    "maidalun1020/bce-reranker-base_v1"

     )
)


system_prompt = st.text_area("system prompt:",value=rag.original_system_prompt)

prompt = st.text_area("prompt:")
ask = st.button(
    "🔥 Ask",
)

if ask and prompt:
    results = rag.query_collection(prompt)
    context = results.get("documents")[0]
    relevant_text, relevant_text_ids = rag.re_rank_cross_encoders(prompt,context)
    response = rag.call_llm(context=relevant_text, prompt=prompt,system_prompt=system_prompt)
    st.write_stream(response)

    with st.expander("See retrieved documents"):
        st.write(results)

    with st.expander("See most relevant document ids"):
        st.write(relevant_text_ids)
        st.write(relevant_text)
