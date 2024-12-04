import os
import tempfile

import json.scanner


#https://stackoverflow.com/questions/76958817/streamlit-your-system-has-an-unsupported-version-of-sqlite3-chroma-requires-sq
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
import ollama
import streamlit as st


from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from streamlit.runtime.uploaded_file_manager import UploadedFile

import json  

from app import *

#----------------------------------------------------------------
uploaded_file = open("/home/bc2/bruno/work/rinkai/material/docs/201015 -- Rinkai Routing - EN.pdf", "rb")
all_splits = process_document(uploaded_file)

strs=list(map(lambda x:x.page_content,all_splits))

with open("savedata.json", "w") as save_file:  
  json.dump(strs, save_file, indent = 6)  


results = query_collection("how to increase a vehicle capacity?")