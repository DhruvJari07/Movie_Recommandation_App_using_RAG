import os
import pandas as pd
from typing import List, Dict
from warnings import simplefilter
from langchain import hub
#from langchain_community.llms import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import Chroma, LanceDB
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from dotenv import load_dotenv
from langchain_community.retrievers import BM25Retriever
from constants import *
from functions import *
from pathlib import Path

load_dotenv()

os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv('HF_TOKEN')
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

embed_fn = SentenceTransformerEmbeddings(model_name="all-minilm-l6-v2")

create_vector_embeddings(csv_file_path, embed_fn, chunk_overlap, chunk_size, top_k)


