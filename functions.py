import os
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.retrievers import BM25Retriever
from langchain.schema.runnable import RunnablePassthrough
from operator import itemgetter
import pickle
from constants import *

def create_vector_embeddings(csv_file_path, embed_fn, chunk_overlap, chunk_size, top_k):
  print(f"Current working directory: {os.getcwd()}")
  print(f"Attempting to load file: {csv_file_path}")
    
  csv_path = Path(csv_file_path)
  print(f"Absolute path: {csv_path.absolute()}")
  
  if not csv_path.exists():
      raise FileNotFoundError(f"CSV file not found: {csv_path.absolute()}")
  
  print("File exists, loading data...")
  loader = CSVLoader(csv_file_path, encoding='utf-8')
  print("loading data")
  data = loader.load()
  print("file loaded")
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(data)
  bm25_retriever = BM25Retriever.from_documents(docs)
  bm25_retriever.k = top_k
  save_bm25_index(bm25_retriever, bm25_file_path)
  print("bm25 retriever saved")
  vectorstore = Chroma.from_documents(docs, embed_fn, persist_directory=persist_directory)
  print("chroma retreiver saved")



def create_qa_chain(retriever):
  primary_qa_llm = HuggingFaceEndpoint(
    repo_id=repo_id, max_length=2048, temperature=0.5, huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)
  created_qa_chain = (
    {"context": itemgetter("question") | retriever,
     "question": itemgetter("question")
    }
    | RunnablePassthrough.assign(
        context=itemgetter("context")
      )
    | {
         "response": prompt | primary_qa_llm,
         "context": itemgetter("context"),
      }
  )

  return created_qa_chain


def save_bm25_index(bm25_retriever, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(bm25_retriever, f)

def load_bm25_index(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)