from langchain.prompts import ChatPromptTemplate
from pathlib import Path
import os

repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
template = """Based on the provided context, list the movies that match the query. If no relevant movies are found, respond with 'No Movies Found'. Provide the titles in the format: Movie_Title (Release Year). Do not provide any extra details regarding context or answer. Do not add any explanation for your answer.
  ### CONTEXT
  {context}
  ### QUESTION
  Question: {question}
  ### ANSWER
  Answer:
  """

prompt = ChatPromptTemplate.from_template(template)

chunk_size = 1000
chunk_overlap = 100
top_k = 2
# csv_file_path = 'Input_data/filtered_data.csv'
persist_directory = 'db'
bm25_file_path = 'bm25_index.pkl'

# Get the absolute path to the project root
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Define the CSV file path relative to the project root
csv_file_path = PROJECT_ROOT / "Input_data" / "test_data.csv"