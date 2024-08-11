from functions import *
from constants import *
from langchain.retrievers import EnsembleRetriever


embed_fn = load_local_embedding_model(model_directory)

vectordb = Chroma(persist_directory=persist_directory, embedding_function=embed_fn)

chroma_retriever = vectordb.as_retriever(search_kwargs={"k": 3})

bm25_retriever = load_bm25_index(bm25_file_path)

ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, chroma_retriever], weights=[0.75, 0.25])

ensemble_chain = create_qa_chain(ensemble_retriever)

# print(ensemble_chain.invoke({"question" : "Which movies involves a young Tibetan Mastiff who is expected to be the next guard of the village of Snow Mountain?"}))
print(ensemble_chain.invoke({"question" : "Which movies involves a vampire and lycans?"}))