import streamlit as st
from functions import *
from constants import *
from langchain.retrievers import EnsembleRetriever

# Load the model and create the ensemble chain
@st.cache_resource
def load_model():
    embed_fn = load_local_embedding_model(model_directory)
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embed_fn)
    chroma_retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    bm25_retriever = load_bm25_index(bm25_file_path)
    ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, chroma_retriever], weights=[0.75, 0.25])
    return create_qa_chain(ensemble_retriever)

# Main Streamlit app
def main():
    st.title("Movie Recommendation System")

    # Load the model
    ensemble_chain = load_model()

    # User input
    user_query = st.text_input("Enter your movie query:", "")

    if st.button("Get Recommendations"):
        if user_query:
            with st.spinner("Searching for movies..."):
                # Get recommendations
                result = ensemble_chain.invoke({"question": user_query})

                # Display results
                st.subheader("Recommended Movies:")
                st.write(result['response'])

                # Display context (optional)
                if st.checkbox("Show Context"):
                    st.subheader("Context:")
                    for doc in result['context']:
                        st.write(doc.page_content)
                        st.write("---")
        else:
            st.warning("Please enter a query.")

if __name__ == "__main__":
    main()