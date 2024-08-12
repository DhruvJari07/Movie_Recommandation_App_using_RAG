import streamlit as st
from functions import *
from constants import *
from langchain.retrievers import EnsembleRetriever
import re
import random

# Load the model and create the ensemble chain
@st.cache_resource
def load_model():
    embed_fn = load_local_embedding_model(model_directory)
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embed_fn)
    chroma_retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    bm25_retriever = load_bm25_index(bm25_file_path)
    ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, chroma_retriever], weights=[0.75, 0.25])
    return create_qa_chain(ensemble_retriever)

def parse_movies(response):
    movies = re.findall(r'\d+\.\s*(.*?)(?=\n|$)', response)
    return [movie.strip() for movie in movies if movie.strip()]

def create_imdb_search_url(movie_title):
    return f"https://www.imdb.com/find?q={movie_title.replace(' ', '+')}"

def get_random_rating():
    return round(random.uniform(6.0, 9.5), 1)

def display_movie_tiles(movies):
    # CSS for styling
    st.markdown("""
    <style>
    .movie-tile {
        background-color: #f8f9fa;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    .movie-tile:hover {
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        transform: translateY(-3px);
    }
    .movie-title {
        font-size: 18px;
        font-weight: bold;
        margin-bottom: 10px;
        color: #2c3e50;
    }
    .movie-year {
        font-size: 14px;
        color: #7f8c8d;
        margin-bottom: 10px;
    }
    .movie-rating {
        font-size: 16px;
        font-weight: bold;
        color: #f39c12;
        margin-bottom: 15px;
    }
    .visit-site-btn {
        background-color: #3498db;
        color: white;
        padding: 10px 15px;
        border: none;
        border-radius: 5px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 14px;
        transition: background-color 0.3s ease;
    }
    .visit-site-btn:hover {
        background-color: #2980b9;
    }
    </style>
    """, unsafe_allow_html=True)

    # Display movies in a single column layout
    for movie in movies:
        # Extract year if present in the movie title
        year_match = re.search(r'\((\d{4})\)$', movie)
        year = year_match.group(1) if year_match else "N/A"
        title = re.sub(r'\s*\(\d{4}\)$', '', movie) if year_match else movie

        rating = get_random_rating()

        st.markdown(f"""
        <div class="movie-tile">
            <div class="movie-title">{title}</div>
            <div class="movie-year">Year: {year}</div>
            <div class="movie-rating">Rating: {rating}</div>
            <a href="{create_imdb_search_url(title)}" target="_blank" class="visit-site-btn">Visit Site</a>
        </div>
        """, unsafe_allow_html=True)

# Main Streamlit app
def main():
    st.title("Movie Recommendation System")

    # Load the model
    ensemble_chain = load_model()
    
    # Initialize session state for query
    if 'query' not in st.session_state:
        st.session_state.query = ""

    # User input
    user_query = st.text_input("Enter your movie query:", value=st.session_state.query)

    # Update session state when user types
    if user_query != st.session_state.query:
        st.session_state.query = user_query

    if st.button("Get Recommendations"):
        if user_query:
            with st.spinner("Searching for movies..."):
                # Get recommendations
                result = ensemble_chain.invoke({"question": user_query})

                # Parse movies from the response
                movies = parse_movies(result['response'])

                # Display results
                st.subheader("Recommended Movies:")
                display_movie_tiles(movies)

        else:
            st.warning("Please enter a query.")

if __name__ == "__main__":
    main()