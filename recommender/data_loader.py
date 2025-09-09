import pandas as pd

def load_ratings(path="data/ratings.csv"):
    ratings = pd.read_csv(path)
    return ratings

def load_movies(path="data/movies.csv"):
    try:
        movies = pd.read_csv(path)
        return movies
    except FileNotFoundError:
        print("⚠️ movies.csv not found. Titles will not be available yet.")
        return None
