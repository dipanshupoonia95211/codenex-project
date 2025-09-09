from recommender.data_loader import load_ratings, load_movies
from recommender.model import train_model
from recommender.utils import recommend_for_user

def main():
    # load data
    ratings = load_ratings("data/ratings.csv")
    movies = load_movies("data/movies.csv")  # will work later

    print("Ratings shape:", ratings.shape)
    print("Unique users:", ratings['userId'].nunique())
    print("Unique movies:", ratings['movieId'].nunique())

    # train CF model
    algo = train_model(ratings)

    # recommend for user 1
    print("\nTop 5 Recommendations for User 1:")
    recs = recommend_for_user(algo, 1, ratings, topN=5, movies_df=movies)
    for r in recs:
        print(r)

if __name__ == "__main__":
    main()
