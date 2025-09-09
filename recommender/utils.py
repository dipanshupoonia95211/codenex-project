def recommend_for_user(algo, user_id, ratings_df, topN=5, movies_df=None):
    all_movie_ids = ratings_df['movieId'].unique().tolist()
    rated = set(ratings_df[ratings_df.userId == user_id]['movieId'].tolist())
    unrated = [m for m in all_movie_ids if m not in rated]

    preds = [(m, algo.predict(user_id, m).est) for m in unrated]
    preds.sort(key=lambda x: x[1], reverse=True)
    top_preds = preds[:topN]

    if movies_df is not None:
        results = []
        for movie_id, score in top_preds:
            title = movies_df[movies_df.movieId == movie_id]['title'].values
            title = title[0] if len(title) > 0 else "Unknown"
            results.append((title, score))
        return results
    else:
        return top_preds
