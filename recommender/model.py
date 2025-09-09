from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
import pandas as pd

def train_model(ratings):
    reader = Reader(rating_scale=(ratings.rating.min(), ratings.rating.max()))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

    algo = SVD(n_factors=50, random_state=42)
    algo.fit(trainset)

    predictions = algo.test(testset)
    print("Model Evaluation:")
    accuracy.rmse(predictions)
    accuracy.mae(predictions)

    return algo
