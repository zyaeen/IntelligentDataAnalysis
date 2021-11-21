import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import pairwise_distances


def prepare_data(path, sample_size, test_size):
    ratings_df = pd.read_csv(path)
    ratings_df_sample = ratings_df[: sample_size]

    n_users = len(ratings_df_sample['userId'].unique())
    n_movies = len(ratings_df_sample['movieId'].unique())

    movie_ids = ratings_df_sample['movieId'].unique()

    def scale_movie_id(movie_id):
        scaled = np.where(movie_ids == movie_id)[0][0] + 1
        return scaled

    ratings_df_sample['movieId'] = ratings_df_sample['movieId'].apply(scale_movie_id)

    train_data, test_data = train_test_split(ratings_df_sample, test_size=test_size)

    # Получение характеристической матрицы, т.е. матрицы соответствия пользователей и фильмов
    # В ячейках хранятся оценки пользователя фильмам

    train_data_matrix = np.zeros((n_users, n_movies))
    for line in train_data.itertuples():
        train_data_matrix[line[1] - 1, line[2] - 1] = line[3]

    test_data_matrix = np.zeros((n_users, n_movies))
    for line in test_data.itertuples():
        test_data_matrix[line[1] - 1, line[2] - 1] = line[3]

    # Расчет "схожести" для пользователей
    user_similarity = pairwise_distances(train_data_matrix, metric='cosine')
    # Расчет "схожести" для фильмов
    item_similarity = pairwise_distances(train_data_matrix.T, metric='cosine')

    prepared_data = {
        "user_similarity": user_similarity,
        "item_similarity": item_similarity,
        "test_data": test_data,
        "train_data": train_data,
        "train_data_matrix": train_data_matrix,
        "test_data_matrix": test_data_matrix,
        "n_users": n_users,
        "n_movies": n_movies
    }

    return prepared_data
