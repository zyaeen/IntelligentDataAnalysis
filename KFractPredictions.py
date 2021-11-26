import numpy as np

from DataPreparing import prepare_data
from RMSE import rmse

#Рекомендации с учётом средних оценок похожих пользователей

def k_fract_predict(top, data):
    pred = np.zeros((data["n_users"], data["n_movies"]))
    top_similar = np.zeros((data["n_users"], top))
    abs_sim = np.abs(data["user_similarity"])

    for i in range(data["n_users"]):
        user_sim = data["user_similarity"][i]
        top_sim_users = user_sim.argsort()[1:top + 1]

        for j in range(top):
            top_similar[i, j] = top_sim_users[j]

    for i in range(data["n_users"]):
        indexes = top_similar[i].astype(np.int)
        numerator = data["user_similarity"][i][indexes]
        product = numerator.dot(data['train_data_matrix'][indexes])
        denominator = abs_sim[i][top_similar[i].astype(np.int)].sum()

        pred[i] = product / denominator

    return pred


def k_fract_predict_item(top, data):
    pred = np.zeros((data["n_movies"], data["n_users"]))
    top_similar = np.zeros((data["n_movies"], top))
    abs_sim = np.abs(data['item_similarity'])

    for i in range(data["n_movies"]):
        movies_sim = data['item_similarity'][i]
        top_sim_movies = movies_sim.argsort()[1:top + 1]

        for j in range(top):
            top_similar[i, j] = top_sim_movies.T[j]

    for i in range(data["n_users"]):
        indexes = top_similar[i].astype(np.int)
        numerator = data['item_similarity'][i][indexes]
        product = numerator.dot(data['train_data_matrix'].T[indexes])
        denominator = abs_sim[i][indexes].sum()
        denominator = denominator if denominator != 0 else 1

        pred[i] = product / denominator

    return pred.T


prepared_data = prepare_data(path='S:/Проекты 5 курс/Python/Датасеты/ratings.csv',
                             sample_size=100_000,
                             test_size=0.2
                             )

pred = k_fract_predict(7, prepared_data)
print('User-based CF RMSE: ', rmse(pred, prepared_data["test_data_matrix"]))

pred_item = k_fract_predict_item(7, prepared_data)
print('Item-based CF RMSE: ', rmse(pred_item, prepared_data["test_data_matrix"]))
