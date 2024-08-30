import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


data = {
    'user': ['Alice', 'Alice', 'Alice', 'Bob', 'Bob', 'Bob', 'Charlie', 'Charlie', 'Charlie'],
    'movie': ['Movie1', 'Movie2', 'Movie3', 'Movie1', 'Movie2', 'Movie4', 'Movie3', 'Movie4', 'Movie5'],
    'rating': [4, 5, 3, 5, 4, 2, 2, 4, 5]
}


df = pd.DataFrame(data)


user_item_matrix = df.pivot_table(index='user', columns='movie', values='rating')
user_item_matrix.fillna(0, inplace=True)


user_similarity = cosine_similarity(user_item_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)


def get_recommendations(user, user_item_matrix, user_similarity_df):
    
    similar_users = user_similarity_df[user].sort_values(ascending=False).index[1:]  # Exclude the user itself
    user_ratings = user_item_matrix.loc[user]  
    
    weighted_ratings = pd.Series(dtype=float)  
    
    
    for other_user in similar_users:
        other_ratings = user_item_matrix.loc[other_user]
        similarity_score = user_similarity_df.loc[user, other_user]
        weighted_ratings = weighted_ratings.add(other_ratings * similarity_score, fill_value=0)
    
    
    weighted_ratings /= user_similarity_df.loc[user, similar_users].sum()
    
    
    recommendations = weighted_ratings[user_ratings == 0].sort_values(ascending=False)
    return recommendations


user = 'Alice'  
recommendations = get_recommendations(user, user_item_matrix, user_similarity_df)


print(f"Recommendations for {user}:")
print(recommendations)
