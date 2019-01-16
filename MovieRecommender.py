import numpy as np
import pandas as pd

#Creating legible pandas dataframe of csv file with reviews
column_names = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('u.data', sep='\t', names=column_names)

#creating pandas dataframe of movie titles with IDs
movie_titles = pd.read_csv("Movie_Id_Titles")

#merging two dataframes on movie ID so new dataframe will have movie titles as well
df = pd.merge(df,movie_titles,on='item_id')

#setting up for matrix with num users as rows and items as columns
n_users = df.user_id.nunique()
n_items = df.item_id.nunique()

#outputs text for size of matrix
print('Num. of Users: '+ str(n_users))
print('Num of Movies: '+str(n_items))

#splitting data forcomparison
from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(df, test_size=0.25)

#making matrix with users as rows movies as columns, each value is user rating
train_data_matrix = np.zeros((n_users, n_items))
for line in train_data.itertuples():
    train_data_matrix[line[1]-1, line[2]-1] = line[3]
    
#two matrices for traing and testing
test_data_matrix = np.zeros((n_users, n_items))
for line in test_data.itertuples():
    test_data_matrix[line[1]-1, line[2]-1] = line[3]

#calculating distance metric, cosine similarity
#user similarity compares rows
#item similarity transposes matrix to make each row a movie and compares each
from sklearn.metrics.pairwise import pairwise_distances
user_similarity = pairwise_distances(train_data_matrix, metric='cosine')
item_similarity = pairwise_distances(train_data_matrix.T, metric='cosine')

#function for predicting ratings based on similarity established

def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        #You use np.newaxis so that mean_user_rating has same format as ratings
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis]) 
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])     
    return pred

#running predictions for each item and user
item_prediction = predict(train_data_matrix, item_similarity, type='item')
user_prediction = predict(train_data_matrix, user_similarity, type='user')    

#finding mean squared error of results, importing and running mean squared error
#use nonzero so unrated movies not counted
#only want movies that are in both train and test matrices
from sklearn.metrics import mean_squared_error
from math import sqrt
def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten() 
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(ground_truth, prediction))
    
#printing results
print('User-based CF RMSE: ' + str(rmse(user_prediction, test_data_matrix)))
print('Item-based CF RMSE: ' + str(rmse(item_prediction, test_data_matrix)))
