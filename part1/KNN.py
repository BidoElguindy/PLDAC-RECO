# Imports
import bson
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.sparse import csr_matrix
from fonctions import *
from surprise import BaselineOnly, SVD
from surprise import Dataset
from surprise import Reader
from surprise import accuracy
from surprise.model_selection import train_test_split as surprise_train_test_split
from surprise import KNNBasic, KNNWithMeans
from sklearn.neighbors import KNeighborsRegressor


###################################################################################################

# Open and read the BSON files
with open("./trictrac/details.bson", "rb") as f:
    details = bson.decode_all(f.read())


with open("./trictrac/infos_scrapping.bson", "rb") as f:
    infos = bson.decode_all(f.read())

with open("./trictrac/jeux.bson", "rb") as f:
    jeux = bson.decode_all(f.read())


with open("./trictrac/avis.bson", "rb") as f:
    avis = bson.decode_all(f.read())

pdDetails = pd.DataFrame(details)
pdJeux = pd.DataFrame(jeux)
pdInfos = pd.DataFrame(infos)
pdAvis = pd.DataFrame(avis)

###################################################################################################
# Nettoyage

# Spam removable 
pdAvis['review_length'] = pdAvis['comment'].str.len()
pdAvis=pdAvis[pdAvis["review_length"] != 58097.0]

#Duplicates removable
pdAvis.drop_duplicates(subset=['author','title_review','note','title','comment'], inplace=True)
pdJeux.drop_duplicates(subset=['title',	'href','avis'], inplace=True)


allgames = sorted(pdJeux["title"].unique())
allusers = sorted(pdAvis['author'].unique())


# Création de la dataFrame  
ratings_matrix = pd.DataFrame(
    index=allusers,
    columns=allgames,
    dtype=float
)

# Remplissage de la matrice avec les notes
for _, row in pdAvis.iterrows():
    ratings_matrix.at[row['author'], row['title']] = row['note']

# Conversion en matrice sparse
mask = ~ratings_matrix.isna()
sparse_ratings_matrix = csr_matrix(
    (
        ratings_matrix.values[mask], 
        np.where(mask)
    ),
    shape=ratings_matrix.shape
)

# Remove users and games with with low numbers of ratings the optimal value found was (14,18)
from fonctions import minipingpong, subtract_mean
cleaned_matrix = minipingpong(sparse_ratings_matrix, allusers, allgames, 14, 18)

###################################################################################################

#Baseline models

train_matrix, test_matrix, train_mask, test_mask = custom_train_test_split(
    cleaned_matrix, 
    test_size=0.2, 
    max_user_loss=0.5, 
    max_game_loss=0.5,
)

test_rows, test_cols = test_matrix.nonzero()
test_ratings = test_matrix.data


global_mean, rmse_global, mae_global, r2_global = global_mean_baseline(train_matrix, test_matrix)
user_means, rmse_user, mae_user, r2_user = user_mean_baseline(train_matrix, test_matrix) 
item_means, rmse_game, mae_game, r2_game = item_mean_baseline(train_matrix, test_matrix)


results = pd.DataFrame({
    'Model': ['Global Mean', 'User Mean', 'Game Mean'],
    'RMSE': [rmse_global, rmse_user, rmse_game, ],
    'MAE': [mae_global, mae_user, mae_game],
    'R2 score': [r2_global, r2_user, r2_game]
})

# Optimale baseline model + SVD

BaselineModel = BaselineOnly()
SVD = SVD()

# Convert the cleaned_matrix data to Surprise format
ratings_list = []
for i, j in zip(*cleaned_matrix.nonzero()):
    user = allusers[i]
    game = allgames[j]
    rating = cleaned_matrix[i, j]
    ratings_list.append((user, game, rating))


ratings_df = pd.DataFrame(ratings_list, columns=['userID', 'itemID', 'rating'])

# Define the reader with appropriate rating scale 
reader = Reader(rating_scale=(0, 10))

# Create a Surprise dataset
data = Dataset.load_from_df(ratings_df, reader)
trainset, testset = surprise_train_test_split(data, test_size=0.2, random_state=42)

BaselineModel.fit(trainset)
SVD.fit(trainset)
predictions = BaselineModel.test(testset)
predictionsSVD = SVD.test(testset)

rmse_surprise = accuracy.rmse(predictions)
mae_surprise = accuracy.mae(predictions)
rmse_SVD = accuracy.rmse(predictionsSVD)
mae_SVD = accuracy.mae(predictionsSVD)

y_true = [pred.r_ui for pred in predictions]
y_pred = [pred.est for pred in predictions]
y_predSVD = [pred.est for pred in predictionsSVD]
r2_surprise = r2_score(y_true, y_pred)
r2_SVD = r2_score(y_true, y_predSVD)


results = pd.concat([results, pd.DataFrame({
    'Model': ['BaselineOnly (Surprise)', 'SVD (Surprise)'], 
    'RMSE': [rmse_surprise, rmse_SVD],                     
    'MAE': [mae_surprise, mae_SVD],                         
    'R2 score': [r2_surprise, r2_SVD]                    
})], ignore_index=True)

########################################################################################################################
##KNN

knn_basic = KNNBasic(k=40, sim_options={'name': 'pearson', 'user_based': True}) # After testing 40 was the best k neighbors found
knn_means = KNNWithMeans(k=40, sim_options={'name': 'pearson', 'user_based': True})


knn_basic.fit(trainset)
knn_means.fit(trainset)


preds_basic = knn_basic.test(testset)
preds_means = knn_means.test(testset)

rmse_knn_basic = accuracy.rmse(preds_basic)
mae_knn_basic = accuracy.mae(preds_basic)
rmse_knn_means = accuracy.rmse(preds_means)
mae_knn_means = accuracy.mae(preds_means)
y_true = [pred.r_ui for pred in preds_basic]
y_pred_basic = [pred.est for pred in preds_basic]
y_pred_means = [pred.est for pred in preds_means]
r2_knn_basic = r2_score(y_true, y_pred_basic)
r2_knn_means = r2_score(y_true, y_pred_means)


knn_results = pd.DataFrame({
    'Model': ['KNNBasic (Suprise)', 'KNNWithMeans (Suprise)'],
    'RMSE': [rmse_knn_basic, rmse_knn_means],
    'MAE': [mae_knn_basic, mae_knn_means],
    'R2 score': [r2_knn_basic, r2_knn_means]
})

# Combine with previous results
results = pd.concat([results, knn_results], ignore_index=True)


###################################################################################################
# KNN sklearn

# Soustraction de la moyenne de chaque utilisateur de ces ratings
normalized_matrix = subtract_mean(cleaned_matrix)
#print(normalized_matrix)


#Train/Test split

from fonctions import custom_train_test_split
train_matrixN, test_matrixN, train_maskn, test_maskN = custom_train_test_split(
    normalized_matrix, 
    test_size=0.2, 
    max_user_loss=0.5, 
    max_game_loss=0.5,
)

# Apply sklearn KNN to the normalized data
train_matrixN_coo = train_matrixN.tocoo()
train_users = train_matrixN_coo.row
train_items = train_matrixN_coo.col
train_ratings = train_matrixN_coo.data

X_train = np.column_stack((train_users, train_items))
y_train = train_ratings 

# Get test data in the right format
test_matrixN_coo = test_matrixN.tocoo()
test_users = test_matrixN_coo.row
test_items = test_matrixN_coo.col
test_ratings = test_matrixN_coo.data

# Create feature vectors for testing
X_test = np.column_stack((test_users, test_items))

# Get user means (needed to revert normalization)
data_array = cleaned_matrix.toarray()
user_means = np.zeros(data_array.shape[0])
for i in range(data_array.shape[0]):
    user_ratings = data_array[i, :]
    non_zero_indices = user_ratings != 0
    if non_zero_indices.any():
        user_means[i] = user_ratings[non_zero_indices].mean()

knn = KNeighborsRegressor(n_neighbors=40, weights='distance')
knn.fit(X_train, y_train)

# Generate predictions (still normalized)
y_pred_normalized = knn.predict(X_test)

# Revert the normalization to get actual predictions
y_pred = y_pred_normalized + np.array([user_means[user] for user in test_users])

# Revert the normalization for test data too
y_test_actual = test_ratings + np.array([user_means[user] for user in test_users])

rmse_sklearn = np.sqrt(mean_squared_error(y_test_actual, y_pred))
mae_sklearn = mean_absolute_error(y_test_actual, y_pred)
r2_sklearn = r2_score(y_test_actual, y_pred)

sklearn_results = pd.DataFrame({
    'Model': ['Sklearn KNN'],
    'RMSE': [rmse_sklearn],
    'MAE': [mae_sklearn],
    'R2 score': [r2_sklearn]
})


results = pd.concat([results, sklearn_results], ignore_index=True)

##############################################################################################################

# Results compairison
print("All Model Comparison:")
print(results)


results['MSE'] = results['RMSE'] ** 2


plt.figure(figsize=(14, 8))
barWidth = 0.25
models = results['Model']
x = np.arange(len(models))

# Create three sets of bars
br1 = x - barWidth
br2 = x 
br3 = x + barWidth


plt.bar(br1, results['RMSE'], width=barWidth, label='RMSE', color='skyblue')
plt.bar(br2, results['MAE'], width=barWidth, label='MAE', color='lightgreen')
plt.bar(br3, results['MSE'], width=barWidth, label='MSE', color='salmon')

plt.xlabel('Model', fontweight='bold')
plt.ylabel('Error Metrics', fontweight='bold')
plt.title('Recommendation Models Performance Comparison')
plt.xticks(x, models, rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.show()


# Calculate rankings for each model across metrics
# For RMSE, MAE, MSE: lower is better
# For R2: higher is better
rmse_rank = results['RMSE'].rank()
mae_rank = results['MAE'].rank()
mse_rank = results['MSE'].rank()
r2_rank = results['R2 score'].rank(ascending=False)  # Reversed for R2

results['RMSE Rank'] = rmse_rank
results['MAE Rank'] = mae_rank
results['MSE Rank'] = mse_rank
results['R2 Rank'] = r2_rank

# Calculate average rank across all metrics
results['Avg Rank'] = (rmse_rank + mae_rank + mse_rank + r2_rank) / 4
ranked_results = results.sort_values('Avg Rank')


print("\nModel Rankings :")
print(ranked_results[['Model', 'RMSE Rank', 'MAE Rank', 'MSE Rank', 'R2 Rank', 'Avg Rank']])

best_rmse = results.loc[results['RMSE'].idxmin()]
best_mae = results.loc[results['MAE'].idxmin()]
best_mse = results.loc[results['MSE'].idxmin()]
best_r2 = results.loc[results['R2 score'].idxmax()]

# Find overall best model 
best_overall = ranked_results.iloc[0]

print("\nBest Models By Metric:")
print(f"Best RMSE: {best_rmse['Model']} ({best_rmse['RMSE']:.4f})")
print(f"Best MAE: {best_mae['Model']} ({best_mae['MAE']:.4f})")
print(f"Best MSE: {best_mse['Model']} ({best_mse['MSE']:.4f})")
print(f"Best R²: {best_r2['Model']} ({best_r2['R2 score']:.4f})")
print(f"\nBest Overall Model: {best_overall['Model']} (Average Rank: {best_overall['Avg Rank']:.2f})")

