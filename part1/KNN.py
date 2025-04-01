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
from surprise import KNNBasic, KNNWithMeans, NMF
from sklearn.neighbors import KNeighborsRegressor
from surprise.model_selection import GridSearchCV
import time


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

###NMF

nmf = NMF()
nmf.fit(trainset)
predNMF = nmf.test(testset)
rmse_NMF = accuracy.rmse(predNMF)
mae_NMF = accuracy.mae(predNMF)
y_predNMF = [pred.est for pred in predNMF]
r2_nmf =  r2_score(y_true, y_predNMF)



results = pd.concat([results, pd.DataFrame({
    'Model': ['BaselineOnly (Surprise)', 'SVD (Surprise)', 'NMF (Suprise)'], 
    'RMSE': [rmse_surprise, rmse_SVD, rmse_NMF],                     
    'MAE': [mae_surprise, mae_SVD, mae_NMF],                         
    'R2 score': [r2_surprise, r2_SVD, r2_nmf]                    
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

# Model comparison with the base parameters

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



#####################################################################################################################

#Testing with different parameters with the worst performer NMF and 2 of the best 3 perfomers KNNwithmean and Baselinemodel

# Function to pretty print grid search results
def print_grid_search_results(results, n_top=5):
    print(f"Top {n_top} parameter combinations:")
    for i, params in enumerate(results[:n_top]):
        print(f"{i+1}. {params['params']} - RMSE: {params['mean_test_rmse']:.4f}")


best_configs = {}
all_best_models = []

# models param to test
param_grid_baseline = {
    'bsl_options': {
        'method': ['als', 'sgd'],
        'reg': [0.001, 0.01, 0.02, 0.05, 0.1, 0.2],
        'learning_rate': [0.001, 0.005, 0.01, 0.02, 0.05] # Only for sgd
    }
}



param_grid_knn = {
    'k': [5, 10, 20, 30, 40, 50, 60],
    'min_k': [1, 2, 3, 5],
    'sim_options': {
        'name': ['pearson', 'cosine', 'msd']

    }
}

param_grid_nmf = {
    'n_factors': [10, 15, 20, 30, 50, 75],
    'n_epochs': [25, 50, 75, 100],
    'reg_pu': [0.01, 0.02, 0.06, 0.1],
    'reg_qi': [0.01, 0.02, 0.06, 0.1]
}

gs_baseline = GridSearchCV(BaselineOnly, param_grid_baseline, measures=['rmse', 'mae'], cv=3)
gs_baseline.fit(data)

best_baseline_params = gs_baseline.best_params['rmse']
best_baseline_score = gs_baseline.best_score['rmse']
print(f"\nBaselineOnly Best Parameters: {best_baseline_params}")
print(f"RMSE: {best_baseline_score:.4f}")

results_baseline = []
for params, mean_rmse, mean_mae in zip(
        gs_baseline.cv_results['params'],
        gs_baseline.cv_results['mean_test_rmse'],
        gs_baseline.cv_results['mean_test_mae']):
    results_baseline.append({'params': params, 'mean_test_rmse': mean_rmse, 'mean_test_mae': mean_mae})
results_baseline.sort(key=lambda x: x['mean_test_rmse'])
print_grid_search_results(results_baseline)

best_configs['BaselineOnly'] = best_baseline_params



gs_knn = GridSearchCV(KNNWithMeans, param_grid_knn, measures=['rmse', 'mae'], cv=3)
gs_knn.fit(data)

best_knn_params = gs_knn.best_params['rmse']
best_knn_score = gs_knn.best_score['rmse']
print(f"\nKNNWithMeans Best Parameters: {best_knn_params}")
print(f"RMSE: {best_knn_score:.4f}")

results_knn = []
for params, mean_rmse, mean_mae in zip(
        gs_knn.cv_results['params'],
        gs_knn.cv_results['mean_test_rmse'],
        gs_knn.cv_results['mean_test_mae']):
    results_knn.append({'params': params, 'mean_test_rmse': mean_rmse, 'mean_test_mae': mean_mae})
results_knn.sort(key=lambda x: x['mean_test_rmse'])
print_grid_search_results(results_knn)

best_configs['KNNWithMeans'] = best_knn_params


gs_nmf = GridSearchCV(NMF, param_grid_nmf, measures=['rmse', 'mae'], cv=3)
gs_nmf.fit(data)

best_nmf_params = gs_nmf.best_params['rmse']
best_nmf_score = gs_nmf.best_score['rmse']
print(f"\nNMF Best Parameters: {best_nmf_params}")
print(f"RMSE: {best_nmf_score:.4f}")

# Sort results by RMSE
results_nmf = []
for params, mean_rmse, mean_mae in zip(
        gs_nmf.cv_results['params'],
        gs_nmf.cv_results['mean_test_rmse'],
        gs_nmf.cv_results['mean_test_mae']):
    results_nmf.append({'params': params, 'mean_test_rmse': mean_rmse, 'mean_test_mae': mean_mae})
results_nmf.sort(key=lambda x: x['mean_test_rmse'])
print_grid_search_results(results_nmf)

best_configs['NMF'] = best_nmf_params

########################################################################################################################################################
# Retrain models with optimal parameters and compare them with everything else I did


best_baseline = BaselineOnly(**best_configs['BaselineOnly'])
best_knn = KNNWithMeans(**best_configs['KNNWithMeans'])
best_nmf = NMF(**best_configs['NMF'])

best_models = [best_baseline, best_knn, best_nmf]
best_model_names = ['BaselineOnly (Optimized)',
                    'KNNWithMeans (Optimized)', 'NMF (Optimized)']

optimized_results = []

for name, model in zip(best_model_names, best_models):
    model.fit(trainset)
    
    predictions = model.test(testset)
    

    rmse = accuracy.rmse(predictions)
    mae = accuracy.mae(predictions)
    y_true = np.array([pred.r_ui for pred in predictions])
    y_pred = np.array([pred.est for pred in predictions])
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    
    optimized_results.append({
        'Model': name,
        'RMSE': rmse,
        'MAE': mae,
        'MSE': mse,
        'R2 score': r2
    })

optimized_df = pd.DataFrame(optimized_results)


comparison_results = pd.concat([results, optimized_df], ignore_index=True)

print("\nComparison of all models (original vs. optimized):")
print(comparison_results[['Model', 'RMSE', 'MAE', 'MSE', 'R2 score']])


rmse_rank = comparison_results['RMSE'].rank()
mae_rank = comparison_results['MAE'].rank()
mse_rank = comparison_results['MSE'].rank()
r2_rank = comparison_results['R2 score'].rank(ascending=False)  # Reversed for R2

comparison_results['RMSE Rank'] = rmse_rank
comparison_results['MAE Rank'] = mae_rank
comparison_results['MSE Rank'] = mse_rank
comparison_results['R2 Rank'] = r2_rank
comparison_results['Avg Rank'] = (rmse_rank + mae_rank + mse_rank + r2_rank) / 4

ranked_final = comparison_results.sort_values('Avg Rank')
print("\nFinal Model Rankings (including optimized models):")
print(ranked_final[['Model', 'RMSE Rank', 'MAE Rank', 'MSE Rank', 'R2 Rank', 'Avg Rank']])


best_final = ranked_final.iloc[0]
print(f"\nBest Overall Model: {best_final['Model']} (Average Rank: {best_final['Avg Rank']:.2f})")

############################################################################################################V
# With knn we found that it has taken the highest k neighbors, so we will try to find a the k neighbors to be able to use later 



param_grid_knn = {
    'k': [65, 70, 80, 90, 100, 110, 120] # best k neighbors found after 90 le RMSE converges so that is the number of neighbors we will use later 
    
}
    
gs_knn = GridSearchCV(KNNWithMeans, param_grid_knn, measures=['rmse', 'mae'], cv=3)
gs_knn.fit(data)

best_knn_params = gs_knn.best_params['rmse']
best_knn_score = gs_knn.best_score['rmse']
print(f"\nKNNWithMeans Best Parameters: {best_knn_params}")
print(f"RMSE: {best_knn_score:.4f}")

results_knn = []
for params, mean_rmse, mean_mae in zip(
        gs_knn.cv_results['params'],
        gs_knn.cv_results['mean_test_rmse'],
        gs_knn.cv_results['mean_test_mae']):
    results_knn.append({'params': params, 'mean_test_rmse': mean_rmse, 'mean_test_mae': mean_mae})
results_knn.sort(key=lambda x: x['mean_test_rmse'])
print_grid_search_results(results_knn)

best_configs['KNNWithMeans'] = best_knn_params
