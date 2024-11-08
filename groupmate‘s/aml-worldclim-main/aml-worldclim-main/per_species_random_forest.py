import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from tqdm import tqdm
from sklearn.decomposition import PCA

data = np.load('species_train.npz')
train_locs = data['train_locs']
train_ids = data['train_ids']
species = data['taxon_ids']
data_test = np.load('species_test.npz', allow_pickle=True)
test_locs = data_test['test_locs']
test_pos_inds = dict(zip(data_test['taxon_ids'], data_test['test_pos_inds']))
feature_names = [
    "latitude", "longitude", "BIO1", "BIO2", "BIO3", "BIO4", "BIO5", "BIO6",
    "BIO7", "BIO8", "BIO9", "BIO10", "BIO11", "BIO12", "BIO13", "BIO14",
    "BIO15", "BIO16", "BIO17", "BIO18", "BIO19", "K-means cluster ID"
]

for i in range(1, 20):
    filename_train = "wc2.1_2.5m_bio/wc2.1_2.5m_bio_" + str(i) + "_train.npy"
    filename_test = "wc2.1_2.5m_bio/wc2.1_2.5m_bio_" + str(i) + "_test.npy"
    train_feature = np.load(filename_train)
    train_locs = np.column_stack((train_locs, train_feature))
    test_feature = np.load(filename_test)
    test_locs = np.column_stack((test_locs, test_feature))

optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(train_locs[:, :2])
train_clusters = kmeans.predict(train_locs[:, :2])
test_clusters = kmeans.predict(test_locs[:, :2])
train_locs = np.column_stack((train_locs, train_clusters))
test_locs = np.column_stack((test_locs, test_clusters))

z_scores_train = np.abs(stats.zscore(train_locs))
train_outliers = np.where(z_scores_train > 3)
train_locs = np.delete(train_locs, train_outliers[0], axis=0)
train_ids = np.delete(train_ids, train_outliers[0], axis=0)

scaler = StandardScaler()
train_locs = scaler.fit_transform(train_locs)
test_locs = scaler.transform(test_locs)

feature_importances = []
models = {}
best_params = {}

for sp_id in tqdm(species, desc="Training models for each species"):
    y_binary = (train_ids == sp_id).astype(int)

    num_positive = np.sum(y_binary)
    pos_indices = np.where(y_binary == 1)[0]
    neg_indices = np.where(y_binary == 0)[0]
    neg_indices_resampled = resample(neg_indices, replace=False, n_samples=num_positive, random_state=42)
    balanced_indices = np.concatenate([pos_indices, neg_indices_resampled])

    train_locs_resample = train_locs[balanced_indices]
    y_binary_resample = y_binary[balanced_indices]

    model = RandomForestClassifier(random_state=42, max_depth=10, max_features='sqrt')
    model.fit(train_locs_resample, y_binary_resample)
    feature_importances.append(model.feature_importances_)

average_feature_importances = np.mean(feature_importances, axis=0)
importances_with_names = list(zip(feature_names, average_feature_importances))
importances_with_names_sorted = sorted(importances_with_names, key=lambda x: x[1], reverse=True)
df_feature_importances = pd.DataFrame(importances_with_names_sorted, columns=["Feature", "Average Importance"])

# plt.figure(figsize=(12, 6))
# plt.barh(df_feature_importances["Feature"], df_feature_importances["Average Importance"], color='skyblue')
# plt.xlabel("Average Importance")
# plt.title("Feature Importance Visualization")
# plt.gca().invert_yaxis()
# plt.tight_layout()
# plt.grid(axis='x', linestyle='--', alpha=0.6)
# plt.savefig("graph/feature_importances.png")
# plt.close()

mean_importance = np.mean(average_feature_importances)

selected_features = df_feature_importances[df_feature_importances["Average Importance"] > mean_importance]["Feature"].values
selected_indices = [feature_names.index(f) for f in selected_features]

train_locs = train_locs[:, selected_indices]
test_locs = test_locs[:, selected_indices]


for sp_id in tqdm(species, desc="Training models for each species on selected features"):
    y_binary = (train_ids == sp_id).astype(int)

    num_positive = np.sum(y_binary)
    pos_indices = np.where(y_binary == 1)[0]
    neg_indices = np.where(y_binary == 0)[0]
    neg_indices_resampled = resample(neg_indices, replace=False, n_samples=num_positive, random_state=42)
    balanced_indices = np.concatenate([pos_indices, neg_indices_resampled])

    train_locs_resample = train_locs[balanced_indices]
    y_binary_resample = y_binary[balanced_indices]

    # model = RandomForestClassifier(random_state=42, max_depth=10, max_features='sqrt')
    model = RandomForestClassifier(random_state=42)
    param_grid = {
        'criterion': ['gini', 'entropy', 'log_loss'],
        'max_leaf_nodes': [None, 10, 20],
        'n_estimators': [100, 200, 300],
        'max_features': ['auto', 'sqrt', 'log2'],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]       
    }
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, scoring='f1')
    grid_search.fit(train_locs_resample, y_binary_resample)
    best_params[sp_id] = grid_search.best_params_

    # model.fit(train_locs_resample, y_binary_resample)
    models[sp_id] = grid_search.best_estimator_

best_params_df = pd.DataFrame(best_params.items(), columns=["Species ID", "Best Parameters"])
best_params_df.to_csv("test_result/RandomForest_feature_selection_best_params.csv", index=False)

y_true_overall = np.zeros((test_locs.shape[0], len(models)))
y_pred_prob_overall = np.zeros((test_locs.shape[0], len(models)))

for i, (sp_id, model) in enumerate(tqdm(models.items(), desc="Evaluating models with probability outputs")):
    y_true = np.zeros(test_locs.shape[0])
    if sp_id in test_pos_inds:
        y_true[test_pos_inds[sp_id]] = 1

    y_pred_prob = model.predict_proba(test_locs)[:, 1]

    y_true_overall[:, i] = y_true
    y_pred_prob_overall[:, i] = y_pred_prob

print("Calculating Micro Metrics...")

threshold = 0.7
y_pred_overall = (y_pred_prob_overall >= threshold).astype(int)

y_true_flat = y_true_overall.flatten()
y_pred_flat = y_pred_overall.flatten()

overall_precision_micro = precision_score(y_true_flat, y_pred_flat)
overall_recall_micro = recall_score(y_true_flat, y_pred_flat)
overall_f1_micro = f1_score(y_true_flat, y_pred_flat)

precision_macro = []
recall_macro = []
f1_macro = []

for i in tqdm(range(species.shape[0]), desc="Calculating Macro Metrics"):
    y_true = y_true_overall[:, i]
    y_pred = y_pred_overall[:, i]

    precision_macro.append(precision_score(y_true, y_pred))
    recall_macro.append(recall_score(y_true, y_pred))
    f1_macro.append(f1_score(y_true, y_pred))

overall_precision_macro = np.mean(precision_macro)
overall_recall_macro = np.mean(recall_macro)
overall_f1_macro = np.mean(f1_macro)

results = {
    "Micro Precision": overall_precision_micro,
    "Micro Recall": overall_recall_micro,
    "Micro F1-Score": overall_f1_micro,
    "Macro Precision": overall_precision_macro,
    "Macro Recall": overall_recall_macro,
    "Macro F1-Score": overall_f1_macro
}
results = {key: round(value, 3) for key, value in results.items()}

df_results = pd.DataFrame(results.items(), columns=['Metric', 'Value'])
df_results.to_csv("test_result/RandomForest_feature_selection_results.csv", index=False)
print(df_results)
