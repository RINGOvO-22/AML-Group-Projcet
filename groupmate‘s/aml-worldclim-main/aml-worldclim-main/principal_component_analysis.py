import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import stats

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

# get this file's name
filepath = __file__.split("/")[-1].split(".")[0]
filename = filepath.split("\\")[-1]

for i in range(1, 20):
    filename_train = "wc2.1_10m_bio/wc2.1_10m_bio_" + str(i) + "_train.npy"
    filename_test = "wc2.1_10m_bio/wc2.1_10m_bio_" + str(i) + "_test.npy"
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

pca = PCA(n_components=0.95)
train_locs_pca = pca.fit_transform(train_locs)
test_locs_pca = pca.transform(test_locs)

print("原始特征数:", train_locs.shape[1])
print("PCA后的特征数:", train_locs_pca.shape[1])