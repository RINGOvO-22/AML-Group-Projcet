import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans


def find_optimal_k(train_locs, max_k=50):
    inertia = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(train_locs)
        inertia.append(kmeans.inertia_)

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, max_k + 1), inertia, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True)
    plt.savefig("graph/Elbow Method for Optimal k.png")
    plt.close()


data = np.load('species_train.npz')
train_locs = data['train_locs']
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
    
find_optimal_k(train_locs, max_k=50)
