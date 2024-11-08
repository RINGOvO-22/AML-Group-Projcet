import numpy as np
import rasterio
from tqdm import tqdm

data = np.load('species_train.npz')
train_locs = data['train_locs']

data_test = np.load('species_test.npz', allow_pickle=True)
test_locs = data_test['test_locs']

for i in range(1, 20):
    filename = "wc2.1_2.5m_bio/wc2.1_2.5m_bio_" + str(i) + ".tif"
    worldclim = rasterio.open(filename)
    train_results = np.zeros(train_locs.shape[0])
    test_results = np.zeros(test_locs.shape[0])

    train_lats = np.array([elem[0] for elem in train_locs])
    train_lons = np.array([elem[1] for elem in train_locs])
    coords = np.column_stack((train_lons, train_lats))
    transformed_train_coords = np.array([worldclim.index(lon, lat) for lon, lat in coords])
    data = worldclim.read(1)
    values = data[transformed_train_coords[:, 0], transformed_train_coords[:, 1]]

    for i, value in enumerate(tqdm(values, desc="Processing train data")):
        train_results[i] = value

    test_lats = np.array([elem[0] for elem in test_locs])
    test_lons = np.array([elem[1] for elem in test_locs])
    test_coords = np.column_stack((test_lons, test_lats))
    transformed_test_coords = np.array([worldclim.index(lon, lat) for lon, lat in test_coords])
    data = worldclim.read(1)
    test_values = data[transformed_test_coords[:, 0], transformed_test_coords[:, 1]]

    for j, value in enumerate(tqdm(test_values, desc="Processing train data")):
        test_results[j] = value

    np.save(filename[:-4] + "_train" + '.npy', train_results)
    np.save(filename[:-4] + "_test" + '.npy', test_results)
