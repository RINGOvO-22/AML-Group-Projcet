import numpy as np
import os
from sklearn.preprocessing import StandardScaler

# load training data and test data into the program
def load():
    # load the npz file
    filepath_train = os.path.join('species_train.npz') # species_train / species_train_extra
    train_data = np.load(filepath_train)
    filepath_test = os.path.join('species_test.npz')
    test_data = np.load(filepath_test, allow_pickle=True)
    return train_data, test_data

# standarize the data about location for both train_data and test_data with a single StandardScaler
def loc_standardized(train_data, test_data):
    # Standardization as a whole
    scaler = StandardScaler()
    train_locs_standardized = scaler.fit_transform(train_data['train_locs'])
    test_locs_standardized = scaler.fit_transform(test_data['test_locs'])

    # print("=====================================================\n")
    # print(f"\nStandardized Array name: train_locs")
    # print(f"Standardized Array content: {train_locs_standardized}")
    # print(f"Standardized Array shape: {train_locs_standardized.shape}\n")

    # print("=====================================================\n")
    # print(f"\nStandardized Array name: test_locs")
    # print(f"Standardized Array content: {test_locs_standardized}")
    # print(f"Standardized Array shape: {test_locs_standardized.shape}\n")

    return train_locs_standardized, test_locs_standardized

# train_data: ['train_locs', 'train_ids', 'taxon_ids', 'taxon_names']
# test_data: ['test_locs', 'test_pos_inds', 'taxon_ids', 'taxon_names', 'allow_pickle']
def checkData(data):
    print(data.files)
    for key in data.files: # test_data
        print("=====================================================\n")
        print(f"Array name: {key}")
        print(f"Array content: {data[key]}")
        print(f"Array shape: {data[key].shape}\n")
        
# check if all elements of "train_locs" appear in "taxon_ids"
# return the filterd arrays "train_locs_filtered, train_ids_filtered"
# N.B., we found that all elements are valid in "train_locs" from the test file after checking!
def check_trainID_taxonID(train_data):
    train_locs = train_data['train_locs']
    train_ids = train_data['train_ids']
    train_taxon_ids = train_data['taxon_ids']

    # check if every element of train_ids is in taxon_ids
    mask = np.isin(train_ids, train_taxon_ids)

    # use mask to reserve train_ids with corresponding train_locs
    train_ids_filtered = train_ids[mask]
    train_locs_filtered = train_locs[mask]
    # check the deleted sample id
    removed_train_ids = train_ids[~mask]

    if removed_train_ids.size == 0:
        print("All data in 'train_ids' are valid !")
    else:
        print("Filtered train_ids:", train_ids_filtered)
        print("Filtered train_locs:", train_locs_filtered)
        print("Removed train_ids:", removed_train_ids)

    return train_locs_filtered, train_ids_filtered

# DBScan
def dbscan():
    return

# main body
train_data, test_data = load()

# print("Train_data:")
# checkData(train_data)
print("\nTest_data:")
checkData(test_data)

# train_locs_standardized, test_locs_standardized = loc_standardized(train_data, test_data)
# train_locs, train_ids = check_trainID_taxonID(train_data)