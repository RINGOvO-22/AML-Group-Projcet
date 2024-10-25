"""
Script demonstrating simple data loading and visualization.

Data Format: 
There are two files 'species_train.npz', and 'species_test.npz'
For the train data, we have the geographical coordinates where different 
species have been observed. This data has been collected by citizen scientists 
so it is noisy. 
For the test data, we have a set of locations for all species from the training, 
set and for each location we know if a species is present there or not. 

You can find out information about each species by appending the taxon_id to this 
URL, e.g. for 22956: 'Leptodactylus mystacinus', the URL is: 
https://www.inaturalist.org/taxa/22956
note some species might not be on the website anymore

Possible questions to explore: 
 - train a separate model to predict what locations a species of interest is present 
 - train a single model instead of per species model 
 - how to deal with "positive only data"
 - dealing with noisy/biased training data
 - using other input features e.g. climate data from  WorldClim Bioclimatic 
   variables  https://www.worldclim.org/data/worldclim21.html
 - how to evaluate e.g. what is a good metric to use?
 
Data sources:
 -  train data is from iNaturalist -  www.inaturalist.org
 -  test data is IUCN - https://www.iucnredlist.org/resources/spatial-data-download
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# loading training data    
data = np.load('species_train.npz')
train_locs = data['train_locs']  # 2D array, rows are number of datapoints and 
                                 # columns are "latitude" and "longitude"
train_ids = data['train_ids']    # 1D array, entries are the ID of the species 
                                 # that is present at the corresponding location in train_locs
species = data['taxon_ids']      # list of species IDe. Note these do not necessarily start at 0 (or 1)
species_names = dict(zip(data['taxon_ids'], data['taxon_names']))  # latin names of species 

# loading test data 
data_test = np.load('species_test.npz', allow_pickle=True)
test_locs = data_test['test_locs']    # 2D array, rows are number of datapoints 
                                      # and columns are "latitude" and "longitude"
# data_test['test_pos_inds'] is a list of lists, where each list corresponds to 
# the indices in test_locs where a given species is present, it can be assumed 
# that they are not present in the other locations
test_pos_inds = dict(zip(data_test['taxon_ids'], data_test['test_pos_inds']))    

# data stats
print('Train Stats:')
print('Number of species in train set:           ', len(species))
print('Number of train locations:                ', train_locs.shape[0])
unique_species, species_counts = np.unique(train_ids, return_counts=True)
print('Average number of locations per species:  ', species_counts.mean())
print('Minimum number of locations for a species:', species_counts.min())
print('Maximum number of locations for a species:', species_counts.max())

def random_dpecies(species, species_names, test_pos_inds, test_locs, train_ids, train_locs):
  # plot train and test data for a random species
  plt.close('all')
  plt.figure(0)

  sp = np.random.choice(species)
  print('\nDisplaying random species:')
  print(str(sp) + ' - ' + species_names[sp]) 

  # get test locations and plot
  # test_inds_pos is the locations where the selected species is present
  # test_inds_neg is the locations where the selected species is not present
  test_inds_pos = test_pos_inds[sp]  
  test_inds_neg = np.setdiff1d(np.arange(test_locs.shape[0]), test_pos_inds[sp])
  plt.plot(test_locs[test_inds_pos, 1], test_locs[test_inds_pos, 0], 'b.', label='test')

  # get train locations and plot
  train_inds_pos = np.where(train_ids == sp)[0]
  plt.plot(train_locs[train_inds_pos, 1], train_locs[train_inds_pos, 0], 'rx', label='train')

  plt.title(str(sp) + ' - ' + species_names[sp])
  plt.grid(True)
  plt.xlim([-180, 180])
  plt.ylim([-90, 90])
  plt.ylabel('latitude')
  plt.xlabel('longitude')
  plt.legend()
  plt.show()

def all_loc(train_locs):
  # 绘制所有训练数据的地理位置
  plt.figure(figsize=(12, 6))
  plt.scatter(train_locs[:, 1], train_locs[:, 0], color='blue', alpha=0.5, s=10)  # s 控制点的大小

  # 图形美化
  plt.title('All Training Locations')
  plt.xlabel('Longitude')
  plt.ylabel('Latitude')
  plt.grid(True)
  plt.tight_layout()

  # 保存图形
  plt.savefig('.\\figure\\all_training_locations.png', dpi=300)  # 保存为PNG文件，dpi为300
  plt.show()

def topN_botM(unique_species, species_counts):
  # 设定前 n 名和后 m 名
  n = 65  # 前 n 名
  m = 10  # 后 m 名

  # 获取按出现次数排序后的索引
  sorted_indices = np.argsort(species_counts)

  # 获取出现次数最多的前 n 名的物种ID及其出现次数
  top_n_ids = unique_species[sorted_indices[-n:]]
  top_n_counts = species_counts[sorted_indices[-n:]]

  # 获取出现次数最少的后 m 名的物种ID及其出现次数
  bottom_m_ids = unique_species[sorted_indices[:m]]
  bottom_m_counts = species_counts[sorted_indices[:m]]

  print(f"\n前 {n} 名出现次数最多的物种ID及其出现次数: {list(zip(top_n_ids, top_n_counts))}")
  print(f"后 {m} 名出现次数最少的物种ID及其出现次数: {list(zip(bottom_m_ids, bottom_m_counts))}\n")

  # Plot
  # 提取对应的地理位置
  top_n_locs = train_locs[np.isin(train_ids, top_n_ids)]
  bottom_m_locs = train_locs[np.isin(train_ids, bottom_m_ids)]

  # 颜色映射
  colors = plt.cm.get_cmap('tab10', len(top_n_ids))  # 对于前 n 名
  colors_bot = plt.cm.get_cmap('tab10', len(bottom_m_ids))  # 对于后 m 名

  # 绘制前 n 名的物种位置
  plt.figure(figsize=(12, 6))
  for idx, species_id in enumerate(top_n_ids):
      species_locs = train_locs[train_ids == species_id]  # 直接索引 train_locs
      plt.scatter(species_locs[:, 1], species_locs[:, 0], color=colors(idx), alpha=0.6, label=f'Species ID {species_id}')

  # 图形美化
  plt.title('Top N Species Locations')
  plt.xlabel('Longitude')
  plt.ylabel('Latitude')
  # plt.legend()
  plt.tight_layout()
  plt.savefig('.\\figure\\topN_Loc.png', dpi=300)
  plt.show()

  # 绘制后 m 名的物种位置
  plt.figure(figsize=(12, 6))
  for idx, species_id in enumerate(bottom_m_ids):
      species_locs = train_locs[train_ids == species_id]  # 直接索引 train_locs
      plt.scatter(species_locs[:, 1], species_locs[:, 0], color=colors_bot(idx), alpha=0.6, label=f'Species ID {species_id}')

  # 图形美化
  plt.title('Bottom M Species Locations')
  plt.xlabel('Longitude')
  plt.ylabel('Latitude')
  # plt.legend()
  plt.tight_layout()
  plt.savefig('.\\figure\\bottomM_loc.png', dpi=300)
  plt.show()

# main body
# random_dpecies(species, species_names, test_pos_inds, test_locs, train_ids, train_locs)
all_loc(train_locs)
# topN_botM(unique_species, species_counts)
