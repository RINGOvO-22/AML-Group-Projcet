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
from scipy.spatial import ConvexHull

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

  plt.title('All Training Locations')
  plt.xlabel('Longitude')
  plt.ylabel('Latitude')
  plt.grid(True)
  plt.tight_layout()

  # 保存图形
  plt.savefig('.\\figure\\all_training_locations.png', dpi=300)  # 保存为PNG文件，dpi为300
  plt.show()

def all_loc_heatmap(train_locs):
  # 绘制热力图
  plt.figure(figsize=(12, 6))
  plt.hist2d(train_locs[:, 1], train_locs[:, 0], bins=100, cmap='hot')  # 设置合适的bin数量和颜色映射

  # 添加颜色栏
  plt.colorbar(label='Location Density')

  plt.title('Heatmap of All Training Locations')
  plt.xlabel('Longitude')
  plt.ylabel('Latitude')
  plt.grid(False)  # 去除网格以便更好地查看热力图
  plt.tight_layout()

  plt.savefig('.\\figure\\all_training_locations_heatmap.png', dpi=300)
  plt.show()

def species_count_distribution(unique_species, species_counts):
  # 定义区间 (每100个样本作为一个区间)
  bins = np.arange(0, 2100, 100)  # 区间：[0, 100), [100, 200), ..., [2000, 2100)

  # 计算每个区间内的物种数量
  hist_counts, bin_edges = np.histogram(species_counts, bins=bins)

  plt.figure(figsize=(10, 6))
  plt.bar(bin_edges[:-1], hist_counts, width=100, align='edge', color='skyblue', edgecolor='black')

  plt.title('Species Count Distribution by Sample Range')
  plt.xlabel('Sample Count Range')
  plt.ylabel('Number of Species')

  # 显示区间标签
  plt.xticks(bins, rotation=45)

  plt.tight_layout()
  plt.savefig('.\\figure\\species_count_distribution.png', dpi=300)
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

  plt.title('Top N Species Locations')
  plt.xlabel('Longitude')
  plt.ylabel('Latitude')
  # plt.legend()
  plt.grid(True)
  plt.tight_layout()
  plt.savefig('.\\figure\\species_count_topN_Loc.png', dpi=300)
  plt.show()

  # 绘制后 m 名的物种位置
  plt.figure(figsize=(12, 6))
  for idx, species_id in enumerate(bottom_m_ids):
      species_locs = train_locs[train_ids == species_id]  # 直接索引 train_locs
      plt.scatter(species_locs[:, 1], species_locs[:, 0], color=colors_bot(idx), alpha=0.6, label=f'Species ID {species_id}')

  plt.title('Bottom M Species Locations')
  plt.xlabel('Longitude')
  plt.ylabel('Latitude')
  # plt.legend()
  plt.grid(True)
  plt.tight_layout()
  plt.savefig('.\\figure\\species_count_bottomM_loc.png', dpi=300)
  plt.show()

# 计算每个物种的最小凸包面积
def convex_hull_area(locs):
    if locs.shape[0] < 3:
        return 0  # 若点数少于3，无法形成凸包
    hull = ConvexHull(locs)
    return hull.volume  # 面积

def plot_convex_hull_boxplot(dispersions):
    areas = list(dispersions.values())  # 所有物种的凸包面积

    # 创建箱线图
    plt.figure(figsize=(10, 6))
    plt.boxplot(areas, vert=False, patch_artist=True, boxprops=dict(facecolor='skyblue', color='blue'))
    plt.title('Box Plot of Convex Hull Areas Across All Species')
    plt.xlabel('Convex Hull Area')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('.\\figure\\species_convex_hull_area_boxplot.png', dpi=300)
    plt.show()

def dispersions_convex_hull_area(species, train_locs, train_ids):
  # 计算每个物种的凸包面积
  dispersions = {species: convex_hull_area(train_locs[train_ids == species]) for species in unique_species}
  plot_convex_hull_boxplot(dispersions)

  # 按凸包面积排序
  sorted_dispersions = sorted(dispersions.items(), key=lambda x: x[1], reverse=True)

  # 设定绘图显示的数量
  top_n = 5  # 最大凸包面积的前 n 名
  bottom_m = 5  # 最小凸包面积的后 m 名
  top_species = sorted_dispersions[:top_n]
  bottom_species = sorted_dispersions[-bottom_m:]

  # 绘制最大和最小凸包面积的物种地理分布
  fig, axes = plt.subplots(2, 1, figsize=(12, 10))

  # 绘制凸包面积最大的前 n 个物种
  axes[0].set_title(f'Top {top_n} Species with Largest Convex Hull Area')
  for species, area in top_species:
      species_locs = train_locs[train_ids == species]
      axes[0].scatter(species_locs[:, 1], species_locs[:, 0], label=f'Species ID {species} (Area: {area:.2f})')
  axes[0].grid(True)

  # 绘制凸包面积最小的后 m 个物种
  axes[1].set_title(f'Bottom {bottom_m} Species with Smallest Convex Hull Area')
  for species, area in bottom_species:
      species_locs = train_locs[train_ids == species]
      axes[1].scatter(species_locs[:, 1], species_locs[:, 0], label=f'Species ID {species} (Area: {area:.2f})')
  axes[1].grid(True)

  for ax in axes:
      ax.set_xlabel('Longitude')
      ax.set_ylabel('Latitude')
      ax.legend(loc='best')
  plt.tight_layout()
  plt.savefig('.\\figure\\dispersions_convex_hull_area.png', dpi=300)
  plt.show()

# main body

# random_dpecies(species, species_names, test_pos_inds, test_locs, train_ids, train_locs)
# all_loc(train_locs)
# all_loc_heatmap(train_locs)
# species_count_distribution(unique_species, species_counts)
# species_count_topN_botM(unique_species, species_counts)
dispersions_convex_hull_area(species, train_locs, train_ids)
