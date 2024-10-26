# Result

EDA(24/10/25)：[here](./figure)



# Before meeing 3

> 2024/10/25





## Data

* **species_train**: 
  * train_locs: (272073, 2)
    * latitude ,longitude
  * train_ids: (272073,)
  * taxon_ids: (500,)
    * 注意：物种 ID 不一定从 0/1 开始
  * taxon_names: (500,)
* **species_train_extra**: 
  * train_locs: (1067592, 2)
  * train_ids: (1067592,)
  * taxon_ids: (1918,)
  * taxon_names: (1918,)
* **species**_test
  * ['test_locs', 'test_pos_inds', 'taxon_ids', 'taxon_names', 'allow_pickle']
  * test_locs: (288122, 2)
  * test_pos_inds: (500,)
    * lists of list
  * taxon_names: (500,)



## Exploratory Data Analysis

### 1) all_loc

不分物种, 二维坐标.



### 2) all_loc_heatmap

不分物种, 二维坐标, 热力图.



### 3) species_count_distribution

不同物种样本数量分布柱状图.



### 4) species_count_topN_botM

不同物种样本数量排序. 最多和最少的物种的分布图(散点).

* n = 65 (基本都是2000)
* m = 5



### 5) dispersions_convex_hull_area

> 最小凸包面积(Minimum Convex Hull Area)

```python
from scipy.spatial import ConvexHull
```

5 个最大和 5 个最小的物种的分布图 (散点), 及箱线图

* `convex_hull_area(locs)`
* `dispersions_convex_hull_area`
* `plot_convex_hull_boxplot`



# Process Report

> before 24/10/30

## 1) Data Preparation

1. 介绍基础数据
2. 额外的 feature, detailed in 'Learning Methods - Feature Selection' Part.

```latex
The basic data set consists of 272073 samples indicating the location of different species that have been observed with a 2 dimensional geographical coordinates -- \textbr{Latitude} and \textbr{Longitude}. The data was offered by \textit{iNaturalist -  www.inaturalist.org}\cite{iNaturalist}. Moreover, supported by \textit{inaturalist taxa}\cite{iNaturalistTaxaDirectory}, the type of species are recorded in a list with 500 integer taxon IDs and the corresponding names of each species. The final test for our models was under 288122 more data points obtained from \textit{IUCN}'s database\cite{IUCNRedListSpatialData}. In addition, we prepared 1067592 more samples of 1918 species in total as backup for carrying out some potential exploration.

Furthermore, in our later experiment, based on data from WorldClim Bioclimatic, we add 19 more bio-climatic features which is introduced in more detail in 'Learning Methods - Feature Selection' section.
```

citation:

```
# train data source
@misc{iNaturalist,
  title = {iNaturalist},
  year = {2024},
  url = {https://www.inaturalist.org},
  note = {Accessed: 2024-10-26},
}
# test data source
@misc{IUCNRedListSpatialData,
  author = {{International Union for Conservation of Nature}},
  title = {IUCN Red List Spatial Data Download},
  year = {2024},
  url = {https://www.iucnredlist.org/resources/spatial-data-download},
  note = {Accessed: 2024-10-26},
}
# species info source
@misc{iNaturalistTaxaDirectory,
  author = {iNaturalist},
  title = {iNaturalist Taxa Directory},
  year = {2024},
  url = {https://www.inaturalist.org/taxa},
  note = {Accessed: 2024-10-26},
}
```



## 2) Exploratory Data Analysis

* only on training set
* plot all sample points in the form of scatter plots: As it is a two-dimensional coordinate, we can thus intuitively feel the geographical distribution of the species sample in general in this way.
* heat map: a widely used method to perform geographic data intensity.
* Minimum Convex Hull Area (boxplot): 

```latex
We utilise some exploratory data anaylsis methods to help us better understanding the data.
```



# Todo

reference list 的格式应该还要改.
