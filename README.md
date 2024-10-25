# Result

EDA：./figure



# Before meeing 3

> 2024/10/25



## Exploratory Data Analysis

### 1) Data

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



### 2) all_loc

不分物种, 二维坐标.



### 3) all_loc_heatmap

不分物种, 二维坐标, 热力图.



### 4) species_count_distribution

不同物种样本数量分布柱状图.



### 5) species_count_topN_botM

不同物种样本数量排序. 最多和最少的物种的分布图(散点).

* n = 65 (基本都是2000)
* m = 5



### 6) dispersions_convex_hull_area

> 最小凸包面积(Minimum Convex Hull Area)

```python
from scipy.spatial import ConvexHull
```

5 个最大和 5 个最小的物种的分布图 (散点), 及箱线图

* `convex_hull_area(locs)`
* `dispersions_convex_hull_area`
* `plot_convex_hull_boxplot`
