## Before meeing 3

> 2024/10/25



### Exploratory Data Analysis

#### 1) data

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



#### 2) topN_botM



####  打算：

* 找 loc 数量最多的几个物种
  * 同理，最少的
* 分布图（无差别 plot 坐标）
* 可能：箱线图，同类物种的 noise location



