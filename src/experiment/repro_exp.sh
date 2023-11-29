#!/bin/sh

# Reproducing experimental results
# Prepare datasets. The prepared datasets are stored in ```datasets/```
./experiment/prepare.sh; python ./other/split.py

# Train gradient boosted trees and logistic regression models. The generated BT modesl are stored in ```src/btmodels/```, while the generated logistic regression models are stored in ```src/jit/models/```.
./experiment/train.sh

# Image AXp's enumeration. Logs are are stored in ```logs/img/```.
./experiment/img1010.sh; ./experiment/img2828.sh; ./experiment/cifar.sh

# AXp's enumeration in tabular data. Logs are are stored in ```logs/tab/```.
./experiment/tabular.sh

# Feature attribution in images and tabular in LIME and SHAP. Logs are are stored in ```logs/hexp/tab/``` and ```logs/hexp/img/```.
./experiment/hexp_img.sh; ./experiment/hexp_tab.sh

# JIT application. Logs are stored in ```logs/jit/```.
./experiment/jit.sh 

# Parse AXp's, LIME's, and SHAP's image logs. The statistics are stored in ```stats/img/```. 
python ./other/parse_img_formal_logs.py; python ./other/parse_img_formal_logs_cifar.py; python ./other/parse_img_hexp_logs.py

# Parse AXp's, LIME's, and SHAP's tabular data logs. The statistics are stored in ```stats/tab/```. 
python ./other/parse_tab_logs.py

# Parse JIT logs. The statistics are stored in ```stats/jit/```. 
python ./other/parse_jit_logs.py

# Produce csv files regarding metrics. The csv files are stored in ```stats/tables/img/```,  ```stats/tables/tab/```, and  ```stats/tables/jit/```. 
python ./other/produce_stats_csv.py; python ./other/produce_stats_cifar.py

# Produce feature attribution plots and tables. Plots are stored in ```plots/img/```, and tables are store in ```stats/latex/```.
python ./other/plot_table.py; python ./other/plot_table_cifar.py; python ./other/plot_table_cifar.py
