# !/bin/sh
python ./explain.py -p --pfiles compas.csv,compas ../datasets/tabular/complete/compas/
python ./explain.py -p --pfiles liver-disorder.csv,liver-disorder ../datasets/tabular/complete/liver-disorder/
python ./explain.py -p --pfiles australian.csv,australian ../datasets/tabular/complete/australian/
python ./explain.py -p --pfiles hungarian.csv,hungarian ../datasets/tabular/complete/hungarian/
python ./explain.py -p --pfiles heart-statlog.csv,heart-statlog ../datasets/tabular/complete/heart-statlog/
python ./explain.py -p --pfiles lending.csv,lending ../datasets/tabular/complete/lending/
python ./explain.py -p --pfiles recidivism.csv,recidivism ../datasets/tabular/complete/recidivism/
python ./explain.py -p --pfiles adult.csv,adult ../datasets/tabular/complete/adult/
python ./other/cifar.py
rm -rf ../datasets/cifar-10/32,32/all
