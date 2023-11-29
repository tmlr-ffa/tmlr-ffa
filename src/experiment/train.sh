# !/bin/sh
python ./explain.py -o ./btmodels/pima/ --testsplit 0 -t -n 25 -d 3 ../datasets/tabular/train/pima/pima_train.csv
python ./explain.py -o ./btmodels/compas/ -c --testsplit 0 -t -n 25 -d 3 ../datasets/tabular/train/compas/compas_train_data.csv
python ./explain.py -o ./btmodels/liver-disorder/ -c --testsplit 0 -t -n 25 -d 3 ../datasets/tabular/train/liver-disorder/liver-disorder_train_data.csv
python ./explain.py -o ./btmodels/australian/ -c --testsplit 0 -t -n 25 -d 3 ../datasets/tabular/train/australian/australian_train_data.csv
python ./explain.py -o ./btmodels/hungarian/ -c --testsplit 0 -t -n 25 -d 3 ../datasets/tabular/train/hungarian/hungarian_train_data.csv
python ./explain.py -o ./btmodels/heart-statlog/ -c --testsplit 0 -t -n 25 -d 3 ../datasets/tabular/train/heart-statlog/heart-statlog_train_data.csv
python ./explain.py -o ./btmodels/lending/ -c --testsplit 0 -t -n 25 -d 3 ../datasets/tabular/train/lending/lending_train_data.csv
python ./explain.py -o ./btmodels/recidivism/ -c --testsplit 0 -t -n 25 -d 3 ../datasets/tabular/train/recidivism/recidivism_train_data.csv
python ./explain.py -o ./btmodels/appendicitis/ --testsplit 0 -t -n 25 -d 3 ../datasets/tabular/train/appendicitis/appendicitis_train.csv
python ./explain.py -o ./btmodels/cars/ --testsplit 0 -t -n 25 -d 3 ../datasets/tabular/train/cars/cars_train.csv
python ./explain.py -o ./btmodels/adult/ -c --testsplit 0 -t -n 25 -d 3 ../datasets/tabular/train/adult/adult_train_data.csv
python ./explain.py -o ./btmodels/mnist/10,10/1,3/ -c --testsplit 0 -t -n 25 -d 3 ../datasets/mnist/10,10/1,3/train_origin_data.csv
python ./explain.py -o ./btmodels/mnist/10,10/1,7/ -c --testsplit 0 -t -n 25 -d 3 ../datasets/mnist/10,10/1,7/train_origin_data.csv
python ./explain.py -o ./btmodels/pneumoniamnist/10,10/ --testsplit 0 -t -n 25 -d 3 ../datasets/pneumoniamnist/10,10/train_origin.csv
python ./explain.py -o ./btmodels/mnist/28,28/1,3/ -c --testsplit 0 -t -n 25 -d 3 ../datasets/mnist/28,28/1,3/train_origin_data.csv
python ./explain.py -o ./btmodels/mnist/28,28/1,7/ -c --testsplit 0 -t -n 25 -d 3 ../datasets/mnist/28,28/1,7/train_origin_data.csv
python ./explain.py -o ./btmodels/pneumoniamnist/28,28/ --testsplit 0 -t -n 25 -d 3 ../datasets/pneumoniamnist/28,28/train_origin.csv
python ./jit/explain.py -t openstack 
python ./jit/explain.py -t qt
python ./explain.py -o ./btmodels/cifar-10/32,32/ship,truck/ -c --testsplit 0 -t -n 50 -d 3 ../datasets/cifar-10/32,32/ship,truck/train_ori_data.csv
