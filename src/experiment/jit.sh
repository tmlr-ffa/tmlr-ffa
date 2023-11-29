# !/bin/sh
python -u ./jit/explain.py -vv -C ./jit/models/openstack_LR_global_model.pkl --nof-inst 500 -i ../datasets/jit/openstack_X_test.csv -N all openstack > ../logs/jit/openstack_LR_formal.log
python -u ./jit/explain.py -vv -C ./jit/models/qt_LR_global_model.pkl --nof-inst 500 -i ../datasets/jit/qt_X_test.csv -N all qt > ../logs/jit/qt_LR_formal.log
python ./jit/hexp.py openstack LR lime 500 > ../logs/jit/openstack_LR_lime.log
python ./jit/hexp.py qt LR lime 500 > ../logs/jit/qt_LR_lime.log
python ./jit/hexp.py openstack LR shap 500 > ../logs/jit/openstack_LR_shap.log
python ./jit/hexp.py qt LR shap 500 > ../logs/jit/qt_LR_shap.log
