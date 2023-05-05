import os, sys
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from data_processing import split_train_test_and_calculate_FPs, PROP_DICT, FPS_DICT, FPS_2_LEN
from qsar_predictions import COMPRESSED_LENGTHS
from joblib import dump

print('Loaded all', flush=True)

fp = sys.argv[1]

if fp not in ['MACCS', 'Morgan_2', 'RDK', 'avalon']:
    raise KeyError(f'Argument must be from this list: MACCS, Morgan_2, RDK, avalon')

def bitstr_to_array(bitstr):
    return np.array([int(x) for x in bitstr])


modeldir = 'Models_linear/'
os.makedirs(modeldir , exist_ok=True)
train_set, _ , _ = split_train_test_and_calculate_FPs('data/BiggerSetFromChEMBL.txt', split=1.0, fps=[fp], props=list(PROP_DICT.keys()))  
print('Training set completed', flush=True)
for n_comp in [1024]:
    fp_array = np.array([bitstr_to_array(bitstr) for bitstr in train_set[fp]])
    pls = PLSRegression(n_components=n_comp)
    targets = np.array([train_set[p] for p in list(PROP_DICT.keys())]).T
    print(f'PLS fitting {targets.shape}', flush=True)
    pls.fit(X=fp_array, Y=targets)
    print(f'PLS fitted {fp_array.shape}', flush=True)
    dump(pls, os.path.join(modeldir, f'PLS_{fp}_{n_comp}.joblib'))
    print(f'PLS_{fp}_{n_comp} completed', flush=True)

    pca = PCA(n_components=n_comp, svd_solver='full')
    print(f'PCA fitting {fp_array.shape}', flush=True)
    pca.fit(X=fp_array)
    print(f'PCA fitted {fp_array.shape}', flush=True)
    dump(pca, os.path.join(modeldir, f'PCA_{fp}_{n_comp}.joblib'))
    print(f'PCA_{fp}_{n_comp} completed', flush=True)