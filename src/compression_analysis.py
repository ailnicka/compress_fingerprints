from cmath import sin
import numpy as np
import pandas as pd
from rdkit.Chem import MACCSkeys
from rdkit.Chem import AllChem
from rdkit.Avalon import pyAvalonTools 
import sys, os
from datetime import date, datetime, timedelta
from numpy.lib.function_base import append
import scipy
import tensorflow as tf
from tensorflow.keras import models
import numpy as np
from scipy.stats import rv_histogram

from new_metrics import matthews_correlation_coefficient, tanimoto_similarity, cos_similarity
from model import Autoencoder_dense, Autoencoder_dense_multi, Variational_Autoencoder_dense, Variational_Autoencoder_dense_multi

custom_objects = {'matthews_correlation_coefficient': matthews_correlation_coefficient, 'tanimoto_similarity': tanimoto_similarity, 'cos_similarity': cos_similarity,
'Autoencoder_dense': Autoencoder_dense, 'Autoencoder_dense_multi': Autoencoder_dense_multi, 'Variational_Autoencoder_dense': Variational_Autoencoder_dense, 'Variational_Autoencoder_dense_multi': Variational_Autoencoder_dense_multi}

WHICH_DATA = sys.argv[1]
WHICH_DIR= sys.argv[2]

BINS = int(sys.argv[3])

FPS_DICT = {'MACCS': lambda x: np.array(list(MACCSkeys.GenMACCSKeys(x))),
            'Morgan_2': lambda x: np.array(list(AllChem.GetMorganFingerprintAsBitVect(x,radius=2))),
            'RDK': lambda x: np.array(list(AllChem.RDKFingerprint(x))),
            'avalon': lambda x: np.array(list(pyAvalonTools.GetAvalonFP(x)))}

MODELS = os.listdir(WHICH_DIR)
MODELS = [md for md in MODELS if 'single' in md]

MODELS_DICT = {}
for fp in FPS_DICT:
    MODELS_DICT[fp] = [md for md in MODELS if fp in md]

data = pd.read_csv(f'data/{WHICH_DATA}.txt', header=None, names=['smiles'])
data['mol'] = data['smiles'].apply(AllChem.MolFromSmiles)
data.drop(columns=['smiles'], inplace=True)
MOLS = data['mol'].to_list()
del data

def entropy_from_histogram(data, bins):
    '''
    H = sum(i - vec dirs) log(bin_width_i) - sum (j - hist bins) p_ij log(p_ij)
    p = counts_in_bin/#samples
    '''
    data = np.array([np.array(x) for x in data])
    n_sample, len_fp = data.shape
    ent = 0
    for i in range(len_fp):
        dt = data[:, i]
        min_dt = min(dt)
        max_dt = max(dt)
        hist, edges = np.histogram(dt, bins=bins, range=(min_dt,max_dt), density=False)
        width = np.abs(edges[0]-edges[1])
        ent += np.log(width)
        hist_dist = rv_histogram((hist, edges))
        ent += hist_dist.entropy()
    return ent
    

def unique_and_diversity(data):
    data = np.array([np.array(x) for x in data])
    n_sample, len_fp = data.shape
    rep_vecs = []
    rep_counts = []
    for i in range(n_sample):
        for j in range(n_sample):
            if i != j:
                if np.allclose(data[i, :], data[j,:]):
                    for idx, vec in rep_vecs:
                        if np.allclose(data[i, :], vec):
                            rep_counts[idx] += 1
                            break
                        if idx == (len(rep_counts) - 1):
                            rep_vecs.append(vec)
                            rep_counts.append(1)
    multiple_occurances = np.sum(rep_counts)
    single_occurances = n_sample - multiple_occurances
    unique = (single_occurances + len(rep_vecs))/n_sample
    div_idx = single_occurances* (- 1/n_sample * np.log2(1/n_sample))
    for cnt in rep_counts:
        div_idx += (- cnt/n_sample * np.log2(cnt/n_sample))
    return unique, div_idx 


compression_summary_list = []
for fp in FPS_DICT:
    fp_list = [FPS_DICT[fp](x) for x in MOLS]
    for model in MODELS_DICT[fp]:
        compr_dict = {}
        if 'model.json' in os.listdir(os.path.join(WHICH_DIR, model)+'/Models/'):
            with open(os.path.join(WHICH_DIR, model)+'/Models/model.json', 'r') as w:
                string = w.read()
            ae_compressor= models.model_from_json(string, custom_objects=custom_objects)
            ae_compressor.load_weights(os.path.join(WHICH_DIR, model)+'/Models/final')
        else:
            ae_compressor = models.load_model(os.path.join(WHICH_DIR, model)+'/Models/final', custom_objects=custom_objects)
        
        def ae_compress(x):
            return ae_compressor.encoder(np.array([x])).numpy().flatten()

        model_name = os.path.split(model)[1]
        model_name = model_name.split('_202')[0]
        prop = False if 'no_prop' in model_name else True
        compressed_list = [ae_compress(x) for x in fp_list]
        compr_len = len(compressed_list[0])
        compr_dict['FP'] = fp
        compr_dict['prop'] = prop
        compr_dict['compr_len'] = compr_len
        scipy_entropy = entropy_from_histogram(compressed_list, BINS)
        compr_dict['bins'] = BINS
        compr_dict['scipy_entropy'] = scipy_entropy
        compression_summary_list.append(compr_dict)

df = pd.DataFrame(compression_summary_list)
df.to_csv(f'Results/compression_analysis_{WHICH_DATA}_{BINS}_{datetime.now().isoformat()}.csv')



