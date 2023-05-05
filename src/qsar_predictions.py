from numpy.lib.function_base import append
from tdc.single_pred import ADME, Tox
from tdc import Evaluator
from sklearn import svm, neural_network, ensemble, metrics
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import models
from rdkit.Chem import PandasTools, MACCSkeys, AllChem
from rdkit.Avalon import pyAvalonTools 
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
from random import randint
import os, sys
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from joblib import load

## Needed to open autoencoder compressors
from new_metrics import matthews_correlation_coefficient, tanimoto_similarity, cos_similarity
from model import Autoencoder_dense, Autoencoder_dense_multi

np.random.seed(42)

def disable_rdkit_logging():
    """
    Disables RDKit whiny logging.
    """
    import rdkit.rdBase as rkrb
    import rdkit.RDLogger as rkl
    logger = rkl.logger()
    logger.setLevel(rkl.ERROR)
    rkrb.DisableLog('rdApp.error')
disable_rdkit_logging()

COMPRESSED_LENGTHS = {'MACCS': [2,4,8,16,32,64],'avalon': [2,4,8,16,32,64,128,256],  'Morgan_2': [2,4,8,16,32,64,128,256], 'RDK': [2,4,8,16,32,64,128,256]}

NOW = datetime.now().isoformat(sep="_", timespec="minutes")

DBS = {'ADME':ADME, 'Tox':Tox}
DB_DATA = {'ADME':{'hia_hou':'bin', 'bbb_martins':'bin', 'lipophilicity_astrazeneca':'reg', 'hydrationfreeenergy_freesolv':'reg'}}

FPS_DICT = {'MACCS': lambda x: np.array(list(MACCSkeys.GenMACCSKeys(x))), 
            'Morgan_2': lambda x: np.array(list(AllChem.GetMorganFingerprintAsBitVect(x,radius=2))),
            'Morgan_4': lambda x: np.array(list(AllChem.GetMorganFingerprintAsBitVect(x,radius=4))),
            'RDK': lambda x: np.array(list(AllChem.RDKFingerprint(x))),
            'avalon': lambda x: np.array(list(pyAvalonTools.GetAvalonFP(x)))}
PRED_TYPES = {'svm': {'bin': svm.SVC, 'reg': svm.SVR}, 
              'rf': {'bin': ensemble.RandomForestClassifier, 'reg': ensemble.RandomForestRegressor},
              'bdt': {'bin': ensemble.GradientBoostingClassifier, 'reg': ensemble.GradientBoostingRegressor},
               'nn': {'bin': neural_network.MLPClassifier, 'reg': neural_network.MLPRegressor}}
PRED_HYPER = {'svm': [{'C': 0.1,'tol': 1e-5}, {'C': 1,'tol': 1e-5}, {'C': 10,'tol': 1e-5}], 
            'rf': [{'n_estimators': 50}, {'n_estimators': 100}, {'n_estimators': 500}],  
            'bdt': [{'n_estimators': 50}, {'n_estimators': 100}, {'n_estimators': 500}], 
            'nn': [{'hidden_layer_sizes':(100,), 'activation': 'logistic'}, {'hidden_layer_sizes':(100,), 'activation': 'relu'}, {'hidden_layer_sizes':(100,100), 'activation': 'logistic'}, {'hidden_layer_sizes':(100,100), 'activation': 'relu'}]}

custom_objects = {'matthews_correlation_coefficient': matthews_correlation_coefficient, 'tanimoto_similarity': tanimoto_similarity, 'cos_similarity': cos_similarity,
'Autoencoder_dense': Autoencoder_dense, 'Autoencoder_dense_multi': Autoencoder_dense_multi}

def create_model_to_fp_dict(model_paths):
    model_to_fp = {}
    for name in model_paths:
        for fp_str, fp in zip(['MACCS', 'Morgan_2', 'Morgan2', 'Morgan_4', 'RDK', 'avalon'], ['MACCS', 'Morgan_2', 'Morgan_2', 'Morgan_4', 'RDK', 'avalon']):
            if fp_str in name:
                model_to_fp[name] = fp

    fp_to_model = {fp: [] for fp in set(model_to_fp.values())}
    for model in model_to_fp.keys():
        fp_to_model[model_to_fp[model]].append(model)
    return model_to_fp, fp_to_model

def create_double_model_to_fp_dict(model_paths):
    model_to_fp = {}
    for name in model_paths:
        model_to_fp[name] = []
        for fp_str, fp in zip(['MACCS', 'Morgan_2', 'Morgan2', 'Morgan_4', 'RDK', 'avalon'], ['MACCS', 'Morgan_2', 'Morgan_2', 'Morgan_4', 'RDK', 'avalon']):
            if fp_str in name:
                model_to_fp[name].append(fp)
    return model_to_fp

def run_qsar_pred(FP_MODEL_DICT: dict = {}, DOUBLE_MODEL_FP_DICT: dict = {}, BENCH_FPS: list = [], linear_benchmarks: bool = False):

    def run_predictors(data, fp_name, db_type, col, compr_time=0):
        for predictor in PRED_TYPES.keys():
            for hyperparams in PRED_HYPER[predictor]:
                # print(f'Experiment for model {model} on data {col} of type {DB_DATA["ADME"][col]} with predictor {predictor} with hyperparams: {hyperparams}.')
                if (predictor in ['svm', 'nn']) and db_type=='reg':
                    pred = make_pipeline(StandardScaler(), PRED_TYPES[predictor][DB_DATA[db_type][col]](**hyperparams))
                else:
                    pred = PRED_TYPES[predictor][DB_DATA[db_type][col]](**hyperparams)
                FP = data[fp_name].to_list()
                Y = data['Y'].to_list()
                if db_type =='bin':
                    cv = StratifiedKFold(n_splits=10, random_state=42)
                    score = cross_validate(pred, FP, Y, cv=cv)
                else:
                    score = cross_validate(pred, FP, Y, cv=10)
                pred_time = score['fit_time']
                score = score['test_score']
                # print(f'{fp}: Took: {pred_time[fp]} s')
                results.append({'FP': fp_name, 'data': col, 'compr_time': compr_time, 'predictor': predictor, 'hyperparams': hyperparams,
                'cv_score_mean': np.mean(score), 'cv_score_std': np.std(score),  
                'pred_time_mean': np.mean(pred_time), 'pred_time_std': np.std(pred_time)})
    results = []
    for db_type in DB_DATA.keys():
        for col in DB_DATA[db_type].keys():
            data = DBS[db_type](name=col)
            data = data.get_data()
            PandasTools.AddMoleculeColumnToFrame(data,'Drug','Mol')
            for fp in list(FP_MODEL_DICT.keys())+BENCH_FPS:
                data[fp] =  data.Mol.apply(FPS_DICT[fp])
                run_predictors(data, fp, db_type, col)
                # Benchmark compressions PCA and PLS
                if linear_benchmarks:
                    for c_len in COMPRESSED_LENGTHS[fp]:
                        for lin_comp_type in ['PCA', 'PLS']:
                            model_name = f'{lin_comp_type}_{fp}_{c_len}'
                            lin_compr = load(os.path.join('Models_linear/', f'{model_name}.joblib'))
                            t1 = datetime.now()
                            data[model_name] = data[fp].apply(lambda x: lin_compr.transform(np.array([x]))[0])
                            t2 = datetime.now()
                            compr_time = (t2-t1).total_seconds()
                            run_predictors(data, model_name, db_type, col, compr_time)

                # LOOP OVER AE COMPRESSORS PER FP
                # for model in FP_MODEL_DICT[fp]:
                #     if 'model.json' in os.listdir(os.path.join(WHICH_DIR, model)+'/Models/'):
                #         with open(os.path.join(WHICH_DIR, model)+'/Models/model.json', 'r') as w:
                #             string = w.read()
                #         ae_compressor= models.model_from_json(string, custom_objects=custom_objects)
                #         ae_compressor.load_weights(os.path.join(WHICH_DIR, model)+'/Models/final')
                #     else:
                #         ae_compressor = models.load_model(os.path.join(WHICH_DIR, model)+'/Models/final', custom_objects=custom_objects)
                #     def ae_compress(x):
                #         if type(ae_compressor.encoder(np.array([x]))) is tf.Tensor:
                #             return ae_compressor.encoder(np.array([x])).numpy().flatten()
                #         else:
                #             return ae_compressor.encoder(np.array([x])).numpy().flatten()
                #             # return ae_compressor.encoder(np.array([x])).mean().numpy().flatten()
                #     model_name = os.path.split(model)[1]
                #     model_name = model_name.split('_202')[0]
                #     t1 = datetime.now()
                #     data[model_name] = data[fp].apply(ae_compress)
                #     t2 = datetime.now()
                #     compr_time = (t2-t1).total_seconds()
                #     run_predictors(data, model_name, db_type, col, compr_time)
                
            # AFTER ALL FPS WERE CREATED DO ALSO COMPRESSORS OF MIXED FPS
            # for model in DOUBLE_MODEL_FP_DICT.keys():
            #     # check if all fps were already created
            #     for fp in DOUBLE_MODEL_FP_DICT[model]:
            #         if fp not in data.columns:
            #             data[fp] =  data.Mol.apply(FPS_DICT[fp])
            #             run_predictors(data, fp, db_type, col)
            #     if 'model.json' in os.listdir(os.path.join(WHICH_DIR, model)+'/Models/'):
            #         with open(os.path.join(WHICH_DIR, model)+'/Models/model.json', 'r') as w:
            #             string = w.read()
            #         ae_compressor= models.model_from_json(string, custom_objects=custom_objects)
            #         ae_compressor.load_weights(os.path.join(WHICH_DIR, model)+'/Models/final')
            #     else:
            #         ae_compressor = models.load_model(os.path.join(WHICH_DIR, model)+'/Models/final', custom_objects=custom_objects)
            #     def ae_compress(x):
            #         if type(ae_compressor.encoder([np.array([x[0]]), np.array([x[1]])])) is tf.Tensor:
            #             return ae_compressor.encoder([np.array([x[0]]), np.array([x[1]])]).numpy().flatten()
            #         else:
            #             return ae_compressor.encoder([np.array([x[0]]), np.array([x[1]])]).numpy().flatten() # ae_compressor.encoder([np.array([x[0]]), np.array([x[1]])]).mean().numpy().flatten()
            #     model_name = os.path.split(model)[1]
            #     model_name = model_name.split('_202')[0]
            #     t1 = datetime.now()
            #     data[model_name] = data[DOUBLE_MODEL_FP_DICT[model]].apply(ae_compress, axis=1)
            #     t2 = datetime.now()
            #     compr_time = (t2-t1).total_seconds()
            #     run_predictors(data, model_name, db_type, col, compr_time)



    results = pd.DataFrame(results)
    results.to_csv(f'Results/qsar_predictions_{NOW}.csv')
    

if __name__ == '__main__':

    # TO RUN THE USUAL AE COMPRESSIONS

    # WHICH_DIR = sys.argv[1]    

    # ALL_MODELS = os.listdir(WHICH_DIR)
    # DOUBLE_MODELS = [md for md in ALL_MODELS  if 'double' in md]
    # MODELS = [md for md in ALL_MODELS  if 'single' in md]   

    # MODEL_FP_DICT, FP_MODEL_DICT = create_model_to_fp_dict(MODELS)
    # print(MODEL_FP_DICT)
    # DOUBLE_MODEL_FP_DICT = create_double_model_to_fp_dict(DOUBLE_MODELS)
    # print(DOUBLE_MODEL_FP_DICT)
    # print('Starting with QSAR predictions')
    # run_qsar_pred(FP_MODEL_DICT, DOUBLE_MODEL_FP_DICT)

    # TO RUN LINEAR BENCHMARKS
    run_qsar_pred({}, {}, ['Morgan_2'], linear_benchmarks=True)
