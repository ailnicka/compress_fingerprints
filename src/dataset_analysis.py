import numpy as np
import pandas as pd
from rdkit.Chem import MACCSkeys
from rdkit.Chem import AllChem
from rdkit.Avalon import pyAvalonTools 
import sys
from datetime import date

FPS_DICT = {'MACCS': lambda x: np.array(list(MACCSkeys.GenMACCSKeys(x))),
            'Morgan_2': lambda x: np.array(list(AllChem.GetMorganFingerprintAsBitVect(x,radius=2))),
            'RDK': lambda x: np.array(list(AllChem.RDKFingerprint(x))),
            'avalon': lambda x: np.array(list(pyAvalonTools.GetAvalonFP(x)))}

filename = sys.argv[1]

data = pd.read_csv(f'../data/{filename}.txt', header=None, names=['smiles'])
data['mol'] = data['smiles'].apply(AllChem.MolFromSmiles)
def nparray_to_str(y):
    return ''.join((str(x) for x in y))

def calculate_shannon(data, entry):
    df = pd.DataFrame()
    deno = len(data)
    df[entry] = data.groupby(entry)[entry].nunique()
    df['p'] = df[entry].apply(lambda x: x/deno)
    df['-logp'] = df['p'].apply(lambda x: - np.log2(x))
    df['-plogp'] = df['p']*df['-logp']
    shannon = df['-plogp'].sum()
    unique = len(df)/deno
    del df
    return shannon, unique

def calculate_new_shannon(data, entry):
    fps_numpy = data[entry].to_numpy()
    fps_numpy = np.array([np.array(x) for x in fps_numpy])
    numb_ex, len_fp = fps_numpy.shape
    shannon=0
    for i in range(len_fp): 
        pi = fps_numpy[:,i].sum()/numb_ex
        qi = 1 - pi
        if pi == 0 or qi == 0:
            shannon_i = 0
        else:
            shannon_i = -pi*np.log2(pi) - qi*np.log2(qi)
        shannon += shannon_i
    del fps_numpy
    return shannon

summary_dict = {}
for fps in FPS_DICT:
    fp_dict = {}
    data[fps] = data['mol'].apply(FPS_DICT[fps])
    new_shannon = calculate_new_shannon(data, fps)
    fp_dict['new_shannon'] = new_shannon
    data[fps+'_on_bits'] = data[fps].apply(np.sum)
    fp_len = len(data[fps][0])
    fp_dict['on_bits_mean'] = data[fps+"_on_bits"].mean()/fp_len
    fp_dict['on_bits_std'] = data[fps+"_on_bits"].std()/fp_len
    data.drop(columns=fps+'_on_bits', inplace=True)
    data[fps] = data[fps].apply(nparray_to_str)
    shannon, unique = calculate_shannon(data, fps)
    fp_dict['diversity_shannon'] = shannon
    fp_dict['unique'] = unique
    summary_dict[fps] = fp_dict
    data.drop(columns=fps, inplace=True)

df_summary = pd.DataFrame(summary_dict)
df_summary.to_csv(f'data/dataset_analysis_{filename}_{date.today().isoformat()}.csv')