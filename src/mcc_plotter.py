import sys, os
import pandas as pd



FP_SIZE = {'MACCS': 167, 'Morgan': 2048, 'RDK': 2048, 'avalon':512 }
for dirname in ['rep1_new', 'rep2_new', 'rep3_new']:
    mcc_ls = []
    for d in [x for x in os.listdir(f'../Results/{dirname}') if 'layers_2' in x and 'single' in x and 'summary' not in x]:
        dict_to_df = {}
        hist = pd.read_csv(f'../Results/{dirname}/'+d+'/history.csv')
        if 'val_matthews_correlation_coefficient' in list(hist.columns):
            dict_to_df.update({'mcc': hist.val_matthews_correlation_coefficient.iloc[-1]})
            dict_to_df.update({'prop': False})
        else:
            dict_to_df.update({'mcc': hist.val_output_1_matthews_correlation_coefficient.iloc[-1]})
            dict_to_df.update({'prop': True})
        d = d.split('_layers')[0].replace('single_fp_', '').replace('_no_prop', '').replace('Morgan_2', 'Morgan').split('_')
        dict_to_df.update({'FP':d[0], 'LS': int(d[1])/FP_SIZE[d[0]]})
        mcc_ls.append(dict_to_df)
    mcc_df = pd.DataFrame(mcc_ls)
    mcc_df.to_csv(f'mcc_{dirname}.csv')
