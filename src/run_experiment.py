from ast import literal_eval
import argparse
import configparser
import datetime
import os
import random
from shutil import copy2
import tensorflow as tf
import numpy as np
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)

from data_processing import split_train_test_and_calculate_FPs, FingerprintsGenerator, FPS_2_LEN, PROP_DICT
from model import Autoencoder_dense_multi, Autoencoder_dense, Variational_Autoencoder_dense, Variational_Autoencoder_dense_multi
from training import compile_model, train_model

parser = argparse.ArgumentParser(description='Autoencoder compression of molecular fingerprints')
parser.add_argument('-c','--config', type=str, help='Path to the config file', required=True)
parser.add_argument('-v','--verbose', help='Verbose', action='store_true', default=False)
parser.add_argument('-rs', '--randomseed', help='Random seed to set', type=int, default=None)
args = vars(parser.parse_args())

verbose = args['verbose']
config_file = args['config']
rnd_seed = args['randomseed']
exp_dir = f'Results/{config_file.split("/")[-1].replace(".ini","").replace("input_", "")}_{datetime.datetime.now().isoformat(sep="_", timespec="minutes")}'
os.makedirs(exp_dir+'/Models', exist_ok=True)
copy2(config_file, exp_dir)
config = configparser.ConfigParser()
config.read(config_file)

if rnd_seed is not None:
    os.environ['PYTHONHASHSEED'] = str(rnd_seed)
    random.seed(rnd_seed)
    tf.random.set_seed(rnd_seed)
    np.random.seed(rnd_seed)

# READING CONFIG
if verbose:
    t_zero = datetime.datetime.now()
    print(f"RUNINFO: Starting config processing at: {t_zero}")

data_file = config['DATA']['file']
fps = literal_eval(config['DATA']['fingerprints'])
props = literal_eval(config['DATA'].get('properties', 'None'))  
with_props = (props is not None)

if props == ['all']:
    props = list(PROP_DICT.keys())

if with_props is not None:
    props_as_inp = bool(int(config['DATA'].get('props_as_inp', 0)))

multi_inp = (len(fps) > 1) or props_as_inp

train_val_split = float(config['DATA']['train_val_split'])
batch_size = int(config['DATA']['batch_size'])

model_definition = {
    'encoder': {'units_ind': None, 'units_common': None, 'dropout': None, 'activation': None, 'latent_activation': None},
    'decoder': {'units_ind': None, 'units_common': None, 'dropout': None, 'activation': None, 'prop_activation': False},
    'properties_predictor': {'units_common': None, 'dropout': None, 'activation': None, 'batch_norm': None}
}

model_conf_and_type = [('MODEL_ENCODER', 'encoder'), ('MODEL_DECODER', 'decoder')] 
if with_props and not props_as_inp:
    model_conf_and_type.append(('MODEL_PROPERTIES', 'properties_predictor'))

VAE = False

for model_conf, model_type in model_conf_and_type:
    if model_conf in ['MODEL_ENCODER', 'MODEL_DECODER'] and 'units_ind' in config[model_conf]:
        model_definition[model_type]['units_ind'] = literal_eval(config[model_conf]['units_ind'])
    if model_conf == 'MODEL_ENCODER':
        latent_space_size = int(config[model_conf]['latent_space'])
        model_definition[model_type]['latent_activation'] = config[model_conf].get('latent_activation')
        VAE = bool(int(config[model_conf].get('variational', 0)))
    elif model_conf == 'MODEL_PROPERTIES':
        model_definition[model_type]['batch_norm'] = literal_eval(config[model_conf]['batch_norm'])        
    model_definition[model_type]['units_common'] = literal_eval(config[model_conf]['units_common'])
    model_definition[model_type]['dropout'] = float(config[model_conf]['dropout'])
    model_definition[model_type]['activation'] = config[model_conf]['activation']
    model_definition[model_type]['rnd_seed'] = rnd_seed
    

if props_as_inp:
    model_definition['decoder']['prop_activation'] = True

fps_lens = [FPS_2_LEN[fp] for fp in fps]

opt = config['TRAIN']['optimizer']
lr = float(config['TRAIN']['learning_rate'])
loss_fp = literal_eval(config['TRAIN']['loss_fp'])
# if just one loss provided for multiple fingerprints, just use it for all of them, but if too many losses raise an error
if len(loss_fp) == 1 and len(fps_lens) != 1:
    loss_fp = loss_fp * len(fps_lens)
if len(loss_fp) != len(fps_lens):
    raise ValueError('Too many losses for fingerprints!')

if 'focal_loss' in loss_fp:
    loss_fp = [lo+'_'+fp if lo == 'focal_loss' else lo for lo, fp in zip(loss_fp, fps)]

compile_extras = {}
if with_props:
    compile_extras['loss_pp'] = literal_eval(config['TRAIN']['loss_pp'])
    compile_extras['loss_weights'] = literal_eval(config['TRAIN']['loss_weights'])

if 'metrics' in config['TRAIN']:
    compile_extras['metrics'] = literal_eval(config['TRAIN']['metrics'])

if 'metrics_fp' in config['TRAIN']:
    compile_extras['metrics_fp'] = literal_eval(config['TRAIN']['metrics_fp'])

if with_props and 'metrics_pp' in config['TRAIN']:
    compile_extras['metrics_pp'] = literal_eval(config['TRAIN']['metrics_pp'])

epochs = int(config['TRAIN']['epochs'])

lr_on_plateau = {}
lr_on_plateau['patience'] = int(config['TRAIN']['lr_patience'])
lr_on_plateau['factor'] = float(config['TRAIN']['lr_factor'])
lr_on_plateau['lr_min'] = float(config['TRAIN']['lr_min'])

save_model_period = int(config['TRAIN'].get('save_model_period',0))
early_stop = bool(int(config['TRAIN'].get('early_stop', 0)))

# DATA PREPARATION
if verbose:
    t_start = datetime.datetime.now()
    print(f"RUNINFO: Starting data preparation at: {t_start}")

train_dict, test_dict, _ = split_train_test_and_calculate_FPs(data_file, train_val_split, fps=fps, props=props)

fps_generator_train = FingerprintsGenerator(data=train_dict, batch_size=batch_size, fps=fps, props=props, shuf=True, props_as_inp=props_as_inp)
fps_generator_test = FingerprintsGenerator(data=test_dict, batch_size=batch_size, fps=fps, props=props, shuf=False, props_as_inp=props_as_inp)
if verbose:
    t_now = datetime.datetime.now()
    t_delta = (t_now - t_start).total_seconds()
    t_delta = t_delta/60
    print(f'RUNINFO: Finished data preparation. It took: {t_delta:.2f} min.')

# MODEL PREPARATION
if verbose:
    t_start = datetime.datetime.now()
    print(f"RUNINFO: Starting model preparation at: {t_start}")


# CREATE MODEL AS ae
if VAE: 
    if multi_inp:
        if with_props and not props_as_inp:
            ae = Variational_Autoencoder_dense_multi(latent_space_size, fps_lens, with_properties=True, properties_shape=len(props), **model_definition)
        elif with_props and props_as_inp:
            ae = Variational_Autoencoder_dense_multi(latent_space_size, fps_lens+[len(props)], with_properties=False, **model_definition)
        else:
            ae = Variational_Autoencoder_dense_multi(latent_space_size, fps_lens, with_properties=False, **model_definition)
    else:
        if with_props:
            ae = Variational_Autoencoder_dense(latent_space_size, fps_lens[0], with_properties=True, properties_shape=len(props), **model_definition)
        else:
            ae = Variational_Autoencoder_dense(latent_space_size, fps_lens[0], with_properties=False, **model_definition)
else:
    if multi_inp:
        if with_props and not props_as_inp:
            ae = Autoencoder_dense_multi(latent_space_size, fps_lens, with_properties=True, properties_shape=len(props), **model_definition)
        elif with_props and props_as_inp:
            ae = Autoencoder_dense_multi(latent_space_size, fps_lens+[len(props)], with_properties=False, **model_definition)
        else:
            ae = Autoencoder_dense_multi(latent_space_size, fps_lens, with_properties=False, **model_definition)
    else:
        if with_props:
            ae = Autoencoder_dense(latent_space_size, fps_lens[0], with_properties=True, properties_shape=len(props), **model_definition)
        else:
            ae = Autoencoder_dense(latent_space_size, fps_lens[0], with_properties=False, **model_definition)


compile_model(ae, opt, lr, loss_fp, with_props, **compile_extras)

if verbose:
    t_now = datetime.datetime.now()
    t_delta = (t_now - t_start).total_seconds()
    t_delta = t_delta/60
    print(f'RUNINFO: Finished model preparation. It took: {t_delta:.2f} min.')


if verbose:
    t_start = datetime.datetime.now()
    print(f"RUNINFO: Starting model trainig at: {t_start}")

train_model(ae, fps_generator_train, fps_generator_test, epochs, lr_on_plateau, exp_dir, save_model_period, early_stop)

if verbose:
    t_now = datetime.datetime.now()
    t_delta = (t_now - t_start).total_seconds()
    t_delta = t_delta / 3600
    print(f'RUNINFO: Finished model training. It took: {t_delta:2f} h.')
    tot_time = (t_now - t_zero).total_seconds()
    tot_time = tot_time / 3600
    print(f'RUNINFO: Finished run. It took: {tot_time:.2f} h.')
