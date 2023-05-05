import numbers
import sys,os
sys.path.append(os.path.dirname(os.getcwd()))
import datetime

FPS_2_LEN = {'MACCS': 167, 'Morgan_2': 2048, 'Morgan_3': 2048, 'Morgan_4': 2048, 'RDK': 2048, 'avalon': 512}

name_extra = ''

def gen_layers_one_fp(fp:str, num_layers: int, len_latent:int):
    len_fp = FPS_2_LEN[fp]
    # if fp_len much larger than latent space then just divide fp, otherwise just reduce using diff between two
    if len_fp/(2**num_layers) > len_latent:
        encoder_layers = [int(len_fp/(2**i)) for i in range(1,num_layers+1)]
    else:
        len_diff_step = int((len_fp - len_latent)/(num_layers+1))
        encoder_layers = [len_fp - i*len_diff_step for i in range(1,num_layers+1)]
    return encoder_layers, encoder_layers[::-1]

def gen_layers_props(len_latent: int, num_layers:int):
    # assuming 9 props
    len_diff_step = int((len_latent - 9)/(num_layers+1))
    layers = [len_latent - i*len_diff_step for i in range(1,num_layers+1)]
    return layers

conf_str_single = """[DATA]
file = data/BiggerSetFromChEMBL.txt
# which fingerprints to use
fingerprints = {}
# properties used for co-training from the latent space
{}
train_val_split = 0.9
# here rather than in Train as it is used to construct generators
batch_size = 1024

[MODEL_ENCODER]
# last layer have always number of units corresponding to latent space
# common units after concat from in channels: for single input define just this
units_common = {}
dropout = 0.2
activation = relu
# size of the latent space
latent_space = {}
latent_activation = linear

[MODEL_DECODER]
# last layer have always number of units corresponding to input vectors and sigmoid activation
# common units from latent space: for single input define just this
units_common = {}
dropout = 0.2
activation = relu

[MODEL_PROPERTIES]
# last layer have always number of units corresponding to number of properties to predict and linear activation
# batch norm layer at the beginning and end?
batch_norm = [False, False]
units_common = {}
dropout = 0.2
activation = relu

[TRAIN]
epochs = 150
optimizer = adam
learning_rate = 0.001
# parameters for reducing of learning rate on plateau
lr_patience = 5
lr_factor = 0.5
lr_min = 0.000001
# loss for the training of autencoder
loss_fp = ['binary_crossentropy']
# loss for co-trained property prediction
loss_pp = ['mse']
# weights for fp and prop losses: one entry per every fp and one common for properties
loss_weights = [1,1]
metrics_fp = ['binary_accuracy', 'reconstructed_cosine_similarity', 'reconstructed_tanimoto_similarity', 'phi_coefficient', 'hamming_loss']
{}
# period when model is saved
save_model_period = 0
# if early stop when no imporvemnts in val loss for 15 epochs
early_stop = 1
"""

dirname = 'inputs_'+datetime.date.today().isoformat()
os.makedirs(dirname, exist_ok=True)

for n in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
    for layers in [2]:
        for fp in ['Morgan_2', 'RDK']:
            encoder, decoder = gen_layers_one_fp(fp, layers, n)
            with open(f'{dirname}/input_single_fp_{fp}_no_prop_{n}_layers_{layers}{name_extra}.ini', 'w') as f:
                f.write(conf_str_single.format(f"['{fp}']", '', str(encoder) ,str(n), str(decoder), '',''))
            with open(f'{dirname}/input_single_fp_{fp}_{n}_layers_{layers}{name_extra}.ini', 'w') as f:
                prop_pred = gen_layers_props(n, layers)
                f.write(conf_str_single.format(f"['{fp}']", "properties = ['all']", str(encoder) ,str(n), str(decoder), str(prop_pred), "metrics_pp = ['mape', 'mae']"))

for n in [2, 4, 8, 16, 32, 64]:
    for layers in [2]:
        for fp in ['MACCS']:
            encoder, decoder = gen_layers_one_fp(fp, layers, n)
            with open(f'{dirname}/input_single_fp_{fp}_no_prop_{n}_layers_{layers}{name_extra}.ini', 'w') as f:
                f.write(conf_str_single.format(f"['{fp}']", '', str(encoder) ,str(n), str(decoder), '', ''))
            with open(f'{dirname}/input_single_fp_{fp}_{n}_layers_{layers}{name_extra}.ini', 'w') as f:
                prop_pred = gen_layers_props(n, layers)
                f.write(conf_str_single.format(f"['{fp}']", "properties = ['all']", str(encoder) ,str(n), str(decoder), str(prop_pred), "metrics_pp = ['mape', 'mae']"))


for n in [2, 4, 8, 16, 32, 64, 128, 256]:
    for layers in [2]:
        for fp in ['avalon']:
            encoder, decoder = gen_layers_one_fp(fp, layers, n)
            with open(f'{dirname}/input_single_fp_{fp}_no_prop_{n}_layers_{layers}{name_extra}.ini', 'w') as f:
                f.write(conf_str_single.format(f"['{fp}']", '', str(encoder) ,str(n), str(decoder), '', ''))
            with open(f'{dirname}/input_single_fp_{fp}_{n}_layers_{layers}{name_extra}.ini', 'w') as f:
                prop_pred = gen_layers_props(n, layers)
                f.write(conf_str_single.format(f"['{fp}']", "properties = ['all']", str(encoder) ,str(n), str(decoder), str(prop_pred), "metrics_pp = ['mape', 'mae']"))

