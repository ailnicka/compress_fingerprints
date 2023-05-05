import os
import multiprocessing
from numpy.lib.npyio import save
from tensorflow.keras import optimizers, metrics, losses
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import pandas as pd
import matplotlib.pyplot as plt
from new_metrics import PhiCoefficient, ReconstructedCosineSimilarity, ReconstructedTanimotoSimilarity, matthews_correlation_coefficient, tanimoto_similarity, cos_similarity, hamming_loss, multi_category_focal_loss

OPT_DICT = {'adam': optimizers.Adam, 'rmsprop': optimizers.RMSprop, 'sgd': optimizers.SGD}
METRICS_DICT = {'binary_accuracy': metrics.binary_accuracy, 'mape': metrics.mean_absolute_percentage_error, 'mse': metrics.mean_squared_error, 'mae': metrics.mean_absolute_error,
'reconstructed_cosine_similarity': cos_similarity, 'reconstructed_tanimoto_similarity': tanimoto_similarity, 'phi_coefficient': matthews_correlation_coefficient, 'hamming_loss': hamming_loss, 
'focal_loss_MACCS': lambda x,y: multi_category_focal_loss(x, y, alpha=0.31), 'focal_loss_Morgan_2': lambda x,y: multi_category_focal_loss(x, y, alpha=0.02), 
'focal_loss_RDK': lambda x,y: multi_category_focal_loss(x, y, alpha=0.43), 'focal_loss_avalon': lambda x,y: multi_category_focal_loss(x, y, alpha=0.4),}

def compile_model(model, optimizer, lr, loss_fp, with_prop, **kwargs):
    opt = OPT_DICT[optimizer](learning_rate=lr)
    if with_prop:
        loss_prop = kwargs.get('loss_pp', [])
    loss = loss_fp + loss_prop if with_prop else loss_fp
    loss = [METRICS_DICT[lo] if lo in METRICS_DICT.keys() else lo for lo in loss]
    loss_weights = kwargs.get('loss_weights')
    # Throw error if the weights don't comply with amount of losses
    if loss_weights is not None:
        assert len(loss) == len(loss_weights)
    metrics = kwargs.get('metrics')
    if metrics is not None:
        metrics = [METRICS_DICT[metric] for metric in metrics]
    metrics_fp = kwargs.get('metrics_fp')
    metrics_pp = kwargs.get('metrics_pp')
    if metrics_fp is not None:
        num_fp = len(loss_fp)
        metrics_fp = [METRICS_DICT[metric] for metric in metrics_fp]
        metrics = {f'output_{i+1}': metrics_fp for i in range(num_fp)}
        if with_prop and metrics_pp is not None:
            metrics_pp = [METRICS_DICT[metric] for metric in metrics_pp]
            metrics.update({f'output_{num_fp+1}': metrics_pp})

    model.compile(optimizer=opt, loss=loss, loss_weights=loss_weights, metrics=metrics)


def create_model_checkpoint(period, save_path):
    return ModelCheckpoint(filepath=save_path + '{epoch:02d}',
                           verbose=0,
                           save_weights_only=True,
                           save_best_only=False,
                           period=period)


def create_early_stop():
    return EarlyStopping(monitor='val_loss', patience=25)


def train_model(model, generator, val_generator, epochs, lr_on_plateau, save_path, period, early_stop):
    lr_pl = ReduceLROnPlateau(monitor='val_loss', **lr_on_plateau)
    cb = [lr_pl]
    if period !=0:
        cb.append(create_model_checkpoint(period, f'{save_path}/Models/'))
    if early_stop:
        cb.append(create_early_stop())

    history = model.fit(generator, epochs=epochs, validation_data=val_generator, callbacks=cb, verbose=2)

    model.save_weights(f'{save_path}/Models/final')
    with open(f'{save_path}/Models/model.json', 'w') as f:
        f.write(model.to_json())

    df_hist = pd.DataFrame(history.history)
    df_hist.to_csv(f'{save_path}/history.csv')
    plot_training(df_hist, save_path)


def plot_training(df_hist, savepath):
    fig, ax = plt.subplots()
    ax.plot(df_hist['loss'], label='tr_loss')
    ax.plot(df_hist['val_loss'], label='val_loss')
    ax.set_xlabel('Epoch') 
    ax.set_ylabel('Loss') 
    ax.legend()
    fig.savefig(f'{savepath}/history.png')

