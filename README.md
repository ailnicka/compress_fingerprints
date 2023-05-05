### Autoencoder compression of molecular fingerprints ###

This is code associated with the paper [Compression of Molecular Fingerprints with Autoencoder Networks](link_to_come)

To run code from this repository, first [create conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) using env_AE_full.yml or env_AE_gpu.yml if you have access to GPU. 

The repository structiure is as follows: code is in 'src' directory, the dataset is in 'data' directory, input needed to prepare and train autoencoder needs to be provided in 'inputs' directory (there is also script input_generator.py which helps to generate whole series of inputs). The resulting models with training informations are stored in 'Results' directory (see details of the results structure in text file there). The main file governing training of the autoencoder is run_experiment.py which parses input, builds autoencoder and trains it:
> python src/run_experiment.py --c <input_file> --verbose --randomseed <random_seed>

(where two last arguments are optional).

Once there are autoencoder models trained, to get QSAR predictions for compressed fingerprints one needs to run:
> python src/qsar_predictions.py <direcory_with_models_for_which_predictions_are_to_be_run>

To analyse dataset run:
> python src/dataset_analysis.py <file_with_dataset>

and to get entropy estimates:
> python src/compression_analysis.py <file_with_datset> <direcory_with_models_for_which_predictions_are_to_be_run> <number_of_bins_in_histogram_approximation>

# Address #

This code was created in: 

MODLAB <br/>
ETH Zurich <br/>
Inst. of Pharm. Sciences <br/>
HCI H 41 <br/>
Vladimir-Prelog-Weg 4 <br/>
CH-8093 Zurich