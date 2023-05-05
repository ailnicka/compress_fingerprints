from rdkit.Chem import MACCSkeys, AllChem, Descriptors
from rdkit.Chem.AtomPairs import Pairs
from rdkit.Avalon import pyAvalonTools 
import numpy as np
from random import shuffle
from keras.utils import Sequence

FPS_DICT = {'MACCS': lambda x: MACCSkeys.GenMACCSKeys(x).ToBitString(), 
            'Morgan_2': lambda x: AllChem.GetMorganFingerprintAsBitVect(x,radius=2).ToBitString(), 
            'Morgan_3': lambda x: AllChem.GetMorganFingerprintAsBitVect(x,3).ToBitString(), 
            'Morgan_4': lambda x: AllChem.GetMorganFingerprintAsBitVect(x,4).ToBitString(),
            'RDK': lambda x: AllChem.RDKFingerprint(x).ToBitString(),
            # 'atom_pair': lambda x: list(Pairs.GetHashedAtomPairFingerprint(x)),
            'avalon': lambda x: pyAvalonTools.GetAvalonFP(x).ToBitString()
            }


FPS_2_LEN = {'MACCS': 167, 'Morgan_2': 2048, 'Morgan_3': 2048, 'Morgan_4': 2048, 'RDK': 2048, 'avalon': 512}

# Added logs where it makes sense (ie big variations otherwise), alse later additionally standarisation
PROP_DICT = {'LogP': Descriptors.MolLogP,
            'Wt': lambda x: np.log(Descriptors.MolWt(x)),
            'fracCsp3': Descriptors.FractionCSP3,
            'Hacc': Descriptors.NumHAcceptors,
            'Hdon': Descriptors.NumHDonors,
            'rings': Descriptors.RingCount,
            'MR': lambda x: np.log(Descriptors.MolMR(x)),
            'TPSA': Descriptors.TPSA,
            'BJ': Descriptors.BalabanJ
            }


def split_train_test_and_calculate_FPs(filename, split, fps, props=None, shuf=False, save_splits=False):
    with open(filename) as f:
        smiles_list = f.read().splitlines()
    if shuf:
        shuffle(smiles_list)
    cutoff = int(split * len(smiles_list))
    # print(f'DEBUG: SMILES list done. Its size is {getsizeof(smiles_list)} {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}')
    fp_dict = {fp: [] for fp in fps} if props is None else {fp_or_prop:[] for fp_or_prop in fps+props}
    
    for sm in smiles_list:
        mol = AllChem.MolFromSmiles(sm)
        for fp in fps:
            fp_dict[fp].append(FPS_DICT[fp](mol))
        if props is not None:
            for prop in props:
                fp_dict[prop].append(PROP_DICT[prop](mol))

    fps_and_props = fps if props is None else fps+props

    if len(fps) == 1:
        if props is None:
            fp_dict[fp] = list(set(fp_dict[fp]))
        else:
            unique_fp = []
            unique_prop_dict = {}
            fingerprints_list = fp_dict[fp]
            for prop in props:
                unique_prop_dict[prop]=[]
            for i in range(len(fingerprints_list)):
                if fingerprints_list[i] not in unique_fp:
                    unique_fp.append(fingerprints_list[i])
                    for prop in props:
                       unique_prop_dict[prop].append(fp_dict[fp][i]) 
            fp_dict[fp] = unique_fp
            for prop in props:
                fp_dict[prop] = unique_prop_dict[prop]

    
    train_dict = {}
    test_dict = {}

    for fp_or_prop in fps_and_props:
        train_dict[fp_or_prop], test_dict[fp_or_prop] = fp_dict[fp_or_prop][:cutoff], fp_dict[fp_or_prop][cutoff:]
        fp_dict[fp_or_prop] = None

    # get z-scores for properties:
    z_score_reco = {}
    if props is not None:
        for prop in props:
            prop_list = train_dict[prop]
            prop_mean = np.mean(prop_list)
            prop_std = np.std(prop_list)
            z_score_reco[prop] = [prop_mean , prop_std]
            prop_list -= prop_mean
            prop_list /= prop_std
            train_dict[prop] = list(prop_list)
        for prop in props:
            prop_list = test_dict[prop]
            prop_mean , prop_std = z_score_reco[prop]
            prop_list -= prop_mean
            prop_list /= prop_std
            test_dict[prop] = list(prop_list)


    return train_dict, test_dict, z_score_reco


class FingerprintsGenerator(Sequence):

    def __init__(self, data, batch_size, fps, props=None, shuf=False, training=True, props_as_inp=False):
        # TODO: keep all list sep? or work on dict?
        self.data = data
        self.shuf = shuf
        self.batch_size = batch_size
        self.fps = fps
        self.props = props
        self.fps_and_props = fps if props is None else fps+props
        self.props_as_inp = props_as_inp
        self.len = len(data[fps[0]])
        for fp_or_prop in self.fps_and_props:
            assert len(data[fp_or_prop]) == self.len
        self.training = training
        self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuf:
            indexes = list(range(self.len))
            shuffle(indexes)
            for fp_or_prop in self.fps_and_props:
                self.data[fp_or_prop] = [self.data[fp_or_prop][idx] for idx in indexes]

    def __len__(self):
        return int(np.ceil(self.len / self.batch_size))

    def __getitem__(self, index):
        dict_batch = {}
        for fp_or_prop in self.fps_and_props:
                dict_batch[fp_or_prop] = self.data[fp_or_prop][index * self.batch_size : (index + 1) * self.batch_size]
        return self.__data_generation(dict_batch)

    def __data_generation(self, dict_batch):
        def bitstr_to_array(bitstr):
            return np.array([int(x) for x in bitstr])

        fps_batch = [[] for _ in range(len(self.fps))]
        prop_batch = []
        for i, fp in enumerate(self.fps):
            fps_batch[i] = [bitstr_to_array(x) for x in dict_batch[fp]]

        if self.props is not None:
            prop_batch= np.array([dict_batch[prop] for prop in self.props]).T
            
        fps_batch = [np.asarray(fps) for fps in fps_batch]

        if self.training:
            if self.props is not None:
                if self.props_as_inp:
                    return (fps_batch + [prop_batch], fps_batch + [prop_batch])
                else:
                    return (fps_batch, fps_batch + [prop_batch])
            else:
                if len(fps_batch) == 1: # single-fp model expects this flattened
                     return (fps_batch[0], fps_batch[0])
                else:
                    return (fps_batch, fps_batch)
        else:
            if self.props_as_inp:
                return fps_batch + [prop_batch]
            else:
                if len(fps_batch) == 1: # single-fp model expects this flattened
                    return fps_batch[0]
                else:
                    return fps_batch

