from DataLoad import DataLoadDf, ConcatDataset, MultiStreamBatchSampler
from torch.utils.data import DataLoader
from torch import nn
from DatasetDcase2019Task4 import DatasetDcase2019Task4
from DataLoad import AugmentGaussianNoise, ApplyLog, PadOrTrunc, ToTensor, Normalize, Compose
from utils.utils import ManyHotEncoder, create_folder, SaveBest, to_cuda_if_available, weights_init, \
    AverageMeterSet

def get_transforms(frames, scaler=None, add_axis_conv=True, augment_type=None):
    transf = []
    unsqueeze_axis = None
    if add_axis_conv:
        unsqueeze_axis = 0

    # transf.extend([ApplyLog(), PadOrTrunc(nb_frames=frames)])
    transf.extend([PadOrTrunc(nb_frames=frames)])

    if scaler is not None:
        transf.append(Normalize(scaler=scaler))

    return Compose(transf)
