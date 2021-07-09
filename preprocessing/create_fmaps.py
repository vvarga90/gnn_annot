
# 
# GNN_annot IJCNN 2021 implementation
#   Run MobileNet v2 on videos and save extracted feature maps.
#   @author Viktor Varga
#

import os
import numpy as np
import pickle

import torch    # tested with torch 

# MobileNet v2 - Keras layer names vs. PyTorch feature idxs
#   'expanded_conv_project_BN' <-> model.features[1]
#   'block_2_add' <-> model.features[3]
#   'block_5_add' <-> model.features[6]
#   'block_12_add' <-> model.features[13]
#   'block_16_project_BN' <-> model.features[17]
#   'out_relu' <-> model.features[18], not included in output archive
#

#LAYER_NAMES_MOBILENET_V2 = ['expanded_conv_project_BN', 'block_2_add', 'block_5_add', 'block_12_add',\
#                       'block_16_project_BN', 'out_relu']
LAYER_NAMES_MOBILENET_V2 = ['expanded_conv_project_BN', 'block_2_add', 'block_5_add', 'block_12_add',\
                       'block_16_project_BN']


BATCH_SIZE = 16
_outputs_dict_global = {}   # forward hook is extracting layer outputs into this global dict

def _gpu_setup(use_gpu, gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  

    if torch.cuda.is_available() and use_gpu:
        print('cuda available with GPU:',torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print('cuda not available')
        device = torch.device("cpu")
    return device

def create_hook(id):
    def hook(model, input, output):
        _outputs_dict_global[id] = output.detach().cpu().numpy()
    return hook

def extract_and_save_features(fpath_out, ims_arr, model, device):
    '''
    Parameters:
        fpath_out: str, e.g., '/home/my_home/databases/DAVIS/preprocessed_data/fmaps/fmaps_MobileNetV2_bear.pkl'
        ims_arr: ndarray(n_imgs, 480, 854, 3:BGR) of uint8
        model: PyTorch Model
        device: Torch.device
    '''
    yss = []
    for fr_offset in range(0, ims_arr.shape[0], BATCH_SIZE):
        fr_end_offset = min(fr_offset+BATCH_SIZE, ims_arr.shape[0])
        xs = ims_arr[fr_offset:fr_end_offset,:,:,:]  # (batch_size, size_y, size_x, 3)
        xs = xs[:,:,:,::-1].astype(np.float32) / 255. # BGR [0,255] ui8 -> RGB [0,1] fl32
        xs = (xs - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]  # normalize following MobileNet v2 guide in PyTorch docs
        xs = np.transpose(xs, (0, 3, 1, 2))  # channel-last -> channel first

        input_batch = torch.tensor(xs, dtype=torch.float32, device=device)  # (batch_size, 3, sy, sx)
        with torch.no_grad():
            model(input_batch)

        ys = (_outputs_dict_global['expanded_conv_project_BN'],\
              _outputs_dict_global['block_2_add'],\
              _outputs_dict_global['block_5_add'],\
              _outputs_dict_global['block_12_add'],\
              _outputs_dict_global['block_16_project_BN'])   # (batch_size, n_ch, dsy, dsx) each

        _outputs_dict_global.clear()
        yss.append(ys)

    # concatenate batches, convert to channel-last axis order
    feature_viddict = {ln:[yss_batch[l_idx] for yss_batch in yss] for l_idx, ln in enumerate(LAYER_NAMES_MOBILENET_V2)}
    feature_viddict = {ln: np.concatenate(yss, axis=0).transpose((0,2,3,1)).astype(np.float16) for ln, yss in feature_viddict.items()}

    # save features to a pickle archive
    with open(fpath_out, 'wb') as f:
        pickle.dump(feature_viddict, f)   # arrays in dict are (n_imgs, dsy, dsx, n_ch) each


def run(ims_dict, out_folder, out_fname_prefix):
    '''
    Parameters:
        ims_dict: dict{str - vidname: ndarray(n_imgs, 480, 854, 3:RGB) of uint8}
        out_folder: str; output path for single .pkl package, e.g., '/home/my_home/databases/DAVIS/preprocessed_data/fmaps/'
        out_fname_prefix: str; e.g., 'features_modelname_'
    '''
    device = _gpu_setup(use_gpu=True, gpu_id=0)
    model = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True)
    model.eval()
    model.to(device)

    # register forward hooks to the layer of interest to extract their output
    model.features[1].register_forward_hook(create_hook('expanded_conv_project_BN'))
    model.features[3].register_forward_hook(create_hook('block_2_add'))
    model.features[6].register_forward_hook(create_hook('block_5_add'))
    model.features[13].register_forward_hook(create_hook('block_12_add'))
    model.features[17].register_forward_hook(create_hook('block_16_project_BN'))

    os.makedirs(out_folder, exist_ok=True)
    for vidname, ims_arr in ims_dict.items():
        assert ims_arr.shape[1:] == (480, 854, 3)
        fpath_out = os.path.join(out_folder, out_fname_prefix + vidname + '.pkl')
        if os.path.isfile(fpath_out):
            print("    Archive for video '" + vidname + "' found, skipping...")
        else:
            print("    Processing video '" + vidname + "'...")
            extract_and_save_features(fpath_out, ims_arr, model, device)

