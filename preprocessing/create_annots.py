
# 
# GNN_annot IJCNN 2021 implementation
#   Load and save annotations in .pkl archive (one for each video) for more efficient data loading.
#   @author Viktor Varga
#

import os
from PIL import Image    # same as davisinteractive .png loading process (colors are decoded automatically), OpenCV does not do that
import h5py
import numpy as np

def load_annots_for_video(annot_folder, vidname):
    '''
    Parameters:
        annot_folder: str
        vidname: str
    Returns:
        annots: ndarray(n_frames, sy, sx) of uint8
    '''
    vid_folder = os.path.join(annot_folder, vidname)
    n_frames = len(os.listdir(vid_folder))
    annots = []
    for fr_idx in range(n_frames):
        im_fpath = os.path.join(vid_folder, str(fr_idx).zfill(5) + '.png')
        im = Image.open(im_fpath)   # (sy, sx) of ui8, colors are correctly coded as arange(n_labels) values
        if im.size != (854, 480):  # PIL Image.size attribute: x,y order
            assert vidname in ['disc-jockey', 'bike-packing', 'shooting', 'cat-girl']   # safety check
            im = im.resize((854, 480), Image.NEAREST)
        im = np.array(im)
        annots.append(im)
    annots = np.stack(annots, axis=0)    # (n_frames, sy, sx,) of ui8
    assert annots.dtype == np.uint8
    return annots

def save_annots(fpath_out, annots_arr, h5_key):
    '''
    Parameters:
        fpath_out: str
        annots_arr: ndarray(n_frames, sy, sx) of uint8
        h5_key: str
    '''
    assert annots_arr.ndim == 3
    h5f = h5py.File(fpath_out, 'w')
    h5f.create_dataset(h5_key, data=annots_arr, dtype=np.uint8, compression="gzip")
    h5f.close()

def run(annot_folder, vidnames, out_folder, out_fname_prefix, h5_key):
    '''
    Parameters:
        annot_folder: str; DAVIS 2017, Annotations 480p folder path, e.g., '/home/my_home/databases/DAVIS2017/DAVIS/Annotations/480p/'
        vidnames: list of str; e.g., ['bear', 'blackswan']
        out_folder: str; output folder of the .pkl packages, e.g., '/home/my_home/databases/DAVIS/preprocessed_data/'
        out_fname_prefix: str; e.g., 'annots_'
        h5_key: str; key for the dataset in the HDF-5 archive, e.g., 'annots'
    '''
    os.makedirs(out_folder, exist_ok=True)
    for vidname in vidnames:
        fpath_out = os.path.join(out_folder, out_fname_prefix + vidname + '.h5')
        if os.path.isfile(fpath_out):
            print("    Archive for video '" + vidname + "' found, skipping...")
        else:
            print("    Processing video '" + vidname + "'...")
            annots_arr = load_annots_for_video(annot_folder, vidname)
            save_annots(fpath_out, annots_arr, h5_key)
