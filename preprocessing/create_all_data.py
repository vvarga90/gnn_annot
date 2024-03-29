
# 
# GNN_annot IJCNN 2021 implementation
#   Load/create and save data to .pkl/.h5 archives for efficient loading.
#   @author Viktor Varga
#

import sys
sys.path.append('..')

from datasets import DAVIS17
import os
import cv2
import h5py
import numpy as np
import create_annots as CreateAnnots
import create_fmaps as CreateFMaps
import create_segmentation as CreateSeg
import create_optflow_gma as CreateOptFlowGMA

def load_imgs_for_video(base_folder_path, vidname):
    '''
    Parameters:
        base_folder_path: str
        vidname: str
    Returns:
        ims: ndarray(n_frames, sy, sx, 3:BGR) of uint8
    '''
    vid_folder = os.path.join(base_folder_path, vidname)
    n_frames = len(os.listdir(vid_folder))
    ims = []
    for fr_idx in range(n_frames):
        im_path = os.path.join(vid_folder, str(fr_idx).zfill(5) + '.jpg')
        im = cv2.imread(im_path, cv2.IMREAD_COLOR)
        if im.shape != (480,854,3):
            assert vidname in ['disc-jockey', 'bike-packing', 'shooting', 'cat-girl']   # safety check
            im = cv2.resize(im, (854,480), interpolation=cv2.INTER_NEAREST)
        ims.append(im)
    ims = np.stack(ims, axis=0)    # (n_frames, sy, sx, 3) of ui8
    assert ims.dtype == np.uint8
    return ims

def run(vidnames, imgs_folder, annot_folder, optflow_out_folder, fmap_out_folder, seg_out_folder, annot_out_folder):
    '''
    Parameters:
        vidnames: list of str; e.g., ['bear', 'blackswan']
        imgs_folder: str; base folder of DAVIS .jpg images, e.g., '/home/my_home/databases/DAVIS2017/DAVIS/JPEGImages/480p/'
        annot_folder: str; base folder of DAVIS .png format GT annotations, e.g., '/home/my_home/databases/DAVIS2017/DAVIS/Annotations/480p/'
        optflow_out_folder: str; output folder for the generated optical flow result archives
        fmap_out_folder: str; output folder for the generated feature map archives
        seg_out_folder: str; output folder for the generated superpixel segmentation result archives
        annot_out_folder: str; output folder for the generated GT annotation archives
    '''
    # load image data
    print("Preprocessing: loading image data...")
    ims_dict = {vidname: load_imgs_for_video(imgs_folder, vidname) for vidname in vidnames}

    # create optflow data
    #   TODO might not be worth to use compression in hdf5 here
    print("Preprocessing: running optical flow and occlusion estimation...")
    of_fw_dict, of_bw_dict, occl_fw_dict, occl_bw_dict = CreateOptFlowGMA.run(ims_dict, optflow_out_folder, 'optflow_gma_')

    # create feature map data
    print("Preprocessing: creating feature maps...")
    CreateFMaps.run(ims_dict, fmap_out_folder, 'features_MobileNetV2_')

    # create segmentation
    print("Preprocessing: creating segmentation...")
    CreateSeg.run(ims_dict, of_fw_dict, of_bw_dict, vidnames, seg_out_folder)   # may modify optflow data, do not use it later

    # create & save annotations into .h5 archives
    print("Preprocessing: collecting annotations into a single file per video...")
    CreateAnnots.run(annot_folder, vidnames, annot_out_folder, 'annots_', 'annots')

    print("Done preprocessing.")

if __name__ == '__main__':
    vidnames = DAVIS17.get_video_set_vidnames('full (45+15+30)', 'all')
    #vidnames = ['bear', 'blackswan']
    N_VIDS_TO_PROCESS_AT_ONCE = 24   # to reduce memory consumption (approx. 40g if this var is set to 24)
    vidnames_splits = [vidnames[i:i+N_VIDS_TO_PROCESS_AT_ONCE] for i in range(0, len(vidnames), N_VIDS_TO_PROCESS_AT_ONCE)]
    for vidnames_split in vidnames_splits:
        run(vidnames=vidnames_split,
            imgs_folder=DAVIS17.IM_FOLDER,
            annot_folder=DAVIS17.GT_ANNOT_FOLDER,
            optflow_out_folder=DAVIS17.DATA_FOLDER_OPTFLOWS,
            fmap_out_folder=DAVIS17.DATA_FOLDER_FMAPS,
            seg_out_folder=DAVIS17.DATA_FOLDER_BASE_SEGS,
            annot_out_folder=DAVIS17.DATA_FOLDER_GT_ANNOTS)
