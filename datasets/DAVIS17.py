
# 
# GNN_annot IJCNN 2021 implementation
#   DAVIS2017 dataset paths
#   @author Viktor Varga
#

import os
import numpy as np
import h5py
import pickle
import cv2

# ---> For users: set the paths below

DATASET_ID = 'davis2017'
DAVIS_ROOT_FOLDER = '/home/my_home/databases/DAVIS/'   # e.g. '/home/my_home/databases/DAVIS/'
PREPROCESSED_DATA_ROOT_FOLDER = os.path.join(DAVIS_ROOT_FOLDER, 'gnn_annot_preprocessed_data/')
CACHE_FOLDER = os.path.join(DAVIS_ROOT_FOLDER, 'gnn_annot_cache/')

IM_FOLDER = os.path.join(DAVIS_ROOT_FOLDER, 'DAVIS2017/DAVIS/JPEGImages/480p/')
GT_ANNOT_FOLDER = os.path.join(DAVIS_ROOT_FOLDER, 'DAVIS2017/DAVIS/Annotations/480p/')
DATA_FOLDER_OPTFLOWS = os.path.join(PREPROCESSED_DATA_ROOT_FOLDER, 'optflow/')
DATA_FOLDER_FMAPS = os.path.join(PREPROCESSED_DATA_ROOT_FOLDER, 'fmaps/')
DATA_FOLDER_GT_ANNOTS = os.path.join(PREPROCESSED_DATA_ROOT_FOLDER, 'annots/')
DATA_FOLDER_BASE_SEGS = os.path.join(PREPROCESSED_DATA_ROOT_FOLDER, 'segmentation/')

IMFEATURES_MODELKEY = 'MobileNetV2'

VIDEO_SETS = {
    'full (45+15+30)': {'train': ['bear', 'bmx-bumps', 'boat', 'boxing-fisheye', 'breakdance-flare', 'bus', 'car-turn',\
                                   'cat-girl', 'classic-car', 'color-run', 'crossing', 'dance-jump', 'dancing', 'disc-jockey',\
                                   'dog-agility', 'dog-gooses', 'dogs-scale', 'drift-turn', 'drone', 'elephant', 'flamingo',\
                                   'hike', 'hockey', 'horsejump-low', 'kid-football', 'kite-walk', 'koala', 'lady-running',\
                                   'lindy-hop', 'longboard', 'lucia', 'mallard-fly', 'mallard-water', 'miami-surf',\
                                   'motocross-bumps', 'motorbike', 'night-race', 'paragliding', 'planes-water', 'rallye',\
                                   'rhino', 'rollerblade', 'schoolgirls', 'scooter-board', 'scooter-gray'],
                         'val': ['sheep',\
                                'skate-park', 'snowboard', 'soccerball', 'stroller', 'stunt', 'surf', 'swing', 'tennis',\
                                'tractor-sand', 'train', 'tuk-tuk', 'upside-down', 'varanus-cage', 'walking'],
                        'test': ['bike-packing', 'blackswan', 'bmx-trees', 'breakdance', 'camel', 'car-roundabout',\
                                 'car-shadow', 'cows', 'dance-twirl', 'dog', 'dogs-jump', 'drift-chicane', 'drift-straight',\
                                 'goat', 'gold-fish', 'horsejump-high', 'india', 'judo', 'kite-surf', 'lab-coat', 'libby',\
                                 'loading', 'mbike-trick', 'motocross-jump', 'paragliding-launch', 'parkour', 'pigs',\
                                 'scooter-black', 'shooting', 'soapbox']},
    'full_trainval (45+15+1)': {'train': ['bear', 'bmx-bumps', 'boat', 'boxing-fisheye', 'breakdance-flare', 'bus', 'car-turn',\
                                   'cat-girl', 'classic-car', 'color-run', 'crossing', 'dance-jump', 'dancing', 'disc-jockey',\
                                   'dog-agility', 'dog-gooses', 'dogs-scale', 'drift-turn', 'drone', 'elephant', 'flamingo',\
                                   'hike', 'hockey', 'horsejump-low', 'kid-football', 'kite-walk', 'koala', 'lady-running',\
                                   'lindy-hop', 'longboard', 'lucia', 'mallard-fly', 'mallard-water', 'miami-surf',\
                                   'motocross-bumps', 'motorbike', 'night-race', 'paragliding', 'planes-water', 'rallye',\
                                   'rhino', 'rollerblade', 'schoolgirls', 'scooter-board', 'scooter-gray'],
                         'val': ['sheep',\
                                'skate-park', 'snowboard', 'soccerball', 'stroller', 'stunt', 'surf', 'swing', 'tennis',\
                                'tractor-sand', 'train', 'tuk-tuk', 'upside-down', 'varanus-cage', 'walking'],
                        'test': ['bike-packing']},
    'reduced_test (45+15+15)': {'train': ['bear', 'bmx-bumps', 'boat', 'boxing-fisheye', 'breakdance-flare', 'bus', 'car-turn',\
                                           'cat-girl', 'classic-car', 'color-run', 'crossing', 'dance-jump', 'dancing', 'disc-jockey',\
                                           'dog-agility', 'dog-gooses', 'dogs-scale', 'drift-turn', 'drone', 'elephant', 'flamingo',\
                                           'hike', 'hockey', 'horsejump-low', 'kid-football', 'kite-walk', 'koala', 'lady-running',\
                                           'lindy-hop', 'longboard', 'lucia', 'mallard-fly', 'mallard-water', 'miami-surf',\
                                           'motocross-bumps', 'motorbike', 'night-race', 'paragliding', 'planes-water', 'rallye',\
                                           'rhino', 'rollerblade', 'schoolgirls', 'scooter-board', 'scooter-gray'],
                                 'val': ['sheep',\
                                        'skate-park', 'snowboard', 'soccerball', 'stroller', 'stunt', 'surf', 'swing', 'tennis',\
                                        'tractor-sand', 'train', 'tuk-tuk', 'upside-down', 'varanus-cage', 'walking'],
                                'test': ['bike-packing', 'blackswan', 'bmx-trees', 'breakdance', 'camel', 'car-roundabout',\
                                         'car-shadow', 'cows', 'dance-twirl', 'dog', 'dogs-jump', 'drift-chicane', 'drift-straight',\
                                         'goat', 'gold-fish']},
    'reduced (30+10+15)': {'train': ['bmx-bumps', 'boat', 'boxing-fisheye', 'breakdance-flare', 'bus', 'car-turn',\
                                   'cat-girl', 'classic-car', 'color-run', 'crossing', 'dance-jump', 'dancing', 'disc-jockey',\
                                   'dog-agility', 'dog-gooses', 'dogs-scale', 'drift-turn', 'drone', 'elephant', 'flamingo',\
                                   'hike', 'hockey', 'horsejump-low', 'kid-football', 'kite-walk', 'koala', 'lady-running',\
                                   'lindy-hop', 'longboard', 'lucia'],
                             'val': ['sheep',\
                                    'skate-park', 'snowboard', 'soccerball', 'stroller', 'stunt', 'surf', 'swing', 'tennis',\
                                    'tractor-sand'],
                            'test': ['bike-packing', 'blackswan', 'bmx-trees', 'breakdance', 'camel', 'car-roundabout',\
                                     'car-shadow', 'cows', 'dance-twirl', 'dog', 'dogs-jump', 'drift-chicane', 'drift-straight',\
                                     'goat', 'gold-fish']},
    'trainval_reduced (30+10+1)': {'train': ['bmx-bumps', 'boat', 'boxing-fisheye', 'breakdance-flare', 'bus', 'car-turn',\
                                            'cat-girl', 'classic-car', 'color-run', 'crossing', 'dance-jump', 'dancing', 'disc-jockey',\
                                            'dog-agility', 'dog-gooses', 'dogs-scale', 'drift-turn', 'drone', 'elephant', 'flamingo',\
                                            'hike', 'hockey', 'horsejump-low', 'kid-football', 'kite-walk', 'koala', 'lady-running',\
                                            'lindy-hop', 'longboard', 'lucia'],
                                     'val': ['sheep',\
                                             'skate-park', 'snowboard', 'soccerball', 'stroller', 'stunt', 'surf', 'swing', 'tennis',\
                                             'tractor-sand'],
                                    'test': ['bike-packing']},
    'test_only (1+1+30)': {'train': ['bear'],
                             'val': ['sheep'],
                            'test': ['bike-packing', 'blackswan', 'bmx-trees', 'breakdance', 'camel', 'car-roundabout',\
                                     'car-shadow', 'cows', 'dance-twirl', 'dog', 'dogs-jump', 'drift-chicane', 'drift-straight',\
                                     'goat', 'gold-fish', 'horsejump-high', 'india', 'judo', 'kite-surf', 'lab-coat', 'libby',\
                                     'loading', 'mbike-trick', 'motocross-jump', 'paragliding-launch', 'parkour', 'pigs',\
                                     'scooter-black', 'shooting', 'soapbox']},
    'test_only_reduced (1+1+15)': {'train': ['bear'],
                                     'val': ['sheep'],
                                    'test': ['bike-packing', 'blackswan', 'bmx-trees', 'breakdance', 'camel', 'car-roundabout',\
                                             'car-shadow', 'cows', 'dance-twirl', 'dog', 'dogs-jump', 'drift-chicane', 'drift-straight',\
                                             'goat', 'gold-fish']},
    'train_val_small (5+5+1)': {'train': ['bear', 'bmx-bumps', 'boat', 'boxing-fisheye', 'breakdance-flare'],
                             'val': ['sheep', 'skate-park', 'snowboard', 'soccerball', 'stroller'],
                            'test': ['bike-packing']},
    'debug_train (3+1+1)': {'train': ['bear', 'bmx-bumps', 'boat'],
                             'val': ['sheep'],
                            'test': ['bike-packing']},
    'debug_test (1+1+3)': {'train': ['bear'],
                             'val': ['sheep'],
                            'test': ['bike-packing', 'blackswan', 'bmx-trees']}

}

CUSTOM_IMSIZE_DICT = {'bike-packing': (480, 910), 'disc-jockey': (480, 1138), 'cat-girl': (480, 911), 'shooting': (480, 1152)}

#

def get_video_set_vidnames(video_set_id, split_id):
    '''
    Parameters:
        vidname: str
        video_set_id, split_id: str
    Returns:
        list of str; vidnames in set & split
    '''
    assert split_id in ['train', 'val', 'test', 'all']
    if split_id == 'all':
        video_set = VIDEO_SETS[video_set_id]
        return list(set(video_set['train'] + video_set['val'] + video_set['test']))
    else:
        return VIDEO_SETS[video_set_id][split_id]

def get_true_annots(vidname):
    '''
    Parameters:
        vidname: str
    Returns:
        gt_annot: ndarray(n_ims, sy, sx) of uint8
    '''
    davis_annot_h5_path = os.path.join(DATA_FOLDER_GT_ANNOTS, 'annots_' + vidname + '.h5')
    h5f = h5py.File(davis_annot_h5_path, 'r')
    gt_annot = h5f['annots'][:].astype(np.uint8)
    h5f.close()
    if vidname == 'tennis':   # erroeneous label in 'tennis' sequence annotations: the tennis ball is labeled 255 instead of 3
        error_mask = gt_annot == 255
        assert np.any(error_mask)
        gt_annot[error_mask] = 3
    return gt_annot

def get_segmentation_data(vidname):
    '''
    Parameters:
        vidname: str
    Returns:
        seg_arr: ndarray(n_frames, sy, sx) of i32
    '''
    seg_pkl_path = os.path.join(DATA_FOLDER_BASE_SEGS, 'slich3_' + str(vidname) + '.h5')
    h5f = h5py.File(seg_pkl_path, 'r')
    seg_arr = h5f['lvl0_seg'][:]
    h5f.close()
    return seg_arr

def get_optflow_occlusion_data(vidname):
    '''
    Parameters:
        vidname: str
    Returns:
        flow_fw, flow_bw: ndarray(n_ims, sy, sx, 2:[dy, dx]) of fl16
        occl_fw, occl_bw: ndarray(n_ims, sy, sx) of bool_
    '''
    flows_h5_path = os.path.join(DATA_FOLDER_OPTFLOWS, 'flownet2_' + vidname + '.h5')
    h5f = h5py.File(flows_h5_path, 'r')
    flow_fw = h5f['flows'][:].astype(np.float16)
    flow_bw = h5f['inv_flows'][:].astype(np.float16)
    occl_fw = h5f['occls'][:].astype(np.bool_)
    occl_bw = h5f['inv_occls'][:].astype(np.bool_)
    h5f.close()
    return flow_fw, flow_bw, occl_fw, occl_bw

def get_featuremap_data(vidname):
    '''
    Parameters:
        vidname: str
    Returns:
        fmaps: dict{str - ndarray of fl16}; see details in the funciton body
    '''
    features_pkl_path = os.path.join(DATA_FOLDER_FMAPS, \
               'features_' + IMFEATURES_MODELKEY + '_' + vidname + '.pkl')
    with open(features_pkl_path, 'rb') as f:
        # for each vid a dict: 
        # {'expanded_conv_project_BN': nd(82, 240, 427, 16) of fl16,
        #  'block_2_add': nd(82, 120, 213, 24) of fl16,
        #  'block_5_add': nd(82, 60, 106, 32) of fl16,
        #  'block_12_add': nd(82, 30, 53, 96) of fl16,
        #  'block_16_project_BN': nd(82, 15, 26, 320) of fl16,
        #  (NOT STORED, LEGACY) 'out_relu': nd(82, 15, 26, 1280) of fl16}

        fmaps = pickle.load(f)
        if 'out_relu' in fmaps.keys():
            del fmaps['out_relu']   # dropping 'out_relu'
    return fmaps

def get_img_data(vidname):
    '''
    Parameters:
        vidname: str
    Returns:
        ims_bgr: ndarray(n_frames, sy, sx, 3) of uint8
    '''
    vid_folder = os.path.join(IM_FOLDER, vidname)
    n_frames = len(os.listdir(vid_folder))
    ims_bgr = []
    for fr_idx in range(n_frames):
        im_path = os.path.join(vid_folder, str(fr_idx).zfill(5) + '.jpg')
        im = cv2.imread(im_path, cv2.IMREAD_COLOR)
        if im.shape != (480,854,3):
            assert vidname in ['disc-jockey', 'bike-packing', 'shooting', 'cat-girl']   # safety check
            im = cv2.resize(im, (854,480), interpolation=cv2.INTER_NEAREST)
        ims_bgr.append(im)
    ims_bgr = np.stack(ims_bgr, axis=0)    # (n_frames, sy, sx, 3) of ui8
    assert ims_bgr.dtype == np.uint8
    return ims_bgr


