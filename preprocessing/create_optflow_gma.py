
# 
# GNN_annot IJCNN 2021 implementation
#   Run GMA optical flow estimation to videos and save results as .h5 packages.
#   GMA optical flow estimation:
#       Jiang et al.: Learning to Estimate Hidden Motions with Global Motion Aggregation
#       https://github.com/zacjiang/GMA
#   @author Viktor Varga
#

GMA_OPTICAL_FLOW_CORE_FOLDER_PATH = '/home/my_home/git/GMA/core/'   # e.g. /home/my_home/git/GMA/core/
GMA_MODEL_PATH = '/home/my_home/git/GMA/checkpoints/gma-sintel.pth'

import sys
sys.path.append(GMA_OPTICAL_FLOW_CORE_FOLDER_PATH)

from argparse import Namespace 
import os
import h5py
import cv2
import numpy as np
import torch

# from GMA
from network import RAFTGMA
from utils.utils import InputPadder

CREATE_VIDEO = True

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

def cart2polar(x, y):
    """
    Elementwise Cartesian2Polar for arrays. x and y should be of the same size.
    Parameters:
        x, y: ndarray(...); cartesian coordinate arrays
    Returns:
        r: ndarray(...); radius
        phi: ndarray(...); angle in radians, -pi..pi
    """
    r = np.sqrt(np.square(x) + np.square(y))
    phi = np.arctan2(y, x)
    return r, phi

def create_flow_visualization(flow, brightness_mul=10.):
    """
    Creates an optical flow visualization, where for each pixel, brightness represents the flow vector
        distance and the color its angle (projected to the hue in the HSV coding).
    Parameters:
        flow: ndarray(size_y, size_x, 2:[y,x]) of float32; output of optical flow algorithms
        brightness_mul: float;
    Returns:
        flowviz_bgr: ndarray(size_y, size_x, 3) of uint8
    """
    assert flow.ndim == 3
    assert flow.shape[2] == 2
    MAX_HUE = 179.
    flowviz_bgr = np.empty(flow.shape[:2] + (3,), dtype=np.uint32)
    flowviz_bgr.fill(255)
    r, phi = cart2polar(flow[:, :, 1], flow[:, :, 0])
    flowviz_bgr[:, :, 0] = ((phi + np.pi) / (2. * np.pi) * MAX_HUE).astype(np.uint32)
    flowviz_bgr[:, :, 2] = (r * brightness_mul).astype(np.uint32)
    flowviz_bgr[:, :, 1:] = np.clip(flowviz_bgr[:, :, 1:], 0, 255)
    flowviz_bgr[:, :, 0] = np.clip(flowviz_bgr[:, :, 0], 0, int(MAX_HUE))
    flowviz_bgr = flowviz_bgr.astype(np.uint8)
    flowviz_bgr = cv2.cvtColor(flowviz_bgr, cv2.COLOR_HSV2BGR)
    return flowviz_bgr

def visualize_bit_mask(mask, color=(255, 255, 255)):
    """
    Parameters:
        mask: ndarray(size_y, size_x) of bool_ OR any uint (if uint, 0 is False, any other value is True)
        color: tuple(3); color in bgr format, uint8 values
    Returns:
        ndarray(size_y, size_x, 3) of uint8
    """
    assert np.issubdtype(mask.dtype, np.unsignedinteger) or mask.dtype == np.bool_
    if mask.dtype != np.bool_:
        mask = np.clip(mask, 0, 1)
    return np.broadcast_to((mask.astype(np.uint8) * 255)[:,:,None], mask.shape + (3,))

def compute_occlusions(flow_fw, flow_bw):
    '''
    Parameters:
        flow_fw, flow_bw: ndarray(sy, sx, 2:[dy, dx]) of fl32
    Returns:
        occl_fw, occl_bw: ndarray(sy, sx) of bool_
    '''
    assert flow_fw.shape == flow_bw.shape
    assert flow_fw.shape[2:] == (2,)
    NOISE_REDUCTION_BLOB_RADIUS = 0.1
    imsize = np.asarray(flow_fw.shape[:2], dtype=np.int32)

    occl_fw = np.empty(flow_fw.shape[:2], dtype=np.bool_)
    occl_bw = occl_fw.copy()

    vecs_to_2d_dict = {}
    occl_masks_dict = {}
    for direction in ['fw', 'bw']:
        flow_fwd_im = flow_fw if direction == 'fw' else flow_bw

        vecs_to = np.zeros_like(flow_fwd_im)    # (box_y, box_x, 2)
        indices = np.indices(flow_fwd_im.shape[:2], dtype=np.float32)
        vecs_to[:,:,0] = indices[0]
        vecs_to[:,:,1] = indices[1]
        vecs_to += flow_fwd_im
        vecs_to_2d_dict[direction] = vecs_to  # should not modify 'vecs_to' array after this line
        vecs_to = vecs_to[None,:,:,:]
        vecs_to = np.tile(vecs_to, (5,1,1,1))
        vecs_to[1] += np.asarray([NOISE_REDUCTION_BLOB_RADIUS,NOISE_REDUCTION_BLOB_RADIUS], dtype=np.float32)
        vecs_to[2] += np.asarray([-NOISE_REDUCTION_BLOB_RADIUS,NOISE_REDUCTION_BLOB_RADIUS], dtype=np.float32)
        vecs_to[3] += np.asarray([NOISE_REDUCTION_BLOB_RADIUS,-NOISE_REDUCTION_BLOB_RADIUS], dtype=np.float32)
        vecs_to[4] += np.asarray([-NOISE_REDUCTION_BLOB_RADIUS,-NOISE_REDUCTION_BLOB_RADIUS], dtype=np.float32)
        vecs_to = vecs_to.reshape((-1, 2))    # (box_y*box_x(*5), 2)
        vecs_to = np.around(vecs_to).astype(np.int32)
        valid_inds = np.all(vecs_to >= 0, axis=1) & np.all(vecs_to < imsize, axis=1)    # (box_y*box_x,)
        vecs_to = vecs_to[valid_inds]
        
        # occlusion mask
        mask = np.ones(flow_fwd_im.shape[:2], dtype=np.bool_)
        mask[vecs_to[:,0], vecs_to[:,1]] = 0
        occl_masks_dict[direction] = mask

    # remove thick True border from mask: set those pixels zero in the mask from which inverse optflow vectors point out of frame
    for direction in ['fw', 'bw']:
        inv_direction = 'bw' if direction == 'fw' else 'fw'
        vecs_to_inv = vecs_to_2d_dict[inv_direction]
        invalid_mask = np.any(vecs_to_inv < 0, axis=-1) | np.any(vecs_to_inv >= imsize-1, axis=-1)
        mask = occl_masks_dict[direction]
        mask[invalid_mask] = 0

    occl_fw, occl_bw = occl_masks_dict['fw'], occl_masks_dict['bw']
    return occl_fw, occl_bw

def run_and_save_optflow_occl(fpath_out, vidname, ims_arr, model, device):
    '''
    Parameters:
        fpath_out: str, e.g., '/home/my_home/databases/DAVIS/preprocessed_data/optflow/optflow_gma_bear.h5'
        vidname: str
        ims_arr: ndarray(n_imgs, 480, 854, 3:BGR) of uint8
        model: PyTorch Model
        device: Torch.device
    Returns:
        flows, inv_flows: ndarray(n_imgs-1, 480, 854, 2:[dy, dx]) of fl16
        occls, inv_occls: ndarray(n_imgs-1, 480, 854) of bool_
    '''
    if CREATE_VIDEO:
        fpath_out_folder = os.path.split(fpath_out)[0]
        vid_out_path = os.path.join(fpath_out_folder, 'viz_gma_' + str(vidname) + '.avi')
        vr_fourcc = cv2.VideoWriter_fourcc(*'H264')     # use this codec with avi
        vr_fps = 25.  # vid_capture.get(cv2.CAP_PROP_FPS)
        vr_frSize_xy = (854*2, 480*2)
        vid_writer = cv2.VideoWriter(vid_out_path, fourcc=vr_fourcc, fps=vr_fps, frameSize=vr_frSize_xy)
        assert vid_writer.isOpened(), "Unable to open video file for writing: " + vid_out_path

    of_h5 = h5py.File(fpath_out, 'w')
    of_h5.create_dataset("offset_yx", data=np.array([0., 0.], dtype=np.float32))
    of_h5.create_dataset("orig_vidsize_yx", data=np.array([480, 854], dtype=np.float32))
    of_h5.create_dataset("flow_run_size_yx", data=np.array([480, 854], dtype=np.float32))
        
    flows, inv_flows, occls, inv_occls = [], [], [], []

    for fr_idx in range(ims_arr.shape[0]-1):

        im0 = np.copy(ims_arr[fr_idx,:,:,::-1])       # ::-1 for BGR -> RGB, copy because tensors do not support negative stride
        im1 = np.copy(ims_arr[fr_idx+1,:,:,::-1])     # ::-1 for BGR -> RGB, copy because tensors do not support negative stride
        im0_t = torch.from_numpy(im0).permute(2, 0, 1).float()[None,:,:,:].to(device)
        im1_t = torch.from_numpy(im1).permute(2, 0, 1).float()[None,:,:,:].to(device)

        padder = InputPadder(im0_t.shape)
        im0_t, im1_t = padder.pad(im0_t, im1_t)
        _, flow_t = model(im0_t, im1_t, iters=12, test_mode=True)   # returns: flow_low, flow_up
        flow_t = padder.unpad(flow_t)
        flow_t = flow_t[0,:,:,:].detach().permute(1, 2, 0)
        flow = flow_t.cpu().numpy()
        del flow_t          # free gpu memory

        _, inv_flow_t = model(im1_t, im0_t, iters=12, test_mode=True)   # returns: flow_low, flow_up
        inv_flow_t = padder.unpad(inv_flow_t)
        inv_flow = inv_flow_t[0,:,:,:].detach().permute(1, 2, 0).cpu().numpy()
        del inv_flow_t          # free gpu memory

        flow = flow[...,::-1]               # restore [dy, dx] order along last axis
        inv_flow = inv_flow[...,::-1]

        occl, inv_occl = compute_occlusions(flow, inv_flow)

        if CREATE_VIDEO:
            fig1 = create_flow_visualization(flow, brightness_mul=40.)
            fig2 = create_flow_visualization(inv_flow, brightness_mul=40.)
            fig3 = visualize_bit_mask(occl)
            fig4 = visualize_bit_mask(inv_occl)

            fig_top = np.concatenate([fig1, fig3], axis=1)
            fig_bottom = np.concatenate([fig2, fig4], axis=1)
            fig = np.concatenate([fig_top, fig_bottom], axis=0)
            
            vid_writer.write(fig)

        flows.append(flow.astype(np.float16))
        inv_flows.append(inv_flow.astype(np.float16))
        occls.append(occl)
        inv_occls.append(inv_occl)
        
    flows = np.stack(flows, axis=0)
    inv_flows = np.stack(inv_flows, axis=0)
    occls = np.stack(occls, axis=0)
    inv_occls = np.stack(inv_occls, axis=0)
    of_h5.create_dataset('flows', data=flows, compression="gzip")
    of_h5.create_dataset('inv_flows', data=inv_flows, compression="gzip")
    of_h5.create_dataset('occls', data=occls, compression="gzip")
    of_h5.create_dataset('inv_occls', data=inv_occls, compression="gzip")
    of_h5.close()

    if CREATE_VIDEO:
        vid_writer.release()

    return flows, inv_flows, occls, inv_occls

def load_optflow(base_folder_path, vidname, fname_prefix):
    '''
    Loads optical flow archives created previously.
    Parameters:
        base_folder_path: str
        vidname: str
        fname_prefix: str
    Returns:
        of_fw, of_bw: ndarray(n_imgs-1, 480, 854, 2:[dy, dx]) of fl16
        occl_fw, occl_bw: ndarray(n_imgs-1, 480, 854) of bool_
    '''
    flows_h5_path = os.path.join(base_folder_path, fname_prefix + vidname + '.h5')
    h5f = h5py.File(flows_h5_path, 'r')
    of_fw = h5f['flows'][:].astype(np.float16, copy=False)
    of_bw = h5f['inv_flows'][:].astype(np.float16, copy=False)
    occl_fw = h5f['occls'][:].astype(np.bool_, copy=False)
    occl_bw = h5f['inv_occls'][:].astype(np.bool_, copy=False)
    h5f.close()
    return of_fw, of_bw, occl_fw, occl_bw

def run(ims_dict, out_folder, out_fname_prefix):
    '''
    Parameters:
        ims_dict: dict{str - vidname: ndarray(n_imgs, 480, 854, 3:RGB) of uint8}
        out_folder: str; output path for single .h5 package, e.g., '/home/my_home/databases/DAVIS/preprocessed_data/optflow/'
        out_fname_prefix: str; e.g., 'optflow_gma_'
    Returns:
        of_fw_dict, of_bw_dict: dict{vidname - str: ndarray(n_imgs-1, 480, 854, 2:[dy, dx]) of fl16}
        occl_fw_dict, occl_bw_dict: dict{vidname - str: ndarray(n_imgs-1, 480, 854) of bool_}
    '''
    gma_args = Namespace(mixed_precision=False, model=GMA_MODEL_PATH, model_name='GMA', \
                         num_heads=1, position_and_content=False, position_only=False)

    device = _gpu_setup(use_gpu=True, gpu_id=0)
    model = torch.nn.DataParallel(RAFTGMA(gma_args))
    model.load_state_dict(torch.load(gma_args.model))
    #print(f"Loaded checkpoint at {gma_args.model}")
    model = model.module
    model.to(device)
    model.eval()

    of_fw_dict, of_bw_dict, occl_fw_dict, occl_bw_dict = {}, {}, {}, {}
    os.makedirs(out_folder, exist_ok=True)
    for vidname, ims_arr in ims_dict.items():
        assert ims_arr.shape[1:] == (480, 854, 3)
        fpath_out = os.path.join(out_folder, out_fname_prefix + vidname + '.h5')
        if os.path.isfile(fpath_out):
            print("    Archive for video '" + vidname + "' found, loading...")
            of_fw_dict[vidname], of_bw_dict[vidname], occl_fw_dict[vidname], occl_bw_dict[vidname] = \
                                            load_optflow(out_folder, vidname, out_fname_prefix)
        else:
            print("    Processing video '" + vidname + "'...")
            of_fw_dict[vidname], of_bw_dict[vidname], occl_fw_dict[vidname], occl_bw_dict[vidname] = \
                                            run_and_save_optflow_occl(fpath_out, vidname, ims_arr, model, device)

    return of_fw_dict, of_bw_dict, occl_fw_dict, occl_bw_dict