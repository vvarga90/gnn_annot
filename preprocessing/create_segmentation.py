
# Hierarchical SLIC v3
#   Two-stage split:
#       1) splitting hierarchically where intensities (based on change of optical flow) are high (in decreasing order of intensities)
#       2) splitting result segmentation with exact edges (canny on optflow) and removing too small segments
#   using slic-zero algorithm to split segments
#


import sys
sys.path.append('..')

import numpy as np
import cv2
import os
import h5py
from skimage.segmentation import slic, mark_boundaries
from skimage.measure import label, regionprops
import util.util as Util
import util.imutil as ImUtil
from scipy.ndimage.filters import gaussian_filter

N_BASE_SEGMENTS_PER_FR = 300
N_NEW_SEGMENTS_AT_SPLIT = 4     # SLIC gives a less stable result when splitting into 2 segments (sometimes fails to split)
N_K_TARGET_SEGMENTS_TOTAL = 50   # target segment count for the whole video in thousands (before canny edge splitting)
SLIC_COMPACTNESS = 0.1    # compactness in the beginning of the SLIC process (using SLIC-zero), logscale (e.g. try 0.01, 0.1, ..., 100)
SLIC_SIGMA = 0
SLIC_MIN_SIZE_FACTOR = 0.5   # default: 0.5
SLIC_MAX_SIZE_FACTOR = 2.5   # default: 3.
MIN_SEGMENT_SIZE = 40       # minimum number of pixels allowed in the output segmentation
CANNY_HIGH = 200
CANNY_LOW_HIGH_RATE = 2.    # high/low
SOFT_INTENSITY_FROM_CANNY = True
SOFT_INTENSITY_SMOOTHING_SIGMA = 9.
FILL_BORDER_DIR_START_TOWARDS_BOTTOMRIGHT = False
ENABLE_SPLIT_WITH_BORDER = True
N_SPLITS_MAX = 2   # number of times a segment can be split
MAX_REMAINING_SEG_SPLIT_RATIO = [0.4]    # must be N_SPLITS_MAX-1 long; the max ratio of remaining segs split at each level;
                                         #    at level#N_SPLITS_MAX, 1.0 is used
MIN_INTENSITY_VAL_TO_SPLIT = [0.5]      # the minimum relative (to max) intensity to split a segment in each level 
                                         #    at level#N_SPLITS_MAX all remaining segments are allowed to be split
CREATE_VIDEO = True
CREATE_SOFT_INTENSITY_VIDEO = False   # debug
CREATE_HARD_EDGES_VIDEO = False   # debug
CREATE_ARCHIVE = True

def get_video_list():
    '''
    Returns:
        list of str; video names
    '''
    assert os.path.isdir(OPTFLOW_FOLDER)
    vidlist = os.listdir(OPTFLOW_FOLDER)
    vidlist = [fname[9:-3] for fname in vidlist if (fname[:9] == 'flownet2_') and (fname[-3:] == '.h5')]
    return vidlist

class SLICH3:

    '''
    Member fields:
        params_dict: dict{param_name - str: param_value - ?}
    '''

    def __init__(self, params_dict):
        self.params_dict = params_dict

    def _fill_missing_optflows(self, flows, dir_fw, max_val_limit):
        '''
        If optical flow is near zero in frame (e.g. because of repeated img frames), fill with prev frame flow
        Parameters:
            flows: ndarray(n_frames-1, sy, sx, 2:[dy, dx]) of float32
            dir_fw: bool; if True, frame #idx-1 is repeated, otherwise #idx+1 is repeated
            max_val_limit: float; a frame is considered to be missing optical flow if max pixel value is lower than this
        Returns:
            flows_fixed: ndarray(n_frames-1, sy, sx, 2:[dy, dx]) of float32; 
                'flows' object is returned if no changes were made, otherwise returns a new array
        '''
        assert flows.shape[3:] == (2,)
        missing_of_frames = np.amax(np.fabs(flows), axis=(1,2,3)) < max_val_limit   # (n_fr-1,)
        if not np.any(missing_of_frames):
            return flows
        flows_fixed = flows.copy()
        it_range = range(1, flows.shape[0], 1) if dir_fw else range(flows.shape[0]-2, -1, -1)
        for fr_idx in it_range:
            prev_fr_idx = fr_idx-1 if dir_fw else fr_idx+1
            if np.amax(np.fabs(flows[fr_idx])) < max_val_limit:  # rarely happens, not a major issue to recompute again
                flows_fixed[fr_idx,:,:,:] = flows[prev_fr_idx,:,:,:]
                print("Note: _fill_missing_optflows(): optical flow (fw: " + str(dir_fw) + ") in fr#" + \
                                                    str(fr_idx) + " was replaced with fr#"  + str(prev_fr_idx))
        return flows_fixed

    def _viz_optflows(self, flows):
        '''
        Converts 2-channel optical flow to 3-channel (bgr) images.
        Parameters:
            flows: ndarray(n_frames-1, sy, sx, 2:[dy, dx]) of float32
        Returns:
            flows_viz: ndarray(n_frames-1, sy, sx, 3:bgr) of uint8
        '''
        assert flows.shape[3:] == (2,)
        flows_viz = np.empty(flows.shape[:3] + (3,), dtype=np.uint8)
        for fr_idx in range(flows.shape[0]):
            flows_viz[fr_idx] = ImUtil.create_flow_visualization(flows[fr_idx], brightness_mul=40.)   # (sy, sx, 3) of ui8
        return flows_viz


    def _downscale_im_by_int_factor(self, im, ds_factor, ds_op):
        '''
        Downscales an image by an integer factor. Remainder cols/rows are dropped.
        Parameters:
            im: ndarray(sy, sx, ...) of ?
            ds_factor: int; downscale factor
            ds_op: Callable; the downscale operation - must have an "axis" parameter; compatible with np.mean(), np.any()
        Returns:
            im_ds: ndarray(dsy, dsx, ...) of ?; where dsy = sy // ds_factor, ...
        '''
        ds_size_yx = (im.shape[0]//ds_factor, im.shape[1]//ds_factor)
        im_size_yx_div = (ds_size_yx[0]*ds_factor, ds_size_yx[1]*ds_factor)
        im = im[:im_size_yx_div[0], :im_size_yx_div[1]]
        im = im.reshape((ds_size_yx[0], ds_factor, ds_size_yx[1], ds_factor) + im.shape[2:])
        im_ds = ds_op(im, axis=(1,3)).astype(im.dtype, copy=False)
        return im_ds

    def _upscale_im_by_int_factor(self, im_ds, us_factor, target_size):
        '''
        Upscales an image by an integer factor by repeating items. Pads the upscaled image to "target_size" with zeros.
        Parameters:
            im_ds: ndarray(dsy, dsx, ...) of ?
            us_factor: int; upscale factor
            target_size: tuple(2); size of the returned image (sy, sx) dims only; padded cols/rows are set to zero.
        Returns:
            im: ndarray(sy, sx, ...) of ?;
        '''
        assert len(target_size) == 2
        im = np.empty(target_size + im_ds.shape[2:], dtype=im_ds.dtype)
        us_unpadded_size_yx = (im_ds.shape[0]*us_factor, im_ds.shape[1]*us_factor)
        #pad_widths = (target_size[0] - us_unpadded_size_yx[0], target_size[1] - us_unpadded_size_yx[1])
        im[us_unpadded_size_yx[0]:, :us_unpadded_size_yx[1]] = 0   # zeroing padded parts in three assignments
        im[:us_unpadded_size_yx[0], us_unpadded_size_yx[1]:] = 0
        im[us_unpadded_size_yx[0]:, us_unpadded_size_yx[1]:] = 0
        im_ds = np.broadcast_to(im_ds[:,None,:,None], (im_ds.shape[0], us_factor, im_ds.shape[1], us_factor) + im_ds.shape[2:])
        im[:us_unpadded_size_yx[0], :us_unpadded_size_yx[1]] = im_ds.reshape(us_unpadded_size_yx + im_ds.shape[4:])
        return im


    def _create_soft_intensity_img_from_flow_laplacian(self, flows_fw, flows_bw):
        '''
        Returns the smoothed Laplacian of optical flows. L2 is taken over optflow channels.
        Parameters:
            flows_fw, flows_bw: ndarray(n_frames-1, sy, sx, 2:[dy, dx]) of float32
        Returns:
            soft_intensity: ndarray(n_frames, sy, sx) of float32;
        '''
        assert flows_fw.shape[3:] == (2,)
        assert flows_fw.shape == flows_bw.shape
        soft_intensity = np.zeros((flows_fw.shape[0]+1,) + flows_fw.shape[1:3], dtype=np.float32)
        # laplacian
        for fr_idx in range(flows_fw.shape[0]):
            flow_fw_lapl = cv2.Laplacian(flows_fw[fr_idx], ddepth=cv2.CV_32F, ksize=1)
            flow_bw_lapl = cv2.Laplacian(flows_bw[fr_idx], ddepth=cv2.CV_32F, ksize=1)
            soft_intensity[fr_idx] += np.linalg.norm(flow_fw_lapl, ord=2, axis=-1)
            soft_intensity[fr_idx+1] += np.linalg.norm(flow_bw_lapl, ord=2, axis=-1)

        # smooth
        DOWNSCALE_FACTOR = 8
        for fr_idx in range(soft_intensity.shape[0]):
            soft_int_ds = self._downscale_im_by_int_factor(soft_intensity[fr_idx], ds_factor=DOWNSCALE_FACTOR, ds_op=np.amax)
            soft_int_ds = gaussian_filter(soft_int_ds, sigma=self.params_dict['SOFT_INTENSITY_SMOOTHING_SIGMA'])
            soft_intensity[fr_idx] = self._upscale_im_by_int_factor(soft_int_ds, us_factor=DOWNSCALE_FACTOR, \
                                                            target_size=soft_intensity.shape[1:3])
        return soft_intensity

    def _create_soft_intensity_img_from_canny(self, hard_edges):
        '''
        Returns the smoothed version of the given canny derived hard edge image.
        Parameters:
            hard_edges: ndarray(n_frames, sy, sx) of uint8; values in {0,1}
        Returns:
            soft_intensity: ndarray(n_frames, sy, sx) of float32;
        '''
        assert hard_edges.ndim == 3
        soft_intensity = hard_edges.astype(np.float32)*255.
        
        # smooth
        DOWNSCALE_FACTOR = 8
        for fr_idx in range(soft_intensity.shape[0]):
            soft_int_ds = self._downscale_im_by_int_factor(soft_intensity[fr_idx], ds_factor=DOWNSCALE_FACTOR, ds_op=np.amax)
            soft_int_ds = gaussian_filter(soft_int_ds, sigma=self.params_dict['SOFT_INTENSITY_SMOOTHING_SIGMA'])
            soft_intensity[fr_idx] = self._upscale_im_by_int_factor(soft_int_ds, us_factor=DOWNSCALE_FACTOR, \
                                                            target_size=soft_intensity.shape[1:3])
        return soft_intensity

    def _create_optflow_canny_imgs(self, flow_fw_viz, flow_bw_viz, canny_low=100, canny_high=200):
        '''
        Computes edges in optical flow visualization images by canny. Computes forward and backward edges independently.
        Parameters:
            flow_fw_viz, flow_bw_viz: ndarray(n_frames-1, sy, sx, 3:bgr) of uint8
            flow_fw, flow_bw: ndarray(n_frames-1, sy, sx, 2:[dy, dx]) of float32
        Returns:
            canny_ims: ndarray(n_frames, sy, sx, 2:[fw, bw]) of uint8; values in {0,1}
        '''
        assert flow_fw_viz.shape == flow_bw_viz.shape
        assert flow_fw_viz.shape[3:] == (3,)
        canny_ims = np.zeros((flow_fw_viz.shape[0]+1,) + flow_fw_viz.shape[1:3] + (2,), dtype=np.uint8)
        for fr_idx in range(flow_fw_viz.shape[0]+1):

            if fr_idx < flow_fw_viz.shape[0]:
                edges_fw = cv2.Canny(flow_fw_viz[fr_idx], canny_low, canny_high)
            else:
                edges_fw = np.zeros(flow_fw_viz.shape[1:3], dtype=np.uint8)

            if fr_idx > 0:
                edges_bw = cv2.Canny(flow_bw_viz[fr_idx-1], canny_low, canny_high)
            else:
                edges_bw = np.zeros(flow_bw_viz.shape[1:3], dtype=np.uint8)

            canny_ims[fr_idx,:,:,0] = edges_fw
            canny_ims[fr_idx,:,:,1] = edges_bw
        return (canny_ims > 0).astype(np.uint8)   # convert from {0,255} to {0,1}


    def _fill_borders_with_adj_pixels(self, seg, border_val, n_iter_limit=20):
        '''
        Replaces pixels of 'border_val' values with approximately closest different values in image.
        Parameters:
            (MODIFIED) seg: ndarray(sy, sx) of int
            border_val: int; value to replace
            n_iter_limit: int
        '''
        assert seg.ndim == 2
        imsize_yx = np.array(seg.shape, dtype=np.int32)
        border_idxs = np.argwhere(seg == border_val)  # (n_border_pix, 2)
        for iter_idx in range(n_iter_limit):
            if self.params_dict['FILL_BORDER_DIR_START_TOWARDS_BOTTOMRIGHT'] is True:
                delta_dirs = [[0, -1], [-1, 0], [0,1], [1,0]]
            else:
                delta_dirs = [[0,1], [1,0], [0, -1], [-1, 0]]

            for delta_yx in delta_dirs:
                if border_idxs.shape[0] == 0:
                    break
                read_idxs = border_idxs + delta_yx
                readmask = np.all(read_idxs >= 0, axis=1) & np.all(read_idxs < imsize_yx, axis=1)
                read_idxs = read_idxs[readmask,:]
                write_idxs = border_idxs[readmask,:]
                seg[write_idxs[:,0], write_idxs[:,1]] = seg[read_idxs[:,0], read_idxs[:,1]]
                remain_mask = seg[border_idxs[:,0], border_idxs[:,1]] == border_val
                border_idxs = border_idxs[remain_mask,:]
            if border_idxs.shape[0] == 0:
                break
            assert iter_idx != n_iter_limit-1, "Failed to remove all borders in given number of iterations."
        assert not np.any (seg == border_val)
        #

    def _save_seg_archive(self, seg_fpath, seg):
        '''
        Saves segmentation to a .h5 archive.
        Parameters:
            seg_fpath: str
            seg: ndarray(n_frames, sy, sx) of i32; video segmentation with IDs starting from 0
        '''
        assert seg.ndim == 3
        h5f = h5py.File(seg_fpath, 'w')
        h5f.create_dataset('lvl0_seg', data=seg, dtype=np.int32, compression="gzip")
        h5f.close()
        
    def _create_seg_video(self, vid_fpath, ims, seg):
        '''
        Renders segmentation over images and writes it to a video file.
        Parameters:
            vid_fpath: str
            ims: ndarray(n_frames, sy, sx, 3:rgb) of ui8
            seg: ndarray(n_frames, sy, sx) of i32; video segmentation with IDs starting from 0
        '''
        assert ims.shape[3:] == (3,)
        assert seg.shape == ims.shape[:3]
        vr_fourcc = cv2.VideoWriter_fourcc(*'MJPG')     # use this codec with avi
        vr_fps = 25.  # vid_capture.get(cv2.CAP_PROP_FPS)
        vr_frSize_xy = (ims.shape[2], ims.shape[1])
        vid_writer = cv2.VideoWriter(vid_fpath, fourcc=vr_fourcc, fps=vr_fps, frameSize=vr_frSize_xy)
        assert vid_writer.isOpened(), "Unable to open video file for writing: " + vid_out_path

        for fr_idx in range(ims.shape[0]):
            im = ims[fr_idx,:,:,:].copy()
            ImUtil.render_segmentation_edges_BGR(im, seg[fr_idx,:,:], boundary_color_rgb=(255,255,0))
            vid_writer.write(im)

        vid_writer.release()

    def _create_soft_intensity_video(self, vid_fpath, soft_intensity):
        '''
        Renders soft intensity images and writes it to a video file.
        Parameters:
            vid_fpath: str
            soft_intensity: ndarray(n_frames, sy, sx) of float32;
        '''
        assert soft_intensity.ndim == 3
        soft_intensity = soft_intensity.copy()
        assert np.amin(soft_intensity) >= 0.
        soft_intensity = (soft_intensity / (np.amax(soft_intensity)/255.)).astype(np.uint8)
        vr_fourcc = cv2.VideoWriter_fourcc(*'MJPG')     # use this codec with avi
        vr_fps = 25.  # vid_capture.get(cv2.CAP_PROP_FPS)
        vr_frSize_xy = (soft_intensity.shape[2], soft_intensity.shape[1])
        vid_writer = cv2.VideoWriter(vid_fpath, fourcc=vr_fourcc, fps=vr_fps, frameSize=vr_frSize_xy)
        assert vid_writer.isOpened(), "Unable to open video file for writing: " + vid_out_path

        for fr_idx in range(soft_intensity.shape[0]):
            soft_intensity_im = np.broadcast_to(soft_intensity[fr_idx,:,:,None], soft_intensity.shape[1:] + (3,))
            vid_writer.write(soft_intensity_im)

        vid_writer.release()

    def _create_hard_edges_video(self, vid_fpath, hard_edges):
        '''
        Renders hard edge images and writes it to a video file.
        Parameters:
            vid_fpath: str
            hard_edges: ndarray(n_frames, sy, sx) of uint8; values in {0,1}
        '''
        assert hard_edges.ndim == 3
        assert np.amax(hard_edges) <= 1
        vr_fourcc = cv2.VideoWriter_fourcc(*'MJPG')     # use this codec with avi
        vr_fps = 25.  # vid_capture.get(cv2.CAP_PROP_FPS)
        vr_frSize_xy = (hard_edges.shape[2], hard_edges.shape[1])
        vid_writer = cv2.VideoWriter(vid_fpath, fourcc=vr_fourcc, fps=vr_fps, frameSize=vr_frSize_xy)
        assert vid_writer.isOpened(), "Unable to open video file for writing: " + vid_out_path

        for fr_idx in range(hard_edges.shape[0]):
            hard_edge_im = hard_edges[fr_idx,:,:,None]*np.array([255, 255, 255], dtype=np.uint8)
            vid_writer.write(hard_edge_im)

        vid_writer.release()

    def _run_slich_on_frame(self, im_rgb_fl32, soft_intensity, hard_edges, n_target_segs):
        '''
        Returns a segmentation of the frame.
        Parameters:
            im_rgb_fl32: ndarray(sy, sx, 3:rgb) of fl32; in [0,1] range - MUST BE RGB !!! (bgr might not be good, see SLIC details)
            soft_intensity: ndarray(sy, sx) of fl32;
            hard_edges: ndarray(sy, sx) of bool;
            n_target_segs: int
        Returns:
            seg: ndarray(sy, sx) of i32; segmentation with IDs starting from 0
            end_offset: int; max(segs)+1
        '''
        assert im_rgb_fl32.dtype == np.float32
        assert np.amax(im_rgb_fl32) <= 1.
        assert im_rgb_fl32.ndim == 3
        assert im_rgb_fl32.shape == soft_intensity.shape + (3,) == hard_edges.shape + (3,)
        assert hard_edges.dtype == np.bool_
        assert self.params_dict['N_SPLITS_MAX'] == len(self.params_dict['MAX_REMAINING_SEG_SPLIT_RATIO'])+1 \
                                               == len(self.params_dict['MIN_INTENSITY_VAL_TO_SPLIT'])+1
        
        # PHASE#0: run base level segmentation

        seg = slic(im_rgb_fl32, n_segments=self.params_dict['N_BASE_SEGMENTS_PER_FR'], compactness=self.params_dict['SLIC_COMPACTNESS'], \
                    sigma=self.params_dict['SLIC_SIGMA'], multichannel=True, convert2lab=True, \
                    min_size_factor=self.params_dict['SLIC_MIN_SIZE_FACTOR'], max_size_factor=self.params_dict['SLIC_MAX_SIZE_FACTOR'], \
                    slic_zero=True, start_label=1)
        unwritten_mask = seg <= 0
        if np.any(unwritten_mask):
            c_missed_pixels = np.count_nonzero(unwritten_mask)
            print(">>> WARNING! BASE level SLIC missed", c_missed_pixels, "pixels.")
            self._fill_borders_with_adj_pixels(seg, border_val=0)
            assert np.all(seg > 0)

        # PHASE#1: split with 'soft_intensity'

        # get base level segment rprop data
        rprops = regionprops(seg)   # labels are starting from 1 (start_label was specified in slic)
        seg_data = {}   # {seg_id - int: tuple(bbox - tuple(4) of int, ndarray(bb_h, bb_w))}
        new_seg_ids = []
        for rprop in rprops:
            mask_in_bbox = rprop.image
            bbox_tlbr = rprop.bbox
            seg_data[rprop.label] = (bbox_tlbr, mask_in_bbox)
            new_seg_ids.append(rprop.label)
        n_total_segs = [len(seg_data)]

        for split_lvl in range(self.params_dict['N_SPLITS_MAX']):

            # compute intensities for new segments
            curr_max_id = max(new_seg_ids) if len(new_seg_ids) > 0 else curr_max_id
            seg_intensities = np.empty((len(new_seg_ids), 2), dtype=np.int32)   # (n_segs, 2:[ID, value])
            for new_seg_idx in range(len(new_seg_ids)):
                new_seg_id = new_seg_ids[new_seg_idx]
                bbox_tlbr, mask_in_bbox = seg_data[new_seg_id]
                pixs = soft_intensity[bbox_tlbr[0]:bbox_tlbr[2], bbox_tlbr[1]:bbox_tlbr[3]][mask_in_bbox]
                seg_intensities[new_seg_idx, 0] = new_seg_id
                seg_intensities[new_seg_idx, 1] = np.amax(pixs)

            # select segments to split, all conditions below must hold: 
            #   1) maximum 'n_target_segs' total numer of segments EXPECTED after splitting
            #   2) maximum MAX_REMAINING_SEG_SPLIT_RATIO[split_lvl] ratio of the remaining segments selected with highest intensities
            #         for highest level, this is 1.0
            min_intensity_val_to_split = self.params_dict['MIN_INTENSITY_VAL_TO_SPLIT'][split_lvl] \
                                                    if split_lvl < self.params_dict['N_SPLITS_MAX']-1 else 0.    # condition 3)
            max_remaining_seg_split_ratio = self.params_dict['MAX_REMAINING_SEG_SPLIT_RATIO'][split_lvl] \
                                                    if split_lvl < self.params_dict['N_SPLITS_MAX']-1 else 1.    # condition 2)
            n_max_segs_to_split = int(seg_intensities.shape[0]*max_remaining_seg_split_ratio)        # condition 2)
            n_max_remaining_splits = int((n_target_segs - len(seg_data))/self.params_dict['N_NEW_SEGMENTS_AT_SPLIT'])    # condition 1)
            n_max_segs_to_split = min(n_max_segs_to_split, n_max_remaining_splits)

            seg_intensities = seg_intensities[seg_intensities[:,1] >= min_intensity_val_to_split]
            if seg_intensities.shape[0] > n_max_segs_to_split:
                seg_sorter = np.argsort(seg_intensities[:,1])[-n_max_segs_to_split:]
                seg_intensities = seg_intensities[seg_sorter,:]

            # split selected segments
            new_seg_ids = []
            for seg_id in seg_intensities[:,0]:
                bbox_tlbr, mask_in_bbox = seg_data[seg_id]
                del seg_data[seg_id]

                # run SLIC on segment
                im_rgb_fl32_bbox = im_rgb_fl32[bbox_tlbr[0]:bbox_tlbr[2], bbox_tlbr[1]:bbox_tlbr[3], :]
                split_segs = slic(im_rgb_fl32_bbox, n_segments=self.params_dict['N_NEW_SEGMENTS_AT_SPLIT'], \
                                    compactness=self.params_dict['SLIC_COMPACTNESS'], sigma=self.params_dict['SLIC_SIGMA'], \
                                    multichannel=True, convert2lab=True, min_size_factor=self.params_dict['SLIC_MIN_SIZE_FACTOR'], \
                                    max_size_factor=self.params_dict['SLIC_MAX_SIZE_FACTOR'], slic_zero=True, start_label=1, \
                                    mask=mask_in_bbox)

                unwritten_mask = (split_segs <= 0) & mask_in_bbox
                if np.any(unwritten_mask):
                    if np.all(split_segs[mask_in_bbox] == 0):
                        # if slic did not find any segments then set area with mask as a single segment with ID == 1
                        #   happens more rarely if slic_zero is enabled
                        split_segs[mask_in_bbox] = 1
                        print(">>> WARNING! Split failed.")
                    else:
                        # not all, but some pixels were left as bg within the mask area; happens in a smaller scale with slic_zero
                        # assign label 1 to these pixels (they are merged with other segs in phase#2 if too small)
                        split_segs[unwritten_mask] = 1
                        c_missed_pixels = np.count_nonzero(unwritten_mask)
                        if c_missed_pixels >= 50:
                            print(">>> WARNING! SLIC missed", c_missed_pixels, "pixels.")
                    
                rprops = regionprops(split_segs)   # labels are starting from 1 (start_label was specified in slic)
                assert len(rprops) >= 1
                for rprop in rprops:
                    mask_in_bbox2 = rprop.image
                    bbox_tlbr2 = (rprop.bbox[0]+bbox_tlbr[0], rprop.bbox[1]+bbox_tlbr[1], rprop.bbox[2]+bbox_tlbr[0], rprop.bbox[3]+bbox_tlbr[1])
                    curr_max_id += 1
                    assert curr_max_id not in seg_data.keys()
                    seg_data[curr_max_id] = (bbox_tlbr2, mask_in_bbox2)
                    new_seg_ids.append(curr_max_id)

            n_total_segs.append(len(seg_data))

        # write all segments into final segmentation image
        seg = np.full(im_rgb_fl32.shape[:2], dtype=np.int32, fill_value=-1)
        for seg_label, (bbox_tlbr, mask_in_bbox) in seg_data.items():
            seg[bbox_tlbr[0]:bbox_tlbr[2], bbox_tlbr[1]:bbox_tlbr[3]][mask_in_bbox] = seg_label

        # PHASE#2: split with 'hard_edges'
        #   write border into seg image with 0 (background) value
        #   skimage.label, then rprops, find small segments, get segment adjacencies and their counts
        #   overwrite small seg with non-small most frequent adj seg id
        #       if no such seg id, overwrite with 0 value
        #   fill 0 values iteratively with nonzero adj pixels, (start with appropriate direction matching canny)
        seg += 1   # 0 is ignored as background in skimage.measure.label()
        assert np.all(seg > 0)   # ensure all pixels are written with valid segment values
        if self.params_dict['ENABLE_SPLIT_WITH_BORDER'] is True:
            seg[hard_edges] = 0   # border value: 0 (background)
            seg = label(seg, background=0, return_num=False, connectivity=1).astype(np.int32)   # diagonals must not count as adjacent
            small_segs = {}
            rprops = regionprops(seg)
            for rprop in rprops:
                if rprop.area < self.params_dict['MIN_SEGMENT_SIZE']:
                    small_segs[rprop.label] = (rprop.bbox, rprop.image)
            assert 0 not in small_segs.keys()
            u_edges, c_edges = ImUtil.get_adj_graph_edge_list_fast(seg, ignore_axes=[], return_counts=True)
            nonzero_edges_mask = np.all(u_edges != 0, axis=1)  # filter edges from/to background
            u_edges = u_edges[nonzero_edges_mask]
            c_edges = c_edges[nonzero_edges_mask]
            for small_seg, (bbox_tlbr, mask_in_bbox) in small_segs.items():
                edge_idxs_with_seg = np.where(np.any(u_edges == small_seg, axis=1))[0]
                if edge_idxs_with_seg.shape[0] == 0:   # if small seg only adj to background, overwrite seg with background value
                    seg[bbox_tlbr[0]:bbox_tlbr[2], bbox_tlbr[1]:bbox_tlbr[3]][mask_in_bbox] = 0
                else:
                    max_edge_idx_with_seg = np.argmax(c_edges[edge_idxs_with_seg])  # otherwise overwrite seg with most frequent non-bg adj value
                    max_edge_with_seg = u_edges[edge_idxs_with_seg[max_edge_idx_with_seg]]
                    seg_to_join_idx1 = 1 if max_edge_with_seg[0] == small_seg else 0
                    assert (max_edge_with_seg[1-seg_to_join_idx1] == small_seg) and (max_edge_with_seg[seg_to_join_idx1] != small_seg)
                    seg[bbox_tlbr[0]:bbox_tlbr[2], bbox_tlbr[1]:bbox_tlbr[3]][mask_in_bbox] = max_edge_with_seg[seg_to_join_idx1]
            self._fill_borders_with_adj_pixels(seg, border_val=0)  # replace border pixels with nearby pixel values

        u_seg, inv_seg = np.unique(seg, return_inverse=True)
        #print("    n_segs in split levels:", n_total_segs, "final:", u_seg.shape[0])
        return inv_seg.reshape(seg.shape), u_seg.shape[0]

    def _run_slich_on_video(self, ims_rgb_fl32, flows_fw, flows_bw, n_target_segs_per_fr):
        '''
        Creates SLICH segmentation on a video.
        Parameters:
            ims_rgb_fl32: ndarray(n_frames, sy, sx, 3:rgb) of fl32; in [0,1] range - MUST BE RGB !!! (bgr might not be good, see SLIC details)
            flows_fw, flows_bw: ndarray(n_frames-1, sy, sx, 2:[dy, dx]) of float32
            n_target_segs_per_fr: int
        Returns:
            seg: ndarray(n_frames, sy, sx) of i32; video segmentation with IDs starting from 0
            n_segs: int;
            soft_intensity: ndarray(n_frames, sy, sx) of float32; returned for debugging purposes
            hard_edges: ndarray(n_frames, sy, sx) of uint8; values in {0,1}; returned for debugging purposes
        '''
        assert ims_rgb_fl32.shape[3:] == (3,)
        assert flows_fw.shape[3:] == flows_bw.shape[3:] == (2,)
        assert ims_rgb_fl32.shape[0] == flows_fw.shape[0]+1 == flows_bw.shape[0]+1
        assert self.params_dict['CANNY_LOW_HIGH_RATE'] >= 1.
        seg_end_offset = 0
        seg = np.full(ims_rgb_fl32.shape[:3], dtype=np.int32, fill_value=-1)

        flows_fw = self._fill_missing_optflows(flows_fw, dir_fw=True, max_val_limit=0.1)
        flows_bw = self._fill_missing_optflows(flows_bw, dir_fw=False, max_val_limit=0.1)
        flow_fw_viz = self._viz_optflows(flows_fw)
        flow_bw_viz = self._viz_optflows(flows_bw)
        canny_low = self.params_dict['CANNY_HIGH']/float(self.params_dict['CANNY_LOW_HIGH_RATE'])
        hard_edges = self._create_optflow_canny_imgs(flow_fw_viz, flow_bw_viz, canny_low=canny_low, canny_high=self.params_dict['CANNY_HIGH'])
        hard_edges = np.amax(hard_edges, axis=3)   # OR op on uint8 (merging forward, backward canny results)
        if self.params_dict['SOFT_INTENSITY_FROM_CANNY'] is True:
            soft_intensity = self._create_soft_intensity_img_from_canny(hard_edges)
        else:
            soft_intensity = self._create_soft_intensity_img_from_flow_laplacian(flows_fw, flows_bw)

        for fr_idx in range(ims_rgb_fl32.shape[0]):
            seg_fr, seg_fr_end_offset = self._run_slich_on_frame(ims_rgb_fl32[fr_idx], soft_intensity[fr_idx], \
                                                                        hard_edges[fr_idx].astype(np.bool_), n_target_segs_per_fr)
            seg[fr_idx,:,:] = seg_fr + seg_fr_end_offset
            seg_end_offset += seg_fr_end_offset
        return seg, seg_end_offset, soft_intensity, hard_edges


    def segment_video(self, vidname, ims, of_fw, of_bw, seg_out_folder, save_to_archive=False, save_result_video=False, \
                                            save_soft_video=False, save_hard_video=False, first_few_frames_only=False):
        '''
        Creates segmentation over a video.
        Parameters:
            vidname: str
            ims: ndarray(n_frames, sy, sx, 3) of ui8
            of_fw, of_bw: ndarray(n_frames-1, sy, sx, 3) of fl32
            seg_out_folder: str; output path for the .h5 packages, e.g., '/home/my_home/databases/DAVIS/preprocessed_data/segmentation/'
            save_to_archive, save_result_video, save_soft_video, save_hard_video: bool
            first_few_frames_only: bool; for debug, TODO remove
        '''
        n_orig_frames = of_fw.shape[0]+1
        n_frames = n_orig_frames
        if first_few_frames_only is True:
            n_frames = 5
            of_fw = of_fw[:n_frames-1]
            of_bw = of_bw[:n_frames-1]
        ims_rgb_fl32 = ims[..., ::-1].astype(np.float32)/255.  # bgr ui8  in [0,255] -> rgb fl32 in [0,1]

        target_segs_per_fr = int(self.params_dict['N_K_TARGET_SEGMENTS_TOTAL']*1000 / n_orig_frames)
        print("    Working on video '" + vidname + "', " + str(target_segs_per_fr) + " target segments per frame (before canny splitting)")
        seg, n_segs, soft_intensity, hard_edges = self._run_slich_on_video(ims_rgb_fl32, of_fw, of_bw, target_segs_per_fr)

        if save_to_archive is True:
            seg_fpath = os.path.join(seg_out_folder, 'slich3_' + str(vidname) + '.h5')
            self._save_seg_archive(seg_fpath, seg)

        if save_result_video is True:
            vid_fpath = os.path.join(seg_out_folder, 'viz_slich3_' + str(vidname) + '.avi')
            self._create_seg_video(vid_fpath, ims, seg)

        if save_soft_video is True:
            vid_fpath = os.path.join(seg_out_folder, 'soft_slich3_' + str(vidname) + '.avi')
            self._create_soft_intensity_video(vid_fpath, soft_intensity)

        if save_hard_video is True:
            vid_fpath = os.path.join(seg_out_folder, 'hard_slich3_' + str(vidname) + '.avi')
            self._create_hard_edges_video(vid_fpath, hard_edges)
    #

def _run_for_single_video(vidname, ims, of_fw, of_bw, seg_out_folder):
    '''
    Runs the segmentation script for a single video. Method can be used in a parallel context.
    Parameters:
        vidname: str
        ims: ndarray(n_frames, sy, sx, 3) of ui8
        of_fw, of_bw: ndarray(n_frames-1, sy, sx, 3) of fl16
        seg_out_folder: str; output path for the .h5 packages, e.g., '/home/my_home/databases/DAVIS/preprocessed_data/segmentation/'
    '''

    VIDNAMES_REDUCED_COUNT = ['mallard-water', 'dancing']
    # in case of two training videos (dancing, mallard-water), parameters above produce too high segment count;
    #   using reduced segment count for these two videos

    params_dict = {}
    params_dict['N_BASE_SEGMENTS_PER_FR'] = N_BASE_SEGMENTS_PER_FR
    params_dict['N_NEW_SEGMENTS_AT_SPLIT'] = N_NEW_SEGMENTS_AT_SPLIT
    params_dict['N_K_TARGET_SEGMENTS_TOTAL'] = N_K_TARGET_SEGMENTS_TOTAL
    params_dict['SLIC_COMPACTNESS'] = SLIC_COMPACTNESS
    params_dict['SLIC_SIGMA'] = SLIC_SIGMA
    params_dict['SLIC_MIN_SIZE_FACTOR'] = SLIC_MIN_SIZE_FACTOR
    params_dict['SLIC_MAX_SIZE_FACTOR'] = SLIC_MAX_SIZE_FACTOR
    params_dict['MIN_SEGMENT_SIZE'] = MIN_SEGMENT_SIZE
    params_dict['CANNY_HIGH'] = CANNY_HIGH
    params_dict['CANNY_LOW_HIGH_RATE'] = CANNY_LOW_HIGH_RATE
    params_dict['SOFT_INTENSITY_FROM_CANNY'] = SOFT_INTENSITY_FROM_CANNY
    params_dict['SOFT_INTENSITY_SMOOTHING_SIGMA'] = SOFT_INTENSITY_SMOOTHING_SIGMA
    params_dict['FILL_BORDER_DIR_START_TOWARDS_BOTTOMRIGHT'] = FILL_BORDER_DIR_START_TOWARDS_BOTTOMRIGHT
    params_dict['ENABLE_SPLIT_WITH_BORDER'] = ENABLE_SPLIT_WITH_BORDER
    params_dict['N_SPLITS_MAX'] = N_SPLITS_MAX
    params_dict['MAX_REMAINING_SEG_SPLIT_RATIO'] = MAX_REMAINING_SEG_SPLIT_RATIO
    params_dict['MIN_INTENSITY_VAL_TO_SPLIT'] = MIN_INTENSITY_VAL_TO_SPLIT

    if vidname in VIDNAMES_REDUCED_COUNT:
        params_dict['N_BASE_SEGMENTS_PER_FR'] = 256
        params_dict['N_K_TARGET_SEGMENTS_TOTAL'] = 50
        params_dict['CANNY_HIGH'] = 500

    slich_obj = SLICH3(params_dict)
    slich_obj.segment_video(vidname, ims, of_fw.astype(np.float32, copy=False), of_bw.astype(np.float32, copy=False), seg_out_folder, \
                            save_to_archive=CREATE_ARCHIVE, save_result_video=CREATE_VIDEO, \
                            save_soft_video=CREATE_SOFT_INTENSITY_VIDEO, save_hard_video=CREATE_HARD_EDGES_VIDEO, \
                            first_few_frames_only=False)
    #

def run(ims_dict, of_fw_dict, of_bw_dict, vidnames, seg_out_folder):
    '''
    Parameters:
        ims_dict: dict{vidname - str: ndarray(n_frames, sy, sx, 3) of ui8}
        of_fw_dict, of_bw_dict: dict{vidname - str: ndarray(n_frames-1, sy, sx, 3) of fl16}
        vidnames: list of str; e.g., ['bear', 'blackswan']
        seg_out_folder: str; output path for the .h5 packages, e.g., '/home/my_home/databases/DAVIS/preprocessed_data/segmentation/'
    '''
    assert N_K_TARGET_SEGMENTS_TOTAL < 70
    os.makedirs(seg_out_folder, exist_ok=True)

    N_PARALLEL_JOBS = 12   # disable parallel execution by assigning None

    vidnames_to_process = []
    for vidname in vidnames:
        fpath_out = os.path.join(seg_out_folder, 'slich3_' + str(vidname) + '.h5')
        if os.path.isfile(fpath_out):
            print("    Archive for video '" + vidname + "' found, skipping...")
        else:
            vidnames_to_process.append(vidname)

    import time  # TODO REMOVE
    t0 = time.time()

    if N_PARALLEL_JOBS is None:
        for vidname in vidnames_to_process:
            _run_for_single_video(vidname)
    else:
        from joblib import Parallel, delayed
        Parallel(n_jobs=N_PARALLEL_JOBS)(delayed(_run_for_single_video)\
                    (vidname, ims_dict[vidname], of_fw_dict[vidname], of_bw_dict[vidname], seg_out_folder) \
                        for vidname in vidnames_to_process)

    t1 = time.time()
    print("Total time taken creating segmentation:", round(t1-t0, 2), "seconds. ")