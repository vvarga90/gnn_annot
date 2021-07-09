
# 
# GNN_annot IJCNN 2021 implementation
#   Class to store the state of the superpixel segmentation of a video.
#   @author Viktor Varga
#

import time
import numpy as np
import cv2
import skimage.measure
import h5py  # TODO TEMP
import os  # TODO TEMP

import util.imutil as ImUtil

class Segmentation:

    '''
    Class representing the superpixel segmentation of a video. This implementation only supports constant graphs.
    Member fields:
        vidname: str
        seg_im_fwise: ndarray(n_frames, sy, sx) of int32; framewise seg IDs of the current segmentation
            (SERIALIZED AS UINT16)
        seg_id_fr_end_offsets: ndarray(n_frames,) of int32; seg ID end offests of the current segmentation
        seg_data: ndarray(n_segs_curr, 5:[size, bbox_tlbr]) of int32
    '''

    def __init__(self, vidname, seg_arr):
        '''
        Parameters:
            seg_arr: ndarray(n_frames, sy, sx) of i32
        '''
        self.vidname = vidname
        self._init_segmentation(seg_arr)
        print("Segmentation.__init__(): Sequence '" + str(self.vidname) + \
                                "', n_curr_segs: " + str(self.seg_data.shape[0]))
        #

    def __getstate__(self):
        # converting 'self.seg_im_fwise' to uint16 to reduce cache size
        assert np.all(self.seg_im_fwise >= 0)
        assert np.all(self.seg_im_fwise < 65534)
        d_seg_im_fwise = self.seg_im_fwise.astype(np.uint16)
        d = self.__dict__.copy()
        d['seg_im_fwise'] = d_seg_im_fwise
        return d

    def __setstate__(self, d):
        # restoring int32 type for 'self.seg_im_fwise'
        self.__dict__ = d
        self.seg_im_fwise = self.seg_im_fwise.astype(np.int32)

    # const public

    def get_shape(self):
        return self.seg_im_fwise.shape

    def get_n_frames(self):
        return self.get_shape()[0]

    def get_n_segs_total(self):
        return self.get_fr_seg_id_end_offset(self.get_n_frames()-1)

    def get_n_segs_in_frames(self):
        '''
        Returns:
            ndarray(n_frames,) of int
        '''
        return np.insert(np.diff(self.seg_id_fr_end_offsets), 0, [self.seg_id_fr_end_offsets[0]])

    def get_fr_seg_id_offset(self, fr_idx):
        return 0 if fr_idx == 0 else self.seg_id_fr_end_offsets[fr_idx-1]

    def get_fr_seg_id_end_offset(self, fr_idx):
        return self.seg_id_fr_end_offsets[fr_idx]

    def get_fr_seg_id_offsets(self, fr_idxs):
        seg_id_fr_offsets = np.concatenate([[0], self.seg_id_fr_end_offsets[:-1]])
        return seg_id_fr_offsets[fr_idxs].astype(np.int32)

    def get_seg_mask_for_fr_idxs(self, fr_idxs):
        '''
        Returns a mask over all segments. True where segment is located in frame in 'fr_idxs'.
        Parameters:
            fr_idxs: array-like(n_frames,) of int
        Returns:
            seg_mask: ndarray(n_segs,) of bool_
        '''
        seg_mask = np.zeros((self.get_n_segs_total(),), dtype=np.bool_)
        for fr_idx in fr_idxs:
            offset, end_offset = self.get_fr_seg_id_offset(fr_idx), self.get_fr_seg_id_end_offset(fr_idx)
            seg_mask[offset:end_offset] = True
        return seg_mask

    def get_framewise_seg_ids(self, seg_ids_global):
        fr_idxs = self.get_fr_idxs_from_seg_ids(seg_ids_global)
        return seg_ids_global - self.get_fr_seg_id_offsets(fr_idxs)

    def get_seg_sizes(self):
        '''
        Returns:
            seg_sizes: ndarray(n_segs,) of int32
        '''
        return self.seg_data[:,0]

    def get_seg_bboxes_tlbr(self):
        '''
        Returns:
            bboxes_tlbr: ndarray(n_segs, 4:tlbr) of int32
        '''
        return self.seg_data[:,1:]

    def get_seg_im(self, framewise_seg_ids=False):
        return self.get_seg_region(bbox_stlebr=None, framewise_seg_ids=framewise_seg_ids, keep_frame_dim=False)

    def get_seg_region(self, bbox_stlebr=None, framewise_seg_ids=False, keep_frame_dim=False):
        '''
        Returns pixelwise global or framewise segment IDs in the given 3D bbox.
        Paramters:
            bbox_stlebr: None OR tuple(6:stlebr) of ints (any can be None)
            framewise_seg_ids: bool; if True, seg IDs are returned as in self.seg_im_fwise, if False, the appropriate ID offset is added
            keep_frame_dim: bool; if False, does not keep framewise axis when returned bbox is 1 frame long in time
        Returns:
            ndarray(n_frames_in_bbox, sy, sx) of int32; the global seg IDs if framewise_seg_ids is False, otherwise framewise IDs
        '''
        seg = self.seg_im_fwise if bbox_stlebr is None else \
                        self.seg_im_fwise[bbox_stlebr[0]:bbox_stlebr[3], bbox_stlebr[1]:bbox_stlebr[4], bbox_stlebr[2]:bbox_stlebr[5]]
        if not framewise_seg_ids:
            fr_offset, fr_end_offset = (0, self.seg_im_fwise.shape[0]) if bbox_stlebr is None else (bbox_stlebr[0], bbox_stlebr[3])
            seg = seg + self.get_fr_seg_id_offsets(np.arange(fr_offset, fr_end_offset))[:,None,None]   # DO NOT use +=
        assert seg.ndim == 3
        if (not keep_frame_dim) and (seg.shape[0] == 1):
            seg = seg[0,:,:]
        return seg

    def get_segs_combined_bbox(self, seg_ids):
        '''
        Returns the 3D bbox enclosing the given segments (global IDs)
        Parameters:
            seg_ids: ndarray(n_segments,) of int32
        Returns:
            bbox_stlebr: tuple(6:[stlebr]) of ints; 3D bbox in order: startframe-top-left-endframe-bottom-right (ends exclusive)
        '''
        fr_idxs = self.get_fr_idxs_from_seg_ids(seg_ids)
        fr_start, fr_end = np.amin(fr_idxs), np.amax(fr_idxs)+1
        bboxes = self.get_seg_bboxes_tlbr()[seg_ids,:]
        bbox_tl = np.amin(bboxes[:,:2], axis=0)
        bbox_br = np.amax(bboxes[:,2:], axis=0)
        assert len(bbox_tl) == len(bbox_br) == 2
        return (fr_start, bbox_tl[0], bbox_tl[1], fr_end, bbox_br[0], bbox_br[1])

    def get_fr_idxs_from_seg_ids(self, seg_ids):
        '''
        Paramters:
            seg_ids: ndarray(n_segs) of int; (global IDs)
        Returns:
            fr_idxs: ndarray(n_segs) of int
        '''
        assert np.amax(seg_ids) < self.seg_id_fr_end_offsets[-1]
        return np.digitize(seg_ids, self.seg_id_fr_end_offsets)

    def get_seg_ids_from_coords(self, fr_idxs, points_yx, framewise_seg_ids=False):
        '''
        Paramters:
            fr_idxs: array-like(n_points,)
            points_yx: array-like(n_points, 2:[y, x])
            framewise_seg_ids: bool; if True, seg IDs are returned as in self.seg_im_fwise, if False, the appropriate ID offset is added
        Returns:
            seg_ids: ndarray(n_points,) of int
        '''
        fr_idxs = np.asarray(fr_idxs, dtype=np.int32)
        points_yx = np.asarray(points_yx, dtype=np.int32)
        assert points_yx.shape[1:] == (2,)
        assert fr_idxs.shape[0] == points_yx.shape[0]
        assert np.all(fr_idxs >= 0) and np.all(fr_idxs < self.seg_im_fwise.shape[0])
        assert np.all(points_yx >= 0) and np.all(points_yx < self.seg_im_fwise.shape[1:])
        seg_ids = self.seg_im_fwise[fr_idxs, points_yx[:,0], points_yx[:,1]]
        if not framewise_seg_ids:
            seg_ids += self.get_fr_seg_id_offsets(fr_idxs)
        assert seg_ids.shape == fr_idxs.shape
        return seg_ids


    # private

    def _init_segmentation(self, seg_arr):
        '''
        Init constant segmentation.
        Parameters:
            seg_arr: ndarray(n_frames, sy, sx) of i32; segment IDs starting from 0, NOT framewise
        Sets members:
            seg_im_fwise, seg_id_fr_end_offsets, seg_data
        '''
        assert seg_arr.ndim == 3
        n_frames = seg_arr.shape[0]
        self.seg_im_fwise = np.empty_like(seg_arr, dtype=np.int32)
        seg_id_fr_end_offsets = []
        new_seg_data = {}

        seg_id_offset = 0
        for fr_idx in range(n_frames):
            # store framewise segment IDs in self.seg_im_fwise instead of the global segment IDs
            u_seg_fr_seg, inv_seg_fr_seg, c_seg_fr_seg = np.unique(seg_arr[fr_idx], return_inverse=True, return_counts=True)
            inv_seg_fr_seg = inv_seg_fr_seg.reshape(seg_arr.shape[1:])
            self.seg_im_fwise[fr_idx,:,:] = inv_seg_fr_seg

            # get bbox from regionprops
            rprops = skimage.measure.regionprops(self.seg_im_fwise[fr_idx]+1, cache=True)   # IDs input to rprop must start from 1
            for rprop in rprops:
                # store segment size and bbox in self.seg_data
                new_seg_data[rprop.label+seg_id_offset-1] = (c_seg_fr_seg[rprop.label-1], rprop.bbox)

            seg_id_offset += u_seg_fr_seg.shape[0]
            seg_id_fr_end_offsets.append(seg_id_offset)

        self.seg_id_fr_end_offsets = np.array(seg_id_fr_end_offsets, dtype=np.int32)
        self.seg_data = np.empty((self.seg_id_fr_end_offsets[-1], 5), dtype=np.int32)
        for sp_idx in range(self.seg_data.shape[0]):
            new_seg_data_rec = new_seg_data[sp_idx]
            self.seg_data[sp_idx, :] = new_seg_data_rec[0], *new_seg_data_rec[1]
        #

