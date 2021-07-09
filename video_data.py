
# 
# GNN_annot IJCNN 2021 implementation
#   Class to store all data related to a single video.
#   @author Viktor Varga
#

import time
import numpy as np
import cv2
import os

from segmentation import Segmentation
import util.featuregen as FeatureGen
import util.imutil as ImUtil

class VideoData:

    '''
    Member fields:
        data_source: str; either 'cache' or 'init' based on whether 'self' object was
                     created by loading from cache or through __init__ from preprocessed data
                     (NOT SERIALIZED) 

        vidname: str
        seg: Segmentation
        data: dict{name - str: data_item - ?}; DATA ITEMS ->
            'n_labels': int; the number of label categories (including background)
            'annot_im': ndarray(n_fr, sy, sx) of ui8; the true pixelwise labels
            'bgr_im': ndarray(n_fr, sy, sx, 3) of ui8; image data in BGR format
            'lab_im': ndarray(n_fr, sy, sx, 3) of fl32; image data in CIELAB format, range 0..1
            'of_fw_im': ndarray(n_fr, sy, sx, 2:[dy, dx]) of fl32; optflow data
            'of_bw_im': ndarray(n_fr, sy, sx, 2:[dy, dx]) of fl32; optflow data
            'occl_fw_im': ndarray(n_fr, sy, sx) of bool; optflow-derived occlusion data
            'occl_bw_im': ndarray(n_fr, sy, sx) of bool; optflow-derived occlusion data

            (UPDATED IF SEG is updated) ->
            'border_seg': ndarray(n_seg,) of bool; whether a specific segment touches the image borders
            'annot_seg': ndarray(n_seg,) of ui8; majority true label of segments
            'mean_lab_seg': ndarray(n_seg, 3) of bool; mean color of segments in CIELAB format, range 0..1
            'std_lab_seg': ndarray(n_seg, 3) of bool; color std of segments in CIELAB format, range 0..1
            'fmap_seg': ndarray(n_seg, n_features) of fl32; mean feature vector for segments
            'mean_of_fw_seg': ndarray(n_seg, 2) of fl32; mean fw optflow of segments
            'mean_of_bw_seg': ndarray(n_seg, 2) of fl32; mean bw optflow of segments
            'std_of_fw_seg': ndarray(n_seg, 2) of fl32; fw optflow std of segments
            'std_of_bw_seg': ndarray(n_seg, 2) of fl32; bw optflow std of segments
            'mean_occl_fw_seg': ndarray(n_seg,) of fl32; mean fw occl of segments
            'mean_occl_bw_seg': ndarray(n_seg,) of fl32; mean bw occl of segments

            'spatial_edges': ndarray(n_sp_edges, 2:[from_seg_id, to_seg_id])) of i32; undirected spatial edges, from_seg_id < to_seg_id
            'flow_edges': ndarray(n_fl_edges, 2:[from_seg_id, to_seg_id])) of i32; undirected optflow edges, from_seg_id < to_seg_id
            'all_edges_merger_idxs': ndarray(n_all_edges,) of i32; 'all_edges' can be produced from the concatenation of 
                                            ['spatial_edges', 'flow_edges'] arrays and indexing it with 'all_edges_merger_idxs'
            
            'flow_edge_fs': ndarray(n_all_edges, 2:[from_seg_id, to_seg_id], 4: TODO) of fl32; optflow based edge features
            'mean_lab_diff_edgefs': ndarray(n_all_edges,) of fl32; L2 color distance in CIELAB format
            'mean_of_fw_diff_edgefs': ndarray(n_all_edges, 3:[abs_angular_diff, min_mag, rel_mag]) of fl32; optflow fw difference based features
            'mean_of_bw_diff_edgefs': ndarray(n_all_edges, 3:[abs_angular_diff, min_mag, rel_mag]) of fl32; optflow bw difference based features

            (ONLY IF 'full' load_data protocol was used in CacheManager) ->
            'fmaps_dict': dict; full resolution feature maps, see format description in DAVIS17.get_featuremap_data()
            
    '''

    def __init__(self, vidname, imgs_bgr, gt_annot, flow_fw, flow_bw, occl_fw, occl_bw, fmaps_dict, seg_arr):
        self.data_source = 'init'
        self.vidname = vidname
        flow_fw = flow_fw.astype(np.float32)   # float16 can easily overflow when computing flow directions
        flow_bw = flow_bw.astype(np.float32)
        self.seg = Segmentation(vidname, seg_arr)
        self._init_data(imgs_bgr, gt_annot, flow_fw, flow_bw, occl_fw, occl_bw, fmaps_dict)

    # CUSTOM SERIALIZATION

    def __getstate__(self):
        assert self.data_source == 'init', "Object loaded from cache cannot be serialized again"
        d = self.__dict__.copy()
        del d['data_source']
        return d

    def __setstate__(self, d):
        self.__dict__ = d
        self.data_source = 'cache'
    #

    def get_seg(self):
        return self.seg

    def add_data(self, name, data_item):
        assert name not in self.data.keys()
        self.data[name] = data_item

    def get_data(self, name):
        return self.data[name]

    def drop_data(self, name):
        del self.data[name]

    def drop_multiple_data(self, names):
        for name in names:
            del self.data[name]


    # private

    def _init_data(self, imgs_bgr, gt_annot, flow_fw, flow_bw, occl_fw, occl_bw, fmaps_dict):
        '''
        Processes raw data to init VideoData object.
        Parameters:
            <see CacheManager._create_cache_file() for details>
        Modifies members:
            self.data
        '''
        self.data = {}

        # add images
        self.add_data('annot_im', gt_annot.astype(np.uint8, copy=False))
        self.add_data('bgr_im', imgs_bgr.astype(np.uint8, copy=False))
        lab_im_fl32 = np.stack([cv2.cvtColor(im, cv2.COLOR_BGR2LAB) for im in imgs_bgr], axis=0).astype(np.float32) / 255.
        #self.add_data('lab_im', lab_im_fl32)    # not storing LAB coding now, conversion with cv2 is fast (even faster than ui8 -> fl32 cast)

        flow_fw_padded = np.empty(imgs_bgr.shape[:1] + flow_fw.shape[1:], dtype=np.float32)
        flow_fw_padded[-1,:,:,:] = 0.
        flow_fw_padded[:-1,:,:,:] = flow_fw
        self.add_data('of_fw_im', flow_fw_padded.astype(np.float16))
        flow_bw_padded = np.empty(imgs_bgr.shape[:1] + flow_bw.shape[1:], dtype=np.float32)
        flow_bw_padded[0,:,:,:] = 0.
        flow_bw_padded[1:,:,:,:] = flow_bw
        self.add_data('of_bw_im', flow_bw_padded.astype(np.float16))
        occl_fw_padded = np.empty(imgs_bgr.shape[:1] + occl_fw.shape[1:], dtype=np.float32)
        occl_fw_padded[-1,:,:] = 0.
        occl_fw_padded[:-1,:,:] = occl_fw
        self.add_data('occl_fw_im', occl_fw_padded.astype(np.bool_))
        occl_bw_padded = np.empty(imgs_bgr.shape[:1] + occl_bw.shape[1:], dtype=np.float32)
        occl_bw_padded[0,:,:] = 0.
        occl_bw_padded[1:,:,:] = occl_bw
        self.add_data('occl_bw_im', occl_bw_padded.astype(np.bool_))
        self._update_seg_features(lab_im_fl32, flow_fw_padded, flow_bw_padded, occl_fw_padded, occl_bw_padded, fmaps_dict)
        #
        
    def _update_seg_features(self, lab_im_fl32=None, flow_fw_padded=None, flow_bw_padded=None, \
                            occl_fw_padded=None, occl_bw_padded=None, fmaps_dict=None):
        '''
        Recomputes all segmentation-dependent features.
        Expects all pixelwise features to be already added to self.
        Modifies members:
            self.data
        '''
        imgs_bgr = self.get_data('bgr_im')
        lab_im_fl32 = np.stack([cv2.cvtColor(im, cv2.COLOR_BGR2LAB) for im in imgs_bgr], axis=0).astype(np.float32) / 255. \
                                                                                        if lab_im_fl32 is None else lab_im_fl32
        flow_fw_padded = self.get_data('of_fw_im').astype(np.float32) if flow_fw_padded is None else flow_fw_padded
        flow_bw_padded = self.get_data('of_bw_im').astype(np.float32) if flow_bw_padded is None else flow_bw_padded
        occl_fw_padded = self.get_data('occl_fw_im').astype(np.float32) if occl_fw_padded is None else occl_fw_padded
        occl_bw_padded = self.get_data('occl_bw_im').astype(np.float32) if occl_bw_padded is None else occl_bw_padded
        fmaps_dict = self.get_data('fmaps_dict') if fmaps_dict is None else fmaps_dict

        # get segmentation image
        seg_im = self.seg.get_seg_im(framewise_seg_ids=False)
        seg_fr_idxs = self.seg.get_fr_idxs_from_seg_ids(np.arange(self.seg.get_n_segs_total()))
        seg_sizes = self.seg.get_seg_sizes()
        seg_bboxes_tlbr = self.seg.get_seg_bboxes_tlbr()
        seg_masks_in_bboxes = ImUtil.get_masks_in_bboxes(seg_im=seg_im, seg_fr_idxs=seg_fr_idxs, \
                                                         seg_bboxes_tlbr=seg_bboxes_tlbr)  # list(n_segs) of 2D ndarrays
        seg_idxs_yx, seg_idxs_offsets = ImUtil.get_mask_pix_idxs_in_bboxes(seg_im=seg_im, seg_fr_idxs=seg_fr_idxs, \
                                                seg_bboxes_tlbr=seg_bboxes_tlbr, seg_masks_in_bboxes=seg_masks_in_bboxes)

        # create segment features
        border_segs = np.zeros((self.seg.get_n_segs_total()), dtype=np.float32)
        border_segs_true = FeatureGen.get_border_segments(seg_im=seg_im)
        border_segs[border_segs_true] = 1.
        self.add_data('border_seg', border_segs)
        self.add_data('annot_seg', FeatureGen.apply_reduce_im2seg(img=self.get_data('annot_im')[..., None], seg_im=seg_im, seg_ids=None, \
                                seg_fr_idxs=seg_fr_idxs, seg_bboxes_tlbr=seg_bboxes_tlbr, seg_masks_in_bboxes=seg_masks_in_bboxes, \
                                seg_idxs_yx=seg_idxs_yx, seg_offsets=seg_idxs_offsets, reduction_func=FeatureGen.reduce_most_frequent_item)[:,0])
        u_labels_im = np.unique(self.get_data('annot_im'))
        u_labels_seg = np.unique(self.get_data('annot_seg'))
        assert np.array_equal(u_labels_im, u_labels_seg), "A label category is not present in the rounded to segments labeling"
        assert np.array_equal(u_labels_im, np.arange(u_labels_im.shape[0])), "Labels not from 0 to n_labels-1"
        assert u_labels_im.shape[0] >= 2
        self.add_data('n_labels', u_labels_im.shape[0])

        meanstd_lab_seg = FeatureGen.apply_reduce_im2seg(img=lab_im_fl32, seg_im=seg_im, seg_ids=None, \
                                seg_fr_idxs=seg_fr_idxs, seg_bboxes_tlbr=seg_bboxes_tlbr, seg_masks_in_bboxes=seg_masks_in_bboxes, \
                                seg_idxs_yx=seg_idxs_yx, seg_offsets=seg_idxs_offsets, reduction_func=FeatureGen.reduce_meanstd_nocast)
        assert meanstd_lab_seg.shape[1:] == (6,)
        self.add_data('mean_lab_seg', meanstd_lab_seg[:,:3])
        self.add_data('std_lab_seg', meanstd_lab_seg[:,3:])

        meanstd_of_fw_seg = FeatureGen.apply_reduce_im2seg(img=flow_fw_padded, seg_im=seg_im, seg_ids=None, \
                                seg_fr_idxs=seg_fr_idxs, seg_bboxes_tlbr=seg_bboxes_tlbr, seg_masks_in_bboxes=seg_masks_in_bboxes, \
                                seg_idxs_yx=seg_idxs_yx, seg_offsets=seg_idxs_offsets, reduction_func=FeatureGen.reduce_meanstd_nocast)
        meanstd_of_bw_seg = FeatureGen.apply_reduce_im2seg(img=flow_bw_padded, seg_im=seg_im, seg_ids=None, \
                                seg_fr_idxs=seg_fr_idxs, seg_bboxes_tlbr=seg_bboxes_tlbr, seg_masks_in_bboxes=seg_masks_in_bboxes, \
                                seg_idxs_yx=seg_idxs_yx, seg_offsets=seg_idxs_offsets, reduction_func=FeatureGen.reduce_meanstd_nocast)
        assert meanstd_of_fw_seg.shape[1:] == (4,)
        assert meanstd_of_bw_seg.shape[1:] == (4,)
        self.add_data('mean_of_fw_seg', meanstd_of_fw_seg[:,:2])
        self.add_data('std_of_fw_seg', meanstd_of_fw_seg[:,2:])
        self.add_data('mean_of_bw_seg', meanstd_of_bw_seg[:,:2])
        self.add_data('std_of_bw_seg', meanstd_of_bw_seg[:,2:])

        self.add_data('mean_occl_fw_seg', FeatureGen.apply_reduce_im2seg(img=occl_fw_padded[..., None], seg_im=seg_im, seg_ids=None, \
                                seg_fr_idxs=seg_fr_idxs, seg_bboxes_tlbr=seg_bboxes_tlbr, seg_masks_in_bboxes=seg_masks_in_bboxes, \
                                seg_idxs_yx=seg_idxs_yx, seg_offsets=seg_idxs_offsets, reduction_func=FeatureGen.reduce_mean))
        self.add_data('mean_occl_bw_seg', FeatureGen.apply_reduce_im2seg(img=occl_bw_padded[..., None], seg_im=seg_im, seg_ids=None, \
                                seg_fr_idxs=seg_fr_idxs, seg_bboxes_tlbr=seg_bboxes_tlbr, seg_masks_in_bboxes=seg_masks_in_bboxes, \
                                seg_idxs_yx=seg_idxs_yx, seg_offsets=seg_idxs_offsets, reduction_func=FeatureGen.reduce_mean))

        fvecs = FeatureGen.reduce_fmaps(fmap_dict=fmaps_dict, orig_size_yx=(480,854), seg_fr_idxs=seg_fr_idxs, \
                                        seg_idxs_yx=seg_idxs_yx, seg_offsets=seg_idxs_offsets)
        self.add_data('fmap_seg', fvecs.astype(np.float16))

        spatial_edges = FeatureGen.find_spatial_temporal_edges(seg_im=seg_im, edge_type='spatial')  # (n_edges, 2)
        self.add_data('spatial_edges', spatial_edges)
        flow_edges, flow_edge_fs = FeatureGen.find_flow_edges_and_gnn_features(seg_im=seg_im, seg_sizes=seg_sizes, \
                            flow_fwd=flow_fw_padded, flow_bwd=flow_bw_padded)  # (n_edges, 2), (n_edges, 2, 4)
        self.add_data('flow_edges', flow_edges)
        self.add_data('flow_edge_fs', flow_edge_fs)

        all_edges, all_edges_merger_indices = ImUtil.merge_edge_lists_only([spatial_edges, flow_edges], return_index=True)
        self.add_data('all_edges_merger_idxs', all_edges_merger_indices)

        mean_lab_diff_edgefs = FeatureGen.edgefs_segdist_pairwise_L2dist(seg_fs=self.get_data('mean_lab_seg'), edges=all_edges)
        self.add_data('mean_lab_diff_edgefs', mean_lab_diff_edgefs)
        mean_of_fw_diff_edgefs = FeatureGen.edgefs_optflow_diff(seg_fs_flow=self.get_data('mean_of_fw_seg'), edges=all_edges)
        self.add_data('mean_of_fw_diff_edgefs', mean_of_fw_diff_edgefs)
        mean_of_bw_diff_edgefs = FeatureGen.edgefs_optflow_diff(seg_fs_flow=self.get_data('mean_of_bw_seg'), edges=all_edges)
        self.add_data('mean_of_bw_diff_edgefs', mean_of_bw_diff_edgefs)

        
