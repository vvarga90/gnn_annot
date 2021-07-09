
# 
# GNN_annot IJCNN 2021 implementation
#   A collection of utility functions used in computing and updating graph features in VideoData.
#   @author Viktor Varga
#

import time

import numpy as np
import util.imutil as ImUtil
import util.util as Util


def reduce_mean(arr2d):
    # ndarray(n_pixels, n_chans) -> ndarray(n_chans,)
    assert arr2d.ndim == 2
    return np.mean(arr2d.astype(np.float64), axis=0, dtype=np.float64).astype(np.float32)

def reduce_std(arr2d):
    # ndarray(n_pixels, n_chans) -> ndarray(n_chans,)
    assert arr2d.ndim == 2
    return np.std(arr2d.astype(np.float64), axis=0, dtype=np.float64).astype(np.float32)

def reduce_meanstd_nocast(arr2d):
    # ndarray(n_pixels, n_chans) -> ndarray(n_chans*2:[means, stds],), no casting to float64 for increased precision
    assert arr2d.ndim == 2
    m = np.mean(arr2d, axis=0)  # (n_chans,)
    d = arr2d - m
    # following line (multiple dots independently) would be simpler with np.einsum('ij,ij->j'), but einsum is not using BLAS
    std = np.array([np.dot(d[:,ch_idx], d[:,ch_idx]) for ch_idx in range(arr2d.shape[-1])], dtype=m.dtype)
    std = np.sqrt(std / float(arr2d.shape[0]))
    return np.concatenate([m, std], axis=0)

def reduce_any(arr2d):
    # ndarray(n_pixels, n_chans) -> ndarray(n_chans,)
    assert (arr2d.ndim == 2) and (arr2d.dtype == np.bool_)
    return np.any(arr2d, axis=0)

def reduce_most_frequent_item(arr2d):
    # ndarray(n_pixels, n_chans) -> ndarray(1,)
    assert arr2d.shape[1:] == (1,)
    return np.array([np.argmax(np.bincount(arr2d.reshape(-1)))], dtype=arr2d.dtype)


def apply_reduce_im2seg(img, seg_im, seg_ids, seg_fr_idxs, seg_bboxes_tlbr, seg_masks_in_bboxes, \
                        seg_idxs_yx, seg_offsets, reduction_func, **func_kwargs):
    '''
    Applies 'reduction_func' over the pixels of each segment.
    Parameters:
        img: ndarray(n_frames, sy, sx, n_inchans) of ?
        seg_im: ndarray(n_frames, sy, sx) of i32
        seg_ids: None OR ndarray(n_segs,) of i32; if None, assuming IDs in range [0, seg_fr_idxs.shape[0])
        seg_fr_idxs: ndarray(n_segs,) of i32;
        seg_bboxes_tlbr: ndarray(n_segs, 4:bbox_tlbr) of i32; indexed with seg IDs
        
        seg_masks_in_bboxes: None OR list(n_segs) of ndarray(bbox_sy, bbox_sx) of bool_; optionally specify precomputed masks
        
        seg_idxs_yx: None OR ndarray(n_pixs, 2) of i32; segment indices; optionally specify precomputed segment idxs
        seg_offsets: None OR ndarray(n_segs+1,) of i32; offset for 'seg_idxs_yx'; idxs for seg#i are stored at seg_idxs_yx[seg_offsets[i]:seg_offsets[i+1],:]
        
        reduction_func: Callable with signature: (ndarray(n_items, n_inchans), **func_kwargs) -> (ndarray(n_outchans,))
        func_kwargs: named varargs; named parameters for 'reduction_func'.
    Returns:
        reduced_seg_arr: ndarray(n_segs, n_out_chans)
    '''
    assert img.ndim == 4
    assert seg_im.ndim == 3
    if seg_ids is None:
        seg_ids = np.arange(seg_fr_idxs.shape[0])
    assert seg_ids.shape == seg_fr_idxs.shape == seg_bboxes_tlbr.shape[:1]
    assert (seg_masks_in_bboxes is None) or (len(seg_masks_in_bboxes) == seg_ids.shape[0])
    masks_avail = seg_masks_in_bboxes is not None
    idxs_avail = seg_idxs_yx is not None
    assert (not idxs_avail) or (seg_offsets is not None)

    reduced_seg_arr = []
    for seg_id_idx in range(seg_ids.shape[0]):
        fr_idx = seg_fr_idxs[seg_id_idx]
        bbox_tlbr = seg_bboxes_tlbr[seg_id_idx]
        if masks_avail is True:
            seg_pixs = img[fr_idx, bbox_tlbr[0]:bbox_tlbr[2], bbox_tlbr[1]:bbox_tlbr[3], :][seg_masks_in_bboxes[seg_id_idx], :]
        elif idxs_avail is True:
            curr_seg_idxs = seg_idxs_yx[seg_offsets[seg_id_idx]:seg_offsets[seg_id_idx+1], :]   # (n_pix, 2)
            seg_pixs = img[fr_idx, curr_seg_idxs[:,0], curr_seg_idxs[:,1], :]
        else:
            seg_pixs, = ImUtil.get_segment_pixels_from_bbox(img[fr_idx], seg_im[fr_idx], bbox_tlbr, [seg_ids[seg_id_idx]])
        assert seg_pixs.shape[0] > 0
        reduced_seg_arr.append(reduction_func(seg_pixs, **func_kwargs))
    reduced_seg_arr = np.stack(reduced_seg_arr, axis=0)
    assert reduced_seg_arr.shape[:-1] == (seg_ids.shape[0],)
    return reduced_seg_arr

def reduce_im2seg_mean(img, seg_im, seg_sizes):
    '''
    Computes mean over the pixels of each segment using np.add.at unbuffered accumulation.
    Slower than most np.mean based methods.
    Parameters:
        img: ndarray(n_frames, sy, sx, n_chans) of ?
        seg_im: ndarray(n_frames, sy, sx) of i32
        seg_sizes: ndarray(n_segs,) of int32
    Returns:
        reduced_seg_arr: ndarray(n_segs, n_chans) of fl32
    '''
    assert img.ndim == 4
    assert seg_im.ndim == 3
    reduced_seg_arr = np.zeros((seg_sizes.shape[0], img.shape[-1]), dtype=np.float32)
    for ch_idx in range(img.shape[-1]):
        np.add.at(reduced_seg_arr[:,ch_idx], seg_im, img[..., ch_idx])
    reduced_seg_arr = reduced_seg_arr / seg_sizes[:,None]
    return reduced_seg_arr

def reduce_downscaledim2seg_mean(img_ds, orig_size_yx, seg_fr_idxs, seg_idxs_yx, seg_offsets):
    '''
    Computes mean over the pixels of each segment in downscaled images.
    Parameters:
        img_ds: ndarray(n_frames, ds_sy, ds_sx, n_chans) of fl32
        orig_size_yx: tuple(int, int); original image size
        seg_fr_idxs: ndarray(n_segs,) of i32;
        seg_idxs_yx: ndarray(n_all_idxs, 2:[y,x]) of int; containing all segment idxs
        seg_offsets: ndarray(n_segs+1,) of int; idxs for seg#i are stored at seg_idxs[seg_offsets[i]:seg_offsets[i+1],:]
    Returns:
        reduced_seg_arr: ndarray(n_segs, n_chans) of fl32
    '''
    assert img_ds.ndim == 4
    assert seg_fr_idxs.shape == (seg_offsets.shape[0]-1,)
    orig_size_yx = np.asarray(orig_size_yx, dtype=np.float32)
    assert orig_size_yx.shape == (2,)
    reduced_seg_arr = np.empty(seg_fr_idxs.shape + (img_ds.shape[-1],), dtype=np.float32)
    ds_factor_yx = orig_size_yx / img_ds.shape[1:3]
    seg_idxs_ds = (seg_idxs_yx / ds_factor_yx).astype(np.uint32)   # (n_idxs, 2:[y,x]) of ui32
    seg_idxs_ds_single = Util.view_multichannel_ui32_as_ui64(seg_idxs_ds, copy=False)   # (n_idxs,) of ui64, for faster np.unique()
    for seg_id in range(seg_fr_idxs.shape[0]):
        curr_seg_idxs_ds_single = seg_idxs_ds_single[seg_offsets[seg_id]:seg_offsets[seg_id+1]]   # (n_currseg_idxs,) of ui64
        curr_seg_idxs_ds_single_u, curr_seg_idxs_ds_c = np.unique(curr_seg_idxs_ds_single, return_counts=True)
        curr_seg_idxs_ds_u = Util.restore_multichannel_ui32_from_ui64(curr_seg_idxs_ds_single_u, copy=False)   # (n_u_currseg_idxs, 2:[y,x]) of ui32
        im_vecs_u = img_ds[seg_fr_idxs[seg_id], curr_seg_idxs_ds_u[:,0], curr_seg_idxs_ds_u[:,1], :]  # (n_u_currseg_idxs, n_ch) of fl32
        # using np.average:
        #reduced_seg_arr[seg_id,:] = np.average(im_vecs_u, axis=0, weights=curr_seg_idxs_ds_c)
        # using np.dot (faster, needs approx. 65% of time compared to np.average):
        reduced_seg_arr[seg_id,:] = np.dot(curr_seg_idxs_ds_c, im_vecs_u) / np.sum(curr_seg_idxs_ds_c)
    return reduced_seg_arr

def reduce_fmaps(fmap_dict, orig_size_yx, seg_fr_idxs, seg_idxs_yx, seg_offsets):
    '''
    Computes mean over the pixels of each segment in downscaled images.
    Parameters:
        fmap_dict: dict{str: ndarray(n_frames, ds_sy, ds_sx, n_chans) of fl32}
        orig_size_yx: tuple(int, int); original image size
        seg_fr_idxs: ndarray(n_segs,) of i32;
        seg_idxs_yx: ndarray(n_all_idxs, 2:[y,x]) of int; containing all segment idxs
        seg_offsets: ndarray(n_segs+1,) of int; idxs for seg#i are stored at seg_idxs[seg_offsets[i]:seg_offsets[i+1],:]
    Returns:
        reduced_seg_arr: ndarray(n_segs, n_chans_total) of fl32
    '''
    FMAP_KEYS_USED = ['expanded_conv_project_BN', 'block_2_add', 'block_5_add', 'block_12_add', 'block_16_project_BN']
    return np.concatenate([reduce_downscaledim2seg_mean(fmap_dict[fmap_key], orig_size_yx, seg_fr_idxs, seg_idxs_yx, seg_offsets) \
                                                    for fmap_key in FMAP_KEYS_USED], axis=-1)


def get_border_segments(seg_im):
    '''
    Applies 'reduction_func' over the pixels of each segment.
    Parameters:
        seg_im: ndarray(n_frames, sy, sx) of i32
    Returns:
        border_segs: ndarray(n_segs,) of i32
    '''
    assert seg_im.ndim == 3
    border_segs = []
    border_segs.append(seg_im[:,0,:].reshape(-1))
    border_segs.append(seg_im[:,-1,:].reshape(-1))
    border_segs.append(seg_im[:,:,0].reshape(-1))
    border_segs.append(seg_im[:,:,-1].reshape(-1))
    border_segs = np.unique(np.concatenate(border_segs, axis=0))
    return border_segs


def find_spatial_temporal_edges(seg_im, edge_type):
    '''
    Searches for spatial edges connecting any two segments.
    Parameters:
        seg_im: seg_im: ndarray(n_frames, sy, sx) of i32
        edge_type: str; either 'spatial' or 'temporal'
    Returns:
        new_edges: ndarray(n_new_edges, 2) of int32
    '''
    assert edge_type in ['spatial', 'temporal']
    assert seg_im.ndim == 3
    ignore_axes = [0] if edge_type == 'spatial' else [1,2]
    return ImUtil.get_adj_graph_edge_list_fast(seg_im, ignore_axes=ignore_axes)


# TODO use util.unique_2chan (2 locations)
# TODO remove seg_sets code
def find_flow_edges_and_gnn_features(seg_im, seg_sizes, flow_fwd, flow_bwd):
    '''
    Searches for flow edges connecting any two segments.
    The returned weights can be used for label propagation with the MRF model.
    Parameters:
        seg_im: ndarray(n_frames, sy, sx) of i32
        seg_sizes: ndarray(n_segs,) of int32
        flow_fwd: ndarray(n_frames, sy, sx, 2:[dy, dx]) of fl32
        flow_bwd: ndarray(n_frames, sy, sx, 2:[dy, dx]) of fl32
    Returns:
        new_edges: ndarray(n_new_edges, 2) of int32
        edgefs: ndarray(n_new_edges, 2:[ID_from -> ID_to, ID_to -> ID_from], n_features) of float32
                        FEATURES (per direction):
                            #0: ratio of ID_from OF vecs pointing to ID_to; (0,1]
                            #1: ratio of these (pointing to ID_to) vectors returning to ID_from with flow_bwd; (0,1]
                            #2-3: mean, std of distances of the returning bwd vectors (positive in #1) target points from the
                                        original starting points of fwd vectors
    '''
    assert seg_im.shape[0] == flow_fwd.shape[0] == flow_bwd.shape[0]
    seg_sets = [None]    # legacy, TODO remove
    new_edges = []
    edgefs = []
    for seg_id_set in seg_sets:
        # get segmentation map and flow images (cropped if a bbox is given)
        if seg_id_set is None:
            flow_fwd = flow_fwd[:-1,:,:]
            flow_bwd = flow_bwd[1:,:,:]
        else:
            assert False, "TODO check if needed, seg_obj was a Segmentation type parameter of the method, was replaced."
            bbox_stlebr = seg_obj.get_segs_bbox(seg_id_set)
            assert bbox_stlebr[3] - bbox_stlebr[0] >= 2
            seg_im = seg_obj.get_seg_region(bbox_stlebr=bbox_stlebr, framewise_seg_ids=False)
            flow_fwd = flow_fwd[bbox_stlebr[0]:bbox_stlebr[3]-1, bbox_stlebr[1]:bbox_stlebr[4], bbox_stlebr[2]:bbox_stlebr[5],:]
            flow_bwd = flow_bwd[bbox_stlebr[0]+1:bbox_stlebr[3], bbox_stlebr[1]:bbox_stlebr[4], bbox_stlebr[2]:bbox_stlebr[5],:]
        # 

        n_fr = seg_im.shape[0]
        sy, sx = seg_im.shape[1:3]
        assert flow_fwd.shape == flow_bwd.shape == (n_fr-1, sy, sx, 2)
        base_mgrid = np.mgrid[:sy, :sx].transpose((1,2,0))  # (sy, sx, 2)
        EPS = 1e-9
        if seg_sizes is None:
            _, seg_sizes = np.unique(seg_im, return_counts=True)
            assert seg_sizes.shape[0] == np.amax(seg_im)+1

        edges_fwd = []
        edges_bwd = []
        fs_fwd = []
        fs_bwd = []
        for direction in ["fwd", "bwd"]:
            edges_ls = edges_fwd if direction == "fwd" else edges_bwd
            fs_ls = fs_fwd if direction == "fwd" else fs_bwd

            for fr_idx in range(n_fr-1):
                # fwd/bwd is swapped when direction is backwards
                flow_to = flow_fwd[fr_idx] if direction == "fwd" else flow_bwd[fr_idx]
                flow_back = flow_bwd[fr_idx] if direction == "fwd" else flow_fwd[fr_idx]
                sp_seg_orig = seg_im[fr_idx] if direction == "fwd" else seg_im[fr_idx+1]
                sp_seg_target = seg_im[fr_idx+1] if direction == "fwd" else seg_im[fr_idx]
                #
                fwd_mgrid, fwdbwd_mgrid, invalid_fwd, invalid_fwdbwd = ImUtil.transform_dense_with_flow_fwdbwd(flow_to, flow_back)
                fwd_mgrid, fwdbwd_mgrid = fwd_mgrid.astype(np.int32), fwdbwd_mgrid.astype(np.int32)   # (sy, sx, 2) each
                fwdbwd_deltalen = np.linalg.norm(fwdbwd_mgrid - base_mgrid, axis=-1)   # (sy, sx)
                ids_fwd = sp_seg_target[fwd_mgrid[:,:,0], fwd_mgrid[:,:,1]]    # (sy, sx) i32
                ids_fwdbwd = sp_seg_orig[fwdbwd_mgrid[:,:,0], fwdbwd_mgrid[:,:,1]]    # (sy, sx) i32

                # feature#0, #1: count flow vectors for each edge and count how many of them return to the origin segment
                flow_vec_ids = np.stack([sp_seg_orig, ids_fwd], axis=-1).astype(np.int32, copy=False)   # (sy, sx, 2) i32
                flow_vec_ret_mask = sp_seg_orig == ids_fwdbwd                                  # (sy, sx) bool
                flow_vec_ids64 = Util.view_multichannel_i32_as_i64(flow_vec_ids, copy=False)       # (sy, sx) ui64
                u_edges64, inv_edges64, c_edges64 = np.unique(flow_vec_ids64, return_inverse=True, return_counts=True)
                returning_inv_edges64 = inv_edges64.reshape(flow_vec_ids64.shape)[flow_vec_ret_mask & (~invalid_fwdbwd)]
                c_edges_ret64 = np.bincount(returning_inv_edges64, minlength=u_edges64.shape[0])
                u_edges = Util.restore_multichannel_i32_from_i64(u_edges64, copy=False)       # (n_edges, 2:[from, to]) i32
                edge_from_sizes = seg_sizes[u_edges[:,0]].astype(np.float32)
                f0 = c_edges64/edge_from_sizes
                f1 = c_edges_ret64/(c_edges64 + EPS)
                assert np.all(f1 <= 1.)

                # feature#2, #3: select returning edges and get their deltas: mean/std over them grouped by the edge IDs
                fwdbwd_deltalen_returning = fwdbwd_deltalen[flow_vec_ret_mask & (~invalid_fwdbwd)]
                #f2 = Util.apply_func_with_groupby_manyIDs(fwdbwd_deltalen_returning, returning_inv_edges64, np.mean,\
                #                                                             assume_arange_to=u_edges64.shape[0], empty_val=0)  
                #f3 = Util.apply_func_with_groupby_manyIDs(fwdbwd_deltalen_returning, returning_inv_edges64, np.std,\
                #                                                             assume_arange_to=u_edges64.shape[0], empty_val=0)
                f23 = Util.get_meanstd_with_groupby_manyIDs(fwdbwd_deltalen_returning, returning_inv_edges64, \
                                                                              assume_arange_to=u_edges64.shape[0], empty_val=0)
                edges_ls.append(u_edges)
                fs_ls.append(np.stack([f0, f1, f23[:,0], f23[:,1]], axis=-1))

        # merge fwd and bwd edges
        edges = np.concatenate(edges_fwd + edges_bwd, axis=0)
        fs = np.concatenate(fs_fwd + fs_bwd, axis=0)
        edges_reversed = edges[:,0] > edges[:,1]
        edges[edges_reversed,:] = edges[edges_reversed,::-1]
        edges64 = Util.view_multichannel_i32_as_i64(edges, copy=False)
        edges64_u, edges_inv = np.unique(edges64, return_inverse=True)
        edges_u = Util.restore_multichannel_i32_from_i64(edges64_u, copy=False)
        edges_u_fs = np.zeros((edges_u.shape[0], 2, fs.shape[1]), dtype=fs.dtype)
        edges_u_fs[edges_inv, edges_reversed.astype(np.int32),:] = fs

        # only keep edges that have both ends in 'seg_id_set'
        if seg_id_set is not None:
            edges_kept_mask = np.all(np.isin(edges_u, seg_id_set), axis=1)
            edges_u = edges_u[edges_kept_mask,:]
            edges_u_fs = edges_u_fs[edges_kept_mask,:,:]
        new_edges.append(edges_u)
        edgefs.append(edges_u_fs)

    if len(new_edges) == 1:
        return new_edges[0], edgefs[0]
    else:
        new_edges, merge_idxs = ImUtil.merge_edge_lists_only(new_edges, return_index=True)
        edgefs = np.concatenate(edgefs, axis=0)
        edgefs = edgefs[merge_idxs,:]
        return new_edges, edgefs
    #


def edgefs_segdist_pairwise_L2dist(seg_fs, edges):
    '''
    Computes edge features by taking L2 distance of specified segment features for segments connected by the given edges.
    Parameters:
        seg_fs: ndarray(n_segs, 2) of float32
        edges: ndarray(n_edges, 2) of int32
    Returns: 
        edgefs: ndarray(n_edges, 1) of ?
    '''
    fs1 = seg_fs[edges[:,0],:]   # (n_edges, n_infeatures)
    fs2 = seg_fs[edges[:,1],:]   # (n_edges, n_infeatures)
    assert fs1.ndim == fs2.ndim == 2
    return np.linalg.norm(fs1.astype(np.float32, copy=False) - fs2.astype(np.float32, copy=False), axis=-1, keepdims=True)

def edgefs_optflow_diff(seg_fs_flow, edges):
    '''
    Computes difference features of two vectors: returns 3 values -> [abs_angular_diff, min_mag, rel_mag]
      abs_angular_diff: in [0,pi]; pi if vectors point to opposite dir, 0 if same dir
      min_mag: length of shorter vec; positive float
      rel_mag: shorter_vec_len/longer_vec_len, in range [0,1], 1 for similar lengths, 0 for infinitely different lengths
    Parameters:
        seg_fs_flow: ndarray(n_segs, 2:[dy, dx]) of float32
        edges: ndarray(n_edges, 2) of int32
    Returns: 
        edgefs: ndarray(n_edges, 3) of float32
    '''
    ofs1, ofs2 = seg_fs_flow[edges[:,0],:], seg_fs_flow[edges[:,1],:]   # (n_edges, 2:[dy, dx])
    assert ofs1.shape[0] == ofs2.shape[0]
    assert ofs1.shape[1:] == ofs2.shape[1:] == (2,)
    EPS = 1e-9
    edgefs = np.empty((ofs1.shape[0], 3), dtype=np.float32)
    angle1, angle2 = np.arctan2(ofs1[:,0], ofs1[:,1]), np.arctan2(ofs2[:,0], ofs2[:,1])
    edgefs[:,0] = np.fabs((angle1 - angle2 + np.pi) % (2.*np.pi) - np.pi)
    len1, len2 = np.linalg.norm(ofs1, axis=-1), np.linalg.norm(ofs2, axis=-1)
    edgefs[:,1] = np.minimum(len1, len2)
    max_len = np.maximum(len1, len2)
    edgefs[:,2] = edgefs[:,1]/(max_len + EPS)
    return edgefs



