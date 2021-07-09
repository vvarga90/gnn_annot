
# 
# GNN_annot IJCNN 2021 implementation
#   A collection of utility functions used in image handling.
#   @author Viktor Varga
#


import numpy as np
import cv2
import scipy.ndimage.filters

import util.util as Util

def fast_resize_video_nearest_singlech(ims, target_size_yx):
    '''
    Resizes video to target size, without smoothing, single channel.
    Parameters:
        ims: ndarray(n_frs, size_y, size_x) of ?
        target_size_yx: tuple(2) of int
    Returns:
        ndarray(n_frs, <target_size_yx>) of ?
    '''
    assert len(target_size_yx) == 2
    assert ims.ndim == 3
    assert ims.shape[1] < ims.shape[2]        # safety check
    ls0 = np.linspace(0., ims.shape[1], target_size_yx[0], endpoint=False, dtype=np.int32)
    ls1 = np.linspace(0., ims.shape[2], target_size_yx[1], endpoint=False, dtype=np.int32)
    return ims[:, ls0[:,None], ls1[None,:]]

def downscale_im_by_int_factor(im, ds_factor, ds_op):
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

def upscale_im_by_int_factor(im_ds, us_factor, target_size):
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

def block_downscale_video_singlech(ims, ds_factor, ds_func):
    '''
    Downscales video efficiently by pooling it with 'ds_func'. Remainder cols/rows are dropped.
    Parameters:
        ims: ndarray(n_frs, size_y, size_x) of ?
        ds_factor: int
        ds_func: downscale func taking tuple of int type 'axis' parameters, e.g. np.mean(), np.any(), ...
    Returns:
        ndarray(n_frs, size_y // ds_factor, size_x // ds_factor) of ?
    '''
    dsize_yx = (ims.shape[1] // ds_factor, ims.shape[2] // ds_factor)
    ims = ims[:, :(dsize_yx[0]*ds_factor), :(dsize_yx[1]*ds_factor)]
    ims = ims.reshape((ims.shape[0], dsize_yx[0], ds_factor, dsize_yx[1], ds_factor))
    return ds_func(ims, axis=(2,4))

def compute_pixelwise_labeling_error(true_lab_im, pred_lab_im, n_labels):
    '''
    Computes mean fg-only IoU error of true and predicted pixelwise labelings.
    Paramters:
        true_lab_im: ndarray(n_frames, sy, sx) of int
        pred_lab_im: ndarray(n_frames, sy, sx) of int
        n_labels: int; number of label categories
    '''
    assert true_lab_im.ndim == 3
    assert true_lab_im.shape == pred_lab_im.shape
    # Note: one-hot encoding below
    #       the ops below seem inefficient as O(n_pixels*n_labels) writing ops are necessary, instead of the minimum O(n_pixels) ops
    #       however, this is a much faster solution than creating one-hot encoding using mgrid advanced indexing or indexing np.eye(n_labels)
    # Following DAVIS benchmark, IoU is computed for each frame, each foreground object, and the mean is taken (bg labels are ignored)
    pred_labels = pred_lab_im[:,:,:,None] == np.arange(1, n_labels)   # (n_fr, sy, sx, n_fg_labels) of bool_
    true_labels = true_lab_im[:,:,:,None] == np.arange(1, n_labels)   # (n_fr, sy, sx, n_fg_labels) of bool_
    c_intersection = np.count_nonzero(pred_labels & true_labels, axis=(1,2))   # (n_fr, n_labels)
    c_union = np.count_nonzero(pred_labels | true_labels, axis=(1,2))   # (n_fr, n_labels)
    # replace indices where c_union is zero (label is not present -> iout is set to 1.)
    no_true_label_mask = c_union == 0
    c_intersection[no_true_label_mask] = 1
    c_union[no_true_label_mask] = 1
    return np.mean(np.mean(c_intersection / c_union, axis=1))   # first, taking mean of fg labels, then taking mean of frames


def get_segment_pixels_from_bbox(img, seg, bbox_tlbr, seg_ids):
    '''
    Extracts image data for segments within a single bbox as a list of pixel arrays.
    Parameters:
        img: ndarray(size_y, size_x, n_chans) of ?
        seg: ndarray(size_y, size_x) of int
        bbox_tlbr: tuple(4:[tlbr])
        seg_ids: ndarray(n_segs_to_get,) of int;
    Returns:
        list(n_segs_to_get) of ndarray(n_seg_pixels, n_chans) of img.dtype
    '''
    assert img.ndim == 3
    assert seg.shape == img.shape[:2]
    im_in_bbox = img[bbox_tlbr[0]:bbox_tlbr[2], bbox_tlbr[1]:bbox_tlbr[3]]
    seg_in_bbox = seg[bbox_tlbr[0]:bbox_tlbr[2], bbox_tlbr[1]:bbox_tlbr[3]]
    #for seg_id in seg_ids:
    #    print("ID:", seg_id, " -> ", np.count_nonzero(seg_in_bbox == seg_id), "in", seg_in_bbox.shape)
    return [im_in_bbox[seg_in_bbox == seg_id,:] for seg_id in seg_ids]

def get_masks_in_bboxes(seg_im, seg_fr_idxs, seg_bboxes_tlbr):
    '''
    Computes segment masks within bboxes for each segment.
    Parameters:
        seg_im: ndarray(n_frames, size_y, size_x) of int
        seg_fr_idxs: ndarray(n_segs,) of i32;
        seg_bboxes_tlbr: ndarray(n_segs, 4:bbox_tlbr) of i32; indexed with seg IDs
    Returns:
        seg_masks_in_bboxes: list(n_segs) of ndarray(bbox_sy, bbox_sx) of bool_
    '''
    assert seg_im.ndim == 3
    seg_masks_in_bboxes = []
    for seg_id in range(seg_fr_idxs.shape[0]):
        bbox_tlbr = seg_bboxes_tlbr[seg_id]
        mask = seg_im[seg_fr_idxs[seg_id], bbox_tlbr[0]:bbox_tlbr[2], bbox_tlbr[1]:bbox_tlbr[3]] == seg_id
        seg_masks_in_bboxes.append(mask)
    return seg_masks_in_bboxes

def get_mask_pix_idxs_in_bboxes(seg_im, seg_fr_idxs, seg_bboxes_tlbr, seg_masks_in_bboxes):
    '''
    Gets array of indexes of pixels masked within bboxes for each segment.
    Parameters:
        seg_im: ndarray(n_frames, size_y, size_x) of int
        seg_fr_idxs: ndarray(n_segs,) of i32;
        seg_bboxes_tlbr: ndarray(n_segs, 4:bbox_tlbr) of i32; indexed with seg IDs
        seg_masks_in_bboxes: None OR list(n_segs) of ndarray(bbox_sy, bbox_sx) of bool_
    Returns:
        seg_idxs_yx: ndarray(n_all_idxs, 2:[y,x]) of int; containing all segment idxs
        seg_offsets: ndarray(n_segs+1,) of int; idxs for seg#i are stored at seg_idxs[seg_offsets[i]:seg_offsets[i+1],:]
    '''
    assert seg_im.ndim == 3
    masks_avail = seg_masks_in_bboxes is not None
    seg_idxs_yx = np.full((np.prod(seg_im.shape), 2), dtype=np.int32, fill_value=-1)
    seg_offsets = np.zeros((seg_fr_idxs.shape[0]+1,), dtype=np.int32)
    for seg_id in range(seg_fr_idxs.shape[0]):
        bbox_tlbr = seg_bboxes_tlbr[seg_id]
        if masks_avail is True:
            idxs = np.argwhere(seg_masks_in_bboxes[seg_id])
        else:
            idxs = np.argwhere(seg_im[seg_fr_idxs[seg_id], bbox_tlbr[0]:bbox_tlbr[2], bbox_tlbr[1]:bbox_tlbr[3]] == seg_id)
        idxs += bbox_tlbr[:2]
        seg_offsets[seg_id+1] = seg_offsets[seg_id]+idxs.shape[0]
        seg_idxs_yx[seg_offsets[seg_id]:seg_offsets[seg_id+1],:] = idxs
    assert not np.any(np.isnan(seg_idxs_yx))
    return seg_idxs_yx, seg_offsets

# TODO can be faster with reinterpret cast (np.view)
def get_adj_graph_edge_list_fast(seg_im, ignore_axes=[], return_counts=False):
    '''
    Returns the edge list of the adjacency graph of the given SV segmentation.
    Faster method than the one in skimage, with fixed connectivity=1.
    Parameters:
        seg_im: ndarray(?) of int/uint up to 32bits; int64, uint64 are not supported
        ignore_axes: list of ints; these axes are ignored in the adjacency test.
        return_counts: bool; return the number of adjacencies for each seg pair
    Returns:
        edges: ndarray(n_edges, 2:[ID_from, ID_to]) of seg_im.dtype; unique edges where ID_from < ID_to
        (OPTIONAL if 'return_counts' is True) counts: ndarray(n_edges,) of i32
    '''
    ids_from = []
    ids_to = []
    MAXUINT32 = 2**32
    orig_dtype = seg_im.dtype
    assert np.can_cast(seg_im.dtype, np.uint32, casting='safe') or (seg_im.dtype == np.int32 and np.amin(seg_im) >= 0)
    seg_im = seg_im.astype(np.uint32, copy=False)

    for dim_idx in range(seg_im.ndim):
        if dim_idx in ignore_axes:
            continue

        # get each pixel along the given axis which have different IDs than the consecutive pixel along axis
        m = seg_im[(slice(None),)*dim_idx + (slice(None,-1,None),)] != seg_im[(slice(None),)*dim_idx + (slice(1,None,None),)]
        ids_from.append(seg_im[(slice(None),)*dim_idx + (slice(None,-1,None),)][m])
        ids_to.append(seg_im[(slice(None),)*dim_idx + (slice(1,None,None),)][m])

    # create edge arrays, sort: ids_from < ids_to
    edges = np.stack([np.concatenate(ids_from), np.concatenate(ids_to)], axis=1)  # (n_edges, 2)
    swap_mask = edges[:,0] > edges[:,1]
    edges[swap_mask,0], edges[swap_mask,1] = edges[swap_mask,1], edges[swap_mask,0]   # safe if advanced indexing

    # unique edges: unique 1D is faster than unique 2D
    edges_1d = edges[:,0] * MAXUINT32 + edges[:,1]
    if return_counts is True:
        edges_1d_u, edges_1d_c = np.unique(edges_1d, return_counts=True)
        edges = np.stack([edges_1d_u // MAXUINT32, edges_1d_u % MAXUINT32], axis=1)
        return edges.astype(orig_dtype, copy=False), edges_1d_c
    else:
        edges_1d_u = np.unique(edges_1d)
        edges = np.stack([edges_1d_u // MAXUINT32, edges_1d_u % MAXUINT32], axis=1)
        return edges.astype(orig_dtype, copy=False)
    #

# TODO use util.unique_2chan
def merge_edge_lists_only(edge_lists, return_index=False):
    '''
    Parameters:
        edge_lists: list(n_edge_lists) of ndarray(n_edges, 2:[ID_from, ID_to]) of int;
                    UNIQUE edges where ID_from < ID_to;
        return_index: bool; if True, returns index array which produces the merged
                                     edgelist from the concatenation of the source edgelists
    Returns:
        merged_edges: ndarray(n_all_edges, 2:[ID_from, ID_to]) of int; no uniqueness check if len(edge_lists) == 1
        (OPTIONAL) merger_indices: ndarray(n_all_edges,) of int32; see 'return_index' param for details
    '''
    assert all([earr.shape[1:] == (2,) and np.all(earr[:,0] < earr[:,1]) for earr in edge_lists])
    assert len(edge_lists) >= 1
    if len(edge_lists) == 1:
        return edge_lists[0]
    edges = np.concatenate(edge_lists, axis=0)
    edges64 = Util.view_multichannel_i32_as_i64(edges)
    if return_index:
        edges64_u, merger_indices = np.unique(edges64, return_index=True)
        merged_edges = Util.restore_multichannel_i32_from_i64(edges64_u)
        return merged_edges, merger_indices
    else:
        edges64_u,  = np.unique(edges64)
        merged_edges = Util.restore_multichannel_i32_from_i64(edges64_u)
        return merged_edges

def transform_dense_with_flow_fwdbwd(flow_fwd, flow_bwd):
    '''
    Transforms a (size_y, size_x) shaped mgrid with forward then with backward flow. 
        (fwd/bwd can be swapped to apply tranform in reversed direction)
    Parameters:
        flow_fwd: ndarray(size_y, size_x, 2:[dy,dx]) of float32
        flow_bwd: ndarray(size_y, size_x, 2:[dy,dx]) of float32
    Returns:
        fwd_mgrid: ndarray(size_y, size_x, 2:[y,x]) of float32; the fwd transformed mgrid point coordinates
        fwdbwd_mgrid: ndarray(size_y, size_x, 2:[y,x]) of float32; the fwd-bwd transformed mgrid point coordinates
                            invalid coordinates should be masked out with the following masks
        invalid_fwd: ndarray(size_y, size_x) of bool_; True where fwd transformation pointed out of screen
        invalid_fwdbwd: ndarray(size_y, size_x) of bool_; True where fwd or fwd-bwd transformation pointed out of screen
    '''
    assert flow_fwd.shape == flow_bwd.shape
    assert flow_fwd.shape[2:] == (2,)
    size_arr = np.array(flow_fwd.shape[:2], dtype=np.float32)
    #base = np.mgrid[:size_arr[0], :size_arr[1]]  # (2, sy, sx)
    base = (np.broadcast_to(np.arange(size_arr[0], dtype=np.float32)[:,None], flow_fwd.shape[:2]), \
            np.broadcast_to(np.arange(size_arr[1], dtype=np.float32)[None,:], flow_fwd.shape[:2]))    # replacing mgrid
    # forward transform
    
    fwd_mgrid = np.asarray(base, dtype=np.float32)  # (2, sy, sx)
    assert fwd_mgrid.shape == (2,) +  flow_fwd.shape[:2]
    fwd_mgrid += flow_fwd.transpose((2,0,1))
    # TODO remove, lines below replaced by single line above
    #fwd_mgrid[0,:,:] += flow_fwd[:,:,0]
    #fwd_mgrid[1,:,:] += flow_fwd[:,:,1]
    
    invalid_fwd = np.any((fwd_mgrid < 0.) | (fwd_mgrid >= size_arr[:,None,None]), axis=0)
    fwd_mgrid[:,invalid_fwd] = 0.
    # backward transform
    fwd_mgrid_i = fwd_mgrid.astype(np.int32, copy=True)  # (2, sy, sx) i32

    delta_bwd = flow_bwd.transpose((2,0,1))[:, fwd_mgrid_i[0,:,:], fwd_mgrid_i[1,:,:]]
    # TODO remove, lines below replaced by single line above
    #delta_bwd = np.empty_like(fwd_mgrid_i, dtype=np.float32)  # (2, sy, sx) fl32
    #delta_bwd[0,:,:] = flow_bwd[fwd_mgrid_i,0]
    #delta_bwd[1,:,:] = flow_bwd[fwd_mgrid_i,1]

    fwdbwd_mgrid = fwd_mgrid + delta_bwd        # (2, sy, sx) fl32
    invalid_fwdbwd = np.any((fwdbwd_mgrid < 0.) | (fwdbwd_mgrid >= size_arr[:,None,None]), axis=0)
    invalid_fwdbwd |= invalid_fwd
    fwdbwd_mgrid[:,invalid_fwdbwd] = 0.
    return fwd_mgrid.transpose((1,2,0)), fwdbwd_mgrid.transpose((1,2,0)), invalid_fwd, invalid_fwdbwd

# Optflow

def fill_missing_optflows(flows, dir_fw, max_val_limit):
    '''
    If optical flow is near zero in frame (e.g. because of repeated img frames), fill from previous frame
    Parameters:
        (MODIFIED) flows: ndarray(n_frames-1, sy, sx, 2:[dy, dx]) of float32
        dir_fw: bool; if True, frame #idx-1 is repeated, otherwise #idx+1 is repeated
        max_val_limit: float; a frame is considered to be missing optical flow if max pixel value is lower than this
    '''
    assert flows.shape[3:] == (2,)
    it_range = range(1, flows.shape[0], 1) if dir_fw else range(flows.shape[0]-2, -1, -1)
    for fr_idx in it_range:
        prev_fr_idx = fr_idx-1 if dir_fw else fr_idx+1
        if np.amax(np.fabs(flows[fr_idx])) < max_val_limit:
            flows[fr_idx,:,:,:] = flows[prev_fr_idx,:,:,:]
            print("Note: _fill_missing_optflows(): optical flow (fw: " + str(dir_fw) + ") in fr#" + \
                                                str(fr_idx) + " was replaced with fr#"  + str(prev_fr_idx))

def create_optflow_canny_imgs(flow_fw_viz, flow_bw_viz, canny_low, canny_high):
    '''
    Computes edges in optical flow visualization images by canny. Computes forward and backward edges independently.
    Parameters:
        flow_fw_viz, flow_bw_viz: ndarray(n_frames-1, sy, sx, 3:bgr) of uint8
        flow_fw, flow_bw: ndarray(n_frames-1, sy, sx, 2:[dy, dx]) of float32
    Returns:
        canny_ims: ndarray(n_frames, sy, sx, 2:[fw, bw]) of bool_;
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
    return canny_ims > 0

# Visualization

def mark_boundaries_fast(seg, mode='upper-left', bordertype='4way'):
    '''
    A faster alternative to skimage.segmentation.mark_boundaries. Produces a mask instead of a colored image.
    Parameters:
        seg: ndarray(sy, sx) of int
        mode: str; any of ['upper-left', 'lower-right', 'both']; indicates which side of the border should be marked
        bordertype: str; any of ['4way', '8way']; '8way' will include corners
    Returns:
        boundaries_mask: ndarray(sy, sx) of bool_
    '''
    assert seg.ndim == 2
    assert mode in ['upper-left', 'lower-right', 'both']
    assert bordertype in ['4way', '8way']
    boundaries_mask = np.zeros_like(seg, dtype=np.bool_)
    if mode == 'upper-left' or 'both':
        boundaries_mask[:-1,:] |= seg[:-1,:] != seg[1:,:]
        boundaries_mask[:,:-1] |= seg[:,:-1] != seg[:,1:]
        if bordertype == '8way':
            boundaries_mask[:-1,:-1] |= seg[:-1,:-1] != seg[1:,1:]
            boundaries_mask[:-1,1:] |= seg[:-1,1:] != seg[1:,:-1]
    if mode == 'lower-right' or 'both':
        boundaries_mask[1:,:] |= seg[:-1,:] != seg[1:,:]
        boundaries_mask[:,1:] |= seg[:,:-1] != seg[:,1:]
        if bordertype == '8way':
            boundaries_mask[1:,1:] |= seg[:-1,:-1] != seg[1:,1:]
            boundaries_mask[1:,:-1] |= seg[1:,:-1] != seg[:-1,1:]
    return boundaries_mask

def render_segmentation_edges_BGRA(segs, boundary_color_rgb, boundary_alpha=1.):
    '''
    Renders segmentation edges into a BGRA format image.
    Parameters:
        segs: ndarray(size_y, size_x) of uint/int; the segment IDs
        boundary_color_rgb: tuple(3) of uint8
        boundary_alpha: float; the alpha-channel value of the boundaries
    Returns:
        viz_img: ndarray(size_y, size_x, 4:n_ch:[bgra]) of uint8
    '''
    assert segs.ndim == 2
    assert len(boundary_color_rgb) == 3
    assert 0. <= boundary_alpha <= 1.
    '''
    # another version using skimage.segmentation.mark_boundaries()
    alpha_img = np.zeros_like(segs, dtype=np.float32)
    alpha_img = skimage.segmentation.mark_boundaries(alpha_img, segs, color=(1,1,1))  # (sy, sx, 3)
    alpha_img = (np.mean(alpha_img, axis=-1)*255.).astype(np.uint8)       # (sy, sx)
    viz_img = np.zeros(segs.shape + (4,), dtype=np.uint8)
    viz_img[alpha_img > 0, :3] = boundary_color_rgb[::-1]
    viz_img[:,:,3] = alpha_img
    '''
    border_mask = mark_boundaries_fast(segs, mode='upper-left', bordertype='4way')
    viz_img = np.zeros(segs.shape + (4,), dtype=np.uint8)
    viz_img[border_mask, :3] = boundary_color_rgb[::-1]
    viz_img[border_mask, 3] = int(boundary_alpha*255.)
    return viz_img

def render_segmentation_edges_BGR(bg_img, segs, boundary_color_rgb):
    '''
    Renders segmentation edges into a BGR format image.
    Parameters:
        (MODIFIED) bg_img: ndarray(size_y, size_x, 3:n_ch:[bgr]) of uint8; the background image with 'bg_alpha' opacity
        segs: ndarray(size_y, size_x) of uint/int; the segment IDs
        boundary_color_rgb: tuple(3) of uint8
    Returns:
        bg_img: ndarray(size_y, size_x, 3:n_ch:[bgr]) of uint8
    '''
    assert bg_img.shape[2:] == (3,)
    assert segs.shape == bg_img.shape[:2]
    edge_img = render_segmentation_edges_BGRA(segs, boundary_color_rgb, boundary_alpha=1.)  # (sy, sx, 4:bgra)
    bg_img[np.any(edge_img > 0, axis=-1), :] = boundary_color_rgb[::-1]
    return bg_img

def render_segmentation_labels_RGB(segs, seg_labels, label_colors_rgb, color_alpha, \
                                                bg=None, seg_is_gt=None, color_alpha_gt=1.):
    '''
    Renders segmentation labeling into an RGB format image with optional background.
    Parameters:
        segs: ndarray(size_y, size_x) of uint/int; the segment IDs (from 0..n_segs)
        seg_labels: ndarray(n_segs,) of int; the label for each segment; if negative, not filled
        label_colors_rgb: array-like(n_labels, 3:rgb) of uint8
        color_alpha: float; the alpha-channel value of the segment filling
        bg: None OR ndarray(size_y, size_x) of uint8 OR ndarray(size_y, size_x, 3:BGR) of uint8; optional background image
        seg_is_gt: None OR ndarray(n_segs,) of bool_; whether to use 'color_alpha_gt' opacity (True)
                                                                  or 'color_alpha' for each segment
        color_alpha_gt: float; the alpha-channel value of the segment filling for segments where 'seg_is_gt' is True
    Returns:
        viz_img: ndarray(size_y, size_x, 3:n_ch:[rgb]) of uint8
    '''
    assert segs.ndim == 2
    assert seg_labels.ndim == 1
    assert (seg_is_gt is None) or (seg_is_gt.shape == seg_labels.shape)
    label_colors_rgb = np.asarray(label_colors_rgb, dtype=np.uint8)
    assert label_colors_rgb.shape[1:] == (3,)
    assert 0. <= color_alpha <= 1.
    assert (bg is None) or (bg.shape == segs.shape) or (bg.shape == segs.shape + (3,))

    if (bg is not None) and (bg.ndim == 2):
        bg = np.broadcast_to(bg[:,:,None], bg.shape + (3,))

    # create label image
    label_colors_rgb_flat = Util.view_multichannel_ui8_to_ui32(label_colors_rgb)   # speedup

    seg_labeled = seg_labels[segs]
    assert seg_labeled.shape == segs.shape  # TODO remove
    label_colors_rgb_flat = np.insert(label_colors_rgb_flat, 0, 0)  # add dummy color for label -1
    seg_flatcolored = label_colors_rgb_flat[seg_labeled+1]
    assert seg_flatcolored.shape == segs.shape  # TODO remove
    viz_img = Util.restore_multichannel_ui8_from_ui32(seg_flatcolored, n_ch=3)

    # create alpha img
    alpha_values = np.array([0, float(color_alpha), float(color_alpha_gt)], dtype=np.float32) # (3:[unlabeled, pred, gt])
    alpha_img = np.zeros_like(segs, dtype=np.int32)
    alpha_img[:] = seg_labeled >= 0  # set labeled to 1
    if seg_is_gt is not None:
        seg_gt_image = seg_is_gt[segs]
        assert seg_gt_image.shape == segs.shape  # TODO remove
        alpha_img[seg_gt_image] = 2  # set gt to 2
    alpha_img = alpha_values[alpha_img]   # alpha_img type becomes float32
    alpha_img = alpha_img[:,:,None]    # (size_y, size_x, 1)

    # blend bg and labels with alpha
    if bg is None:
        viz_img = (alpha_img*viz_img).astype(np.uint8)
    else:
        viz_img = ((1.-alpha_img)*bg + alpha_img*viz_img).astype(np.uint8)

    return viz_img

def render_segmentation_floatvalues_RGB(segs, seg_fs_values, min_value_color_rgb, max_value_color_rgb, color_alpha, bg=None):
    '''
    Renders segmentation labeling into an RGB format image with optional background.
    Parameters:
        segs: ndarray(size_y, size_x) of uint/int; the segment IDs (from 0..n_segs)
        seg_fs_values: ndarray(n_segs,) of float; the feature value for each segment; clipped to [0,1]
        min_value_color_rgb: array-like(3:rgb) of uint8; color associated with the value 0
        max_value_color_rgb: array-like(3:rgb) of uint8; color associated with the value 1
        color_alpha: float; the alpha-channel value of the segment filling
        bg: None OR ndarray(size_y, size_x) of uint8 OR ndarray(size_y, size_x, 3:BGR) of uint8; optional background image
    Returns:
        viz_img: ndarray(size_y, size_x, 3:n_ch:[rgb]) of uint8
    '''
    assert segs.ndim == 2
    assert seg_fs_values.ndim == 1
    min_value_color_rgb = np.asarray(min_value_color_rgb, dtype=np.uint8)
    max_value_color_rgb = np.asarray(max_value_color_rgb, dtype=np.uint8)
    assert min_value_color_rgb.shape == max_value_color_rgb.shape == (3,)
    assert 0. <= color_alpha <= 1.
    assert (bg is None) or (bg.shape == segs.shape) or (bg.shape == segs.shape + (3,))

    if (bg is not None) and (bg.ndim == 2):
        bg = np.broadcast_to(bg[:,:,None], bg.shape + (3,))

    seg_fs_values = np.clip(seg_fs_values, 0., 1.)
    val_img = seg_fs_values[segs]
    viz_img = min_value_color_rgb*(1.-val_img[..., None]) + max_value_color_rgb*val_img[..., None]
    
    # blend bg and labels with alpha
    alpha_img = np.full((3,), fill_value=color_alpha, dtype=np.float32)    # (size_y, size_x, 1)
    if bg is None:
        viz_img = (alpha_img*viz_img).astype(np.uint8)
    else:
        viz_img = ((1.-alpha_img)*bg + alpha_img*viz_img).astype(np.uint8)

    return viz_img



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
        flow_bgr: ndarray(size_y, size_x, 3:bgr) of uint8
    """
    assert flow.ndim == 3
    assert flow.shape[2] == 2
    MAX_HUE = 179.
    flow_bgr = np.empty((flow.shape[0], flow.shape[1], 3), dtype=np.uint32)
    flow_bgr.fill(255)
    r, phi = cart2polar(flow[:, :, 1], flow[:, :, 0])
    flow_bgr[:, :, 0] = ((phi + np.pi) / (2. * np.pi) * MAX_HUE).astype(np.uint32)
    flow_bgr[:, :, 2] = (r * brightness_mul).astype(np.uint32)
    flow_bgr[:, :, 1:] = np.clip(flow_bgr[:, :, 1:], 0, 255)
    flow_bgr[:, :, 0] = np.clip(flow_bgr[:, :, 0], 0, int(MAX_HUE))
    flow_bgr = flow_bgr.astype(np.uint8)
    flow_bgr = cv2.cvtColor(flow_bgr, cv2.COLOR_HSV2BGR)
    return flow_bgr

def DEBUG_save_refine_map(fpath, refine_map):
    '''
    Renders and saves a refine map as a color image.
    Parameters:
        fpath: str
        refine_map: ndarray(downscaled_sy, downscaled_sx) of int
    '''
    COLORS_BGR = np.array([(0,0,192), (0,128,192), (0,160,160), (0,192,0), (160,160,0), (192,0,0)], dtype=np.uint8)
    refine_map_bgr = COLORS_BGR[refine_map,:]  # (dsy, dsx, 3)
    cv2.imwrite(fpath, refine_map_bgr)

def DEBUG_save_seg_video(fpath, ims, seg):
    '''
    Renders and saves a series of segmentation images into a video file.
    Parameters:
        fpath: str
        ims: ndarray(n_frames, sy, sx, 3:bgr) of ui8
        seg: ndarray(n_frames, sy, sx) of i32; video segmentation with IDs starting from 0
    '''
    assert ims.shape[3:] == (3,)
    assert seg.shape == ims.shape[:3]
    vr_fourcc = cv2.VideoWriter_fourcc(*'MJPG')     # use this codec with avi
    vr_fps = 25.  # vid_capture.get(cv2.CAP_PROP_FPS)
    vr_frSize_xy = (ims.shape[2], ims.shape[1])
    vid_writer = cv2.VideoWriter(fpath, fourcc=vr_fourcc, fps=vr_fps, frameSize=vr_frSize_xy)
    assert vid_writer.isOpened(), "Unable to open video file for writing: " + fpath

    for fr_idx in range(ims.shape[0]):
        im = ims[fr_idx,:,:,:].copy()
        render_segmentation_edges_BGR(im, seg[fr_idx,:,:], boundary_color_rgb=(255,255,0))
        vid_writer.write(im)

    vid_writer.release()
