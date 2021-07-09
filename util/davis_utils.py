
# 
# GNN_annot IJCNN 2021 implementation
#   A collection of utility functions corresponding to processing the DAVIS Interactive benchamrk scribbles/submissions.
#   @author Viktor Varga
#

import numpy as np
import cv2

import util.imutil as ImUtil

def get_new_scribbles(vidname, prev_scribble_dict, curr_scribble_dict, im_size_yx):
    '''
    Extracts labeled seed points from newly returned scribbles. The new scribbles are found by comparing the previously
            returned scribble dict with the updated one.
    Parameters:
        vidname: str; only for handling the 'tennis' sequence label error
        prev_scribble_dict: None OR dict; dict returned in PREVIOUS STEP at #1 by iterator 
                                    from DavisInteractiveSession.scribbles_iterator()
                                    None if there was no previous step;
                                    See: https://interactive.davischallenge.org/docs/session/
        curr_scribble_dict: dict; dict returned in CURRENT STEP at #1 by iterator
                                    from DavisInteractiveSession.scribbles_iterator()
        im_size_yx: tuple(2); target image size
    Returns:
        annotated_frame_idx: int; the frame idx annotated in the CURRENT STEP
        scribble_arrs: dict{lab - int: list(n_scribbles_with_lab) of ndarray(n_points_in_scribble, 2:[y,x]) of int32}
    '''
    im_size_yx = np.array(im_size_yx, dtype=np.int32)
    assert im_size_yx.shape == (2,)

    # get newly annotated frame index
    curr_lens = np.array([len(scrib_fr) for scrib_fr in curr_scribble_dict['scribbles']], dtype=np.int32)
    prev_lens = np.zeros((curr_lens.shape[0],), dtype=np.int32) if prev_scribble_dict is None else \
                                np.array([len(scrib_fr) for scrib_fr in prev_scribble_dict['scribbles']], dtype=np.int32)
    diff_lens = curr_lens - prev_lens
    if np.count_nonzero(diff_lens) > 1:
        # Rarely happens in DAVIS benchmark: i.e. one of the 'lindy-hop' & 'longboard' sequences contains initial scribbles in two frames
        print("!!! davis_utils.get_new_scribbles(): Warning! Multiple frames were annotated since previous scribble dict. ")
    elif np.count_nonzero(diff_lens) < 1:
        # TODO might happen?
        print("!!! davis_utils.get_new_scribbles(): Warning! No frames were annotated since previous scribble dict. ")
    annotated_frame_idx = np.argmax(diff_lens)

    scribble_arrs = {}
    for scribble_data in curr_scribble_dict['scribbles'][annotated_frame_idx]:
        scrib_points, label = scribble_data['path'], scribble_data['object_id']
        if label == 255:
            assert vidname == 'tennis'
            print("!!! davis_utils.get_new_scribbles(): Info: label value 255 was replaced with 3 (suspecting 'tennis' sequence label error) ")
            label = 3
        scrib_points = np.round(np.array(scrib_points, dtype=np.float64)[:,::-1] * im_size_yx).astype(np.int32)
        scrib_points = np.clip(scrib_points, a_min=None, a_max=im_size_yx-1)
        assert scrib_points.shape[1:] == (2,)
        #assert scrib_points.shape[0] >= 2    # assuming at least two points per scribble
        if scrib_points.shape[0] < 2:
            print("!!! Warning! DavisUtils.get_new_scribbles() -> Less than 2 points in scribble, shape:", scrib_points.shape)
        if label not in scribble_arrs.keys():
            scribble_arrs[label] = []
        scribble_arrs[label].append(scrib_points)

    return annotated_frame_idx, scribble_arrs

def render_scribbles(scribble_arrs, im_size_yx, generate_bg_mask=False):
    '''
    Renders all scribbles into masks for each label.
    Parameters:
        scribble_arrs: dict{fg_lab - int: list(n_scribbles_with_lab) of ndarray(n_points_in_scribble, 2:[y,x]) of int32}
                                keys must be from arange(1, n_fg_labels+1); bg label is not allowed
        im_size_yx: tuple(2) of ints
        generate_bg_mask: bool; if True, the method expects no background scribbles and generates extra bg seed points (DAVIS step#1)
                           if False, the method does not generate extra bg points (DAVIS step#2...)
    Returns:
        labs_present: ndarray(n_labels_among_scribbles,) of int32; the corresponding labels to the masks
        scribble_masks: ndarray(n_labels_among_scribbles, sy, sx) of int32; bg mask is at label idx#-1 if 'generate_bg_mask' is True
                            pixel values are (indices of corresponding scribbles)+1
    '''
    labs_present = list(scribble_arrs.keys())
    assert len(labs_present) >= 1
    scribble_masks = np.zeros((len(labs_present)+1,) + im_size_yx, dtype=np.int32)  # +1 to leave space for bg if needed

    for lab_idx in range(len(labs_present)):
        # render all scribbles to images, each label category independently
        lab = int(labs_present[lab_idx])
        scribbles_ls = scribble_arrs[lab]
        scribbles_ls = [scribble_arr[:,::-1] for scribble_arr in scribbles_ls]  # y,x -> x,y lowres
        for scrib_idx in range(len(scribbles_ls)):
            cv2.polylines(scribble_masks[lab_idx,:,:], [scribbles_ls[scrib_idx]], isClosed=False, color=scrib_idx+1, \
                                                                                thickness=1, lineType=cv2.LINE_8)
    if generate_bg_mask is True:
        assert 0 not in labs_present
        DILATE_RADIUS = 6
        DOWNSCALE_FACTOR = 8

        # get mask showing all scribbles at once, then downscale
        fullres_any_scribble = np.any(scribble_masks, axis=0)
        lowres_any_scribble = ImUtil.downscale_im_by_int_factor(fullres_any_scribble, DOWNSCALE_FACTOR, np.any).astype(np.uint8)

        # dilate any-scribble mask
        pad_width = DILATE_RADIUS+1
        kernel_dilate = np.empty((2*DILATE_RADIUS+1, 2*DILATE_RADIUS+1), dtype=np.uint8)
        kernel_dilate[:] = 0
        kernel_dilate = cv2.circle(kernel_dilate, (DILATE_RADIUS, DILATE_RADIUS), radius=DILATE_RADIUS, color=1, thickness=-1)
        lowres_any_scribble = np.pad(lowres_any_scribble, ((pad_width, pad_width), (pad_width, pad_width)))
        lowres_any_scribble = cv2.dilate(lowres_any_scribble, kernel_dilate, iterations=1)
        lowres_any_scribble = lowres_any_scribble[pad_width:-pad_width, pad_width:-pad_width]  # unpad

        # upscale dilated any-scribble mask, then sample background from the negated mask
        scribble_masks[-1,:,:] = ImUtil.upscale_im_by_int_factor(1-lowres_any_scribble, DOWNSCALE_FACTOR, im_size_yx)  # negation op '~' works differently with uint8
        labs_present.append(0)
        
    return scribble_masks, labs_present

def davis_scribbles2seeds_uniform(scribble_arrs, im_size_yx, n_seeds_per_cat, generate_bg=False, sample_from_each_scribble=True):
    '''
    Extracts seed points from DAVIS scribbles. 
    Uniform sampling within each category, based on pixels of rendered scribbles. No error correction.
    Parameters:
        scribble_arrs: dict{fg_lab - int: list(n_scribbles_with_lab) of ndarray(n_points_in_scribble, 2:[y,x]) of int32}
                                keys must be from arange(1, n_fg_labels+1); bg label is not allowed
        im_size_yx: tuple(2) of ints
        n_seeds_per_cat: int; return this many seed points per category
        generate_bg: bool; if True, the method expects no background scribbles and generates extra bg seed points (DAVIS step#1)
                           if False, the method does not generate extra bg points (DAVIS step#2...)
        sample_from_each_scribble: bool; if True, tries to sample from each scribble at least one seed
    Returns:
        ret_scribble_arrs: ndarray(n_seeds, 3:[y,x, label]) of int32
    '''
    all_scrib_labels = list(scribble_arrs.keys())
    if len(all_scrib_labels) == 0:
        print("!!!! davis_utils.davis_scribbles2seeds_uniform(): Warning! Got empty scribble dict, empty array is returned. !!!! ")
        return np.zeros((0, 3), dtype=np.int32)
    scribble_masks, labs_present = render_scribbles(scribble_arrs, im_size_yx, generate_bg_mask=generate_bg)
    ret_scribble_arrs = []
    for lab_idx in range(len(labs_present)):
        lab = labs_present[lab_idx]
        lab_pixs = np.argwhere(scribble_masks[lab_idx,:,:])  # (c_bg_true, 2), transposed np.where
        n_seed_to_sample_from_lab = n_seeds_per_cat
        if (sample_from_each_scribble is True) and ((lab != 0) or (generate_bg is False)):
            # if 'sample_from_each_scribble' and not generated bg, random sample a single seed from each scribble
            shuffler = np.random.permutation(lab_pixs.shape[0])
            lab_pixs = lab_pixs[shuffler,:]
            lab_pix_seed_idxs = scribble_masks[lab_idx, lab_pixs[:,0], lab_pixs[:,1]]
            u_seed_idxs, first_idx_in_seed = np.unique(lab_pix_seed_idxs, return_index=True)
            if first_idx_in_seed.shape[0] > n_seed_to_sample_from_lab:
                # if too many scribbles, select randomly
                np.random.shuffle(first_idx_in_seed)
                first_idx_in_seed = first_idx_in_seed[:n_seed_to_sample_from_lab]
                print("Warning! Too many scribbles, could not sample from each one.")
            lab_pix_chosen = lab_pixs[first_idx_in_seed,:]
            lab_pix_chosen_labeled = np.pad(lab_pix_chosen, ((0,0), (0,1)), constant_values=lab)   # (n_seeds_per_cat, 3)
            ret_scribble_arrs.append(lab_pix_chosen_labeled)
            n_seed_to_sample_from_lab -= lab_pix_chosen_labeled.shape[0]
        # sample the remaining randomly from all scribbles
        if n_seed_to_sample_from_lab > 0:
            sample_idxs = np.random.choice(lab_pixs.shape[0], size=(n_seed_to_sample_from_lab,))
            lab_pixs_labeled = np.pad(lab_pixs[sample_idxs,:], ((0,0), (0,1)), constant_values=lab)   # (n_seeds_per_cat, 3)
            ret_scribble_arrs.append(lab_pixs_labeled)

    return np.concatenate(ret_scribble_arrs, axis=0)

def fix_label_image_error(vidname, pred_label_im):
    '''
    In case of the 'tennis' sequence, replaces the tennis-ball segment label value 3 with the 
        original erroneous value 255 to preserve arange(n_labels) range for labels.
    Parameters:
        vidname: str
        (MODIFIED) pred_label_im: ndarray(n_frames, sy, sx) of ui8
    '''
    if vidname == 'tennis':
        pred_label_im[pred_label_im == 255] = 3
    #

