
# 
# GNN_annot IJCNN 2021 implementation
#   A collection of utility functions for rendering and saving images with label and scribble overlays.
#   @author Viktor Varga
#

import os
import cv2
import numpy as np

def render_img_with_label_overlay(label_im, bgr_im, fg_label_colors_rgb, bgr_saturation=.0, label_alpha=0.5):
    '''
    Renders and saves images with predicted or true label overlays.
    Parameters:
        label_im: ndarray(sy, sx) of int; background is 0
        bgr_im: ndarray(sy, sx, 3:bgr) of uint8
        fg_label_colors_rgb: array-like of tuple(3,), RGB uint8 format
        bgr_saturation: float, in range [0,1], sets color saturation of rendered bgr image (0 is grayscale, 1 is fully saturated)
        label_alpha: float, in range (0,1], sets alpha value of label overlay
    Returns:
        out_im: ndarray(sy, sx, 3:bgr) of uint8
    '''
    assert 0. <= bgr_saturation <= 1.
    assert 0. < label_alpha <= 1.
    color_arr_bgr = np.array(fg_label_colors_rgb, dtype=np.float32)[:,::-1]
    assert color_arr_bgr.shape[1:] == (3,)
    color_arr_bgr = np.concatenate([np.zeros((1, 3), dtype=np.float32), color_arr_bgr], axis=0)  # adding bg as dummy 

    # render background, reduce color saturation if needed
    out_im = bgr_im.astype(np.float32, copy=True)
    if bgr_saturation < 1.:
        gray_im = np.broadcast_to(np.mean(out_im, axis=2)[:,:,None], out_im.shape)
        out_im = (1.-bgr_saturation)*gray_im + bgr_saturation*out_im

    # render label overlay
    fg_mask = label_im > 0
    overlay_im = color_arr_bgr[label_im,:]
    out_im[fg_mask,:] = (1.-label_alpha)*out_im[fg_mask,:] + label_alpha*overlay_im[fg_mask,:]
    out_im = np.clip(out_im, 0., 255.).astype(np.uint8)

    return out_im

def render_img_with_scribble_overlay(bgr_im, scribble_arrs, label_colors_rgb, bgr_saturation=.0, scribble_width=3):
    '''
    Renders and saves images with scribble overlays.
    Parameters:
        bgr_im: ndarray(sy, sx, 3:bgr) of uint8
        scribble_arrs: dict{lab - int: list(n_scribbles_with_lab) of ndarray(n_points_in_scribble, 2:[y,x]) of int32}
        label_colors_rgb: array-like of tuple(3,), RGB uint8 format, including bacground color
        bgr_saturation: float, in range [0,1], sets color saturation of rendered bgr image (0 is grayscale, 1 is fully saturated)
        scribble_width: int
    Returns:
        out_im: ndarray(sy, sx, 3:bgr) of uint8
    '''
    label_im = np.zeros(bgr_im.shape[:2], dtype=np.int32)   # labels are shifted by one, bg scribbles are represented by 1, ...
    for lab, scribs in scribble_arrs.items():
        scribs = [scribble_arr[:,::-1] for scribble_arr in scribs]  # y,x -> x,y
        cv2.polylines(label_im, scribs, isClosed=False, color=lab+1, thickness=scribble_width, lineType=cv2.LINE_8)

    out_im = render_img_with_label_overlay(label_im, bgr_im, label_colors_rgb, bgr_saturation=bgr_saturation, label_alpha=1.)
    return out_im


def save_imgs_with_label_overlay(root_folder_path, vidname, fname_prefix, label_ims, bgr_ims, bgr_saturation=.0, \
                                 label_alpha=0.5, render_scale=1.):
    '''
    Renders and saves images with predicted or true label overlays.
    Parameters:
        root_folder_path: str
        vidname: str
        fname_prefix: str; extra filename tag
        label_ims: ndarray(n_fr, sy, sx) of int; background is 0
        bgr_ims: ndarray(n_fr, sy, sx, 3:bgr) of uint8
        bgr_saturation: float, in range [0,1], sets color saturation of rendered bgr image (0 is grayscale, 1 is fully saturated)
        label_alpha: float, in range [0,1], sets alpha value of label overlay
        render_scale: float, in range [0.1,1], resolution of image can be scaled down
    '''
    assert bgr_ims.shape == label_ims.shape + (3,)
    assert label_ims.ndim == 3
    assert 0.1 <= render_scale <= 1.
    FG_LABEL_COLORS_RGB = [(255,0,0), (0,255,0), (255,255,0), (0,0,255), (255,0,255), (0,255,255), (255,255,255)]

    out_resolution_xy = (int(bgr_ims.shape[2]*render_scale), int(bgr_ims.shape[1]*render_scale))
    out_folder = os.path.join(root_folder_path, vidname)
    os.makedirs(out_folder, exist_ok=True)
    for fr_idx in range(bgr_ims.shape[0]):
        out_im = render_img_with_label_overlay(label_ims[fr_idx,:,:], bgr_ims[fr_idx,:,:,:], FG_LABEL_COLORS_RGB, \
                                                    bgr_saturation=bgr_saturation, label_alpha=label_alpha)
        out_fpath = os.path.join(out_folder, fname_prefix + str(fr_idx).zfill(5) + '.jpg')
        if render_scale < 1.:
            out_im = cv2.resize(out_im, out_resolution_xy)
        cv2.imwrite(out_fpath, out_im)

def save_img_with_scribble_overlay(root_folder_path, vidname, fname_prefix, scribbles, bgr_ims, bgr_saturation=.0, \
                                    scribble_width=3, render_scale=1.):
    '''
    Renders and saves a single image with current scribble overlays.
    Parameters:
        root_folder_path: str
        vidname: str
        fname_prefix: str; extra filename tag
        scribbles: format returned by DavisUtils.get_new_scribbles() -> tuple(2) of
                        annotated_frame_idx: int; the frame idx annotated in the CURRENT STEP
                        scribble_arrs: dict{lab - int: list(n_scribbles_with_lab) of ndarray(n_points_in_scribble, 2:[y,x]) of int32}
        bgr_ims: ndarray(n_fr, sy, sx, 3:bgr) of uint8
        bgr_saturation: float, in range [0,1], sets color saturation of rendered bgr image (0 is grayscale, 1 is fully saturated)
        scribble_width: int
        render_scale: float, in range [0.1,1], resolution of image can be scaled down
    '''
    assert bgr_ims.shape[3:] == (3,)
    assert 0.1 <= render_scale <= 1.
    BGFG_LABEL_COLORS_RGB = [(127,127,127), (255,0,0), (0,255,0), (255,255,0), (0,0,255), (255,0,255), (0,255,255), (255,255,255)]
    annot_fr_idx, scribble_arrs = scribbles

    out_resolution_xy = (int(bgr_ims.shape[2]*render_scale), int(bgr_ims.shape[1]*render_scale))
    out_folder = os.path.join(root_folder_path, vidname)
    os.makedirs(out_folder, exist_ok=True)

    out_im = render_img_with_scribble_overlay(bgr_ims[annot_fr_idx,:,:,:], scribble_arrs, BGFG_LABEL_COLORS_RGB, \
                                                    bgr_saturation=bgr_saturation, scribble_width=scribble_width)
    out_fpath = os.path.join(out_folder, fname_prefix + str(annot_fr_idx).zfill(5) + '.jpg')
    if render_scale < 1.:
        out_im = cv2.resize(out_im, out_resolution_xy)
    cv2.imwrite(out_fpath, out_im)
