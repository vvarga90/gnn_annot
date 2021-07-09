
#
# GNN_annot IJCNN 2021 implementation
#   Debugging GUI for the DAVIS interactive benchmark.
#   @author Viktor Varga
#

#
# USAGE - key bindings:
#   space -> step interactive benchmark forward
#   left/right arrow -> display prev/next frame
#   pageup/pagedown -> display current -5/+5-th frame
#   home/end -> display first/last frame
#   q/e -> display prev/next input feature (if in input feature / binary prediction display mode)
#                               the last feature shows the binary predictions
#   a/d -> display prev/next binary prediction round result (if in input feature / binary prediction display mode)
#   1 -> display video image only
#   2 -> display video image with ground truth (multiclass) annotation overlay
#   3 -> display video image with predicted (multiclass) annotation overlay (available after prediction was made)
#   4 -> display currently selected feature or binary prediction (available after prediction was made)
#   8 -> toggle superpixel segment edge overlay
#   9 -> move infobox display to the other side of the screen
#
#


import numpy as np

import tkinter as tk
from PIL import Image, ImageTk

import util.imutil as ImUtil
import util.davis_utils as DavisUtils
from datasets import DAVIS17
import cv2

import config as Config

class DAVISDebugGUI:

    '''
    Parameters:
        master: Tk; parent GUI item
        canvas: Tk.Canvas class for displaying image layers and text info
        davis_session: DavisInteractiveSession instance
        davis_iterator: iterator returned by DavisInteractiveSession.scribbles_iterator()
        curr_vidname: str
        label_estimation: None OR LabelEstimation instance
        seed_prop: None OR SeedPropagation instance

        n_input_features: int
        n_binpred_rounds: int

        davis_state_dict: dict{'prev_preds': see GNNLabelEstimation.set_prediction_davis_state() arg types,
                               'seed_hist': ...
                               'seed_prop_hist': ...}
        pred_label_im_to_submit: ndarray(n_frames, sy, sx) of int32; the pixelwise label predictions image to submit next to benchmark

        (BASE IMG ARRAYS, VIDEODATA)
        videodata_dict: dict{vidname - str: VideoData instance}
        bg_ims: ndarray(n_frs, size_y, size_x, n_ch=3 [BGR]) of uint8; original color images of current video
        bg_ims_gray: ndarray(n_frs, size_y, size_x) of uint8; original grayscale images of current video

        (DISPLAYED PHOTO IMAGES)
        photo_ims_rgb: list(n_frs) of tk.PhotoImage, RBG mode (3 channel uint8); CONSTANT FOR A SINGLE VIDEO
        photo_ims_gray: list(n_frs) of tk.PhotoImage, L mode (1 channel uint8); CONSTANT FOR A SINGLE VIDEO
        photo_ims_true_labs: list(n_frs) of tk.PhotoImage, RGB mode (3 channel uint8); true labs + gray img; CONSTANT FOR A SINGLE VIDEO
        photo_ims_sp_edges: list(n_frs) of tk.PhotoImage, RGBA mode (4 channel uint8); CONSTANT FOR A SINGLE VIDEO
                NOTE: sp edge images are not updated, even though segmentation might change due to segment splits
        photo_ims_pred_labs: list(n_frs) of tk.PhotoImage, RGB mode (3 channel uint8); pred labs + gray img; MODIFIED ON UPDATE
        
        photo_ims_dynamic: list(n_frs) of tk.PhotoImage, RGB mode (3 channel uint8); current features/binpreds; 
                                    MODIFIED ON UPDATE AND when changing self.curr_feat_idx or self.curr_binpred_idx

        (DISPLAY STATE)
        curr_fr_idx: int; current frame to be shown
        curr_feat_idx: int; current input feature to be shown
        curr_binpred_idx: int; current binary prediction sub-result within the OvR to be shown
        bg_display_mode: str; any of ['bg_rgb', 'bg_gray+true', 'bg_gray+pred', 'features']
        fg_boundaries_toggle: if True, the SP boundaries are shown
        infobox_left_corner_toggle: if True, the infobox is shown in the upper left corner,
                                                                     otherwise in the upper right corner

        iter_state: str; any of ['show_none', 'show_scribble', 'show_seeds', 'show_prediction']
        prev_scribble_dict, curr_scribble_dict: None or dict; returned by the DAVIS iterator as part of a tuple, at position #1
        curr_annot_fr_idx: int
        curr_scribbles: dict{fg_lab - int: list(n_scribbles_with_lab) of ndarray(n_points_in_scribble, 2:[y,x]) of int32}
        pred_metrics: dict{'metric_name': list(n_inputs) of float}; initial state (all -1 labels) is not considered

        prev_input_feats: ndarray(n_binpred_rounds, n_nodes, n_input_features) of fl32
        prev_binpreds: ndarray(n_binpred_rounds, n_nodes) of fl32

    '''

    LABEL_COLORS_RGB = np.array([[128,128,128], [255,0,0], [0,255,0], [0,0,255],\
                                                [255,140,0], [0,255,255], [255,0,255],\
                                                [192,0,192], [127,0,255], [0,128,255],\
                                                [255,255,0], [0,204,102], [51,153,255]], dtype=np.uint8)
    BOUNDARY_COLOR_RGB = (255,0,255)
    FEATURE_COLOR_MIN_BGR = (255,255,255)
    FEATURE_COLOR_MAX_BGR = (128,0,0)
    COLORTUPLE2HEX = lambda rgb_tup: '#%02x%02x%02x' % (rgb_tup[0], rgb_tup[1], rgb_tup[2])

    FEATURES_VMIN = np.zeros((20,), dtype=np.float32) 
                            # 20 features: the last feature is the binary prediction probability,
                            #        18 can be both session idx (optional feature) or the binary probability as they have a simialr range
    FEATURES_VMAX = np.ones((20,), dtype=np.float32)

    def __init__(self, master, videodata_dict, davis_session, lab_est=None, seed_prop=None):
        '''
        Parameters:
            master: Tk instance, parent GUI item
            videodata_dict: dict{vidname - str: VideoData instance}
            davis_session: DavisInteractiveSession instance
            lab_est: None OR LabelEstimation instance
            seed_prop: None OR SeedPropagation instance
        '''
        self.master = master
        master.title("DAVIS Interactive Benchmark - Debug Window")
        self.davis_session = davis_session
        self.davis_iterator = self.davis_session.scribbles_iterator()
        self.videodata_dict = videodata_dict
        self.curr_vidname = None
        self.label_estimation = lab_est
        self.seed_prop = seed_prop
        self.n_input_features = 19 if Config.GNN_ENABLE_NODE_FEATURE_SESSION_STEP_IDX is True else 18

        self.curr_fr_idx = 0
        self.curr_feat_idx = 0
        self.curr_binpred_idx = 0
        self.bg_display_mode = 'bg_gray+pred'
        self.fg_boundaries_toggle = False
        self.infobox_left_corner_toggle = False

        self.photo_ims_pred_labs = []
        self.photo_ims_dynamic = []

        # create canvas
        self.canvas = tk.Canvas(self.master, width=854, height=480)
        self.canvas.pack()

        # bind keypress functions, see Event key details: https://anzeljg.github.io/rin2/book2/2405/docs/tkinter/key-names.html
        self.master.bind("<Left>", self.on_keypress_left)
        self.master.bind("<Right>", self.on_keypress_right)
        self.master.bind("<Prior>", self.on_keypress_pageup)   # PageUp
        self.master.bind("<Next>", self.on_keypress_pagedown)   # PageDown
        self.master.bind("<Home>", self.on_keypress_home)
        self.master.bind("<End>", self.on_keypress_end)
        self.master.bind("q", self.on_keypress_q)
        self.master.bind("e", self.on_keypress_e)
        self.master.bind("a", self.on_keypress_a)
        self.master.bind("d", self.on_keypress_d)
        self.master.bind("<Key-1>", self.on_keypress_1)
        self.master.bind("<Key-2>", self.on_keypress_2)
        self.master.bind("<Key-3>", self.on_keypress_3)
        self.master.bind("<Key-4>", self.on_keypress_4)
        self.master.bind("<Key-8>", self.on_keypress_8)
        self.master.bind("<Key-9>", self.on_keypress_9)
        self.master.bind("<Key-space>", self.on_keypress_space)

        # Step iterator
        self.iter_state = "show_none"
        self._step_davis_iterator()

        # draw images
        self._generate_new_prediction_image()
        self._redraw_bg_img()
        self._redraw_fg_edge_img()
        self._redraw_info_textbox()


    def _step_davis_iterator(self):
        '''
        Queries the DAVIS iterator.
        '''
        try:
            if self.iter_state == 'show_none':
                # INIT & SKIP some sequences if needed

                SKIP_N_VIDEOS = 0
                SKIP_N_SEQS = 0

                # SKIP videos / sequencies
                for seq_idx in range((SKIP_N_VIDEOS*3 + SKIP_N_SEQS)*8):
                    vidname, scribble, is_new_sequence = next(self.davis_iterator)
                    pred_shape = self.videodata_dict[vidname].get_seg().get_shape()
                    pred_shape = (pred_shape[0],) + DAVIS17.CUSTOM_IMSIZE_DICT[vidname] \
                                            if vidname in DAVIS17.CUSTOM_IMSIZE_DICT.keys() else pred_shape
                    dummy_pred_masks = np.zeros(pred_shape, dtype=np.int32)
                    self.davis_session.submit_masks(dummy_pred_masks)
            else:
                # IF NOT FIRST ITER -> submit predicted masks, resize predictions if necessary
                curr_videodata = self.videodata_dict[self.curr_vidname]
                pred_label_im = self.pred_label_im_to_submit
                if self.curr_vidname in DAVIS17.CUSTOM_IMSIZE_DICT.keys():
                    pred_label_im = ImUtil.fast_resize_video_nearest_singlech(pred_label_im, \
                                                                        DAVIS17.CUSTOM_IMSIZE_DICT[self.curr_vidname])

                # explicit frame query options
                DAVIS_frame_query_method = 'default'   # 'default', 'equidistant', 'choose_from_distant'
                assert DAVIS_frame_query_method in ['default', 'equidistant', 'choose_from_distant']
                annotated_fr_idxs = [ann_fr_idx for (ann_fr_idx, _) in self.davis_state_dict['seed_hist']]  # list must have at least one item here
                if DAVIS_frame_query_method == 'default':
                    frames_to_query = None
                elif DAVIS_frame_query_method == 'equidistant':
                    query_ratios = [0.5, 0.25, 0.75, 0.125, 0.375, 0.625, 0.875, 0.]
                    next_fr = int(query_ratios[len(annotated_fr_idxs)-1] * curr_videodata.get_seg().get_n_frames())
                    frames_to_query = [min(max(0, next_fr), curr_videodata.get_seg().get_n_frames()-1)]
                elif DAVIS_frame_query_method == 'choose_from_distant':
                    frame_dists = np.ones((curr_videodata.get_seg().get_n_frames(),), dtype=np.int32)
                    frame_dists[annotated_fr_idxs] = 0
                    frame_dists = distance_transform_cdt(frame_dists)
                    frames_to_query = list(np.argsort(frame_dists)[(3*frame_dists.shape[0])//4:])
                    frames_to_query = frames_to_query


                DavisUtils.fix_label_image_error(self.curr_vidname, pred_label_im)
                self.davis_session.submit_masks(pred_label_im, next_scribble_frame_candidates=frames_to_query)

            # STEP DAVIS benchmark
            vidname, scribble, is_new_sequence = next(self.davis_iterator)
            curr_videodata = self.videodata_dict[vidname]

            if self.curr_vidname != vidname:
                # IF new video
                assert is_new_sequence
                self.curr_vidname = vidname
                self.curr_scribbles = None
                self._update_constant_imgs()

                # init/reset label estimation, seedprop
                self.label_estimation.set_prediction_video(vidname=vidname, videodata=curr_videodata)
                self.n_binpred_rounds = curr_videodata.get_data('n_labels')
                self.n_binpred_rounds = 1 if self.n_binpred_rounds == 2 else self.n_binpred_rounds
                self.n_binpred_rounds = self.n_binpred_rounds*2 if Config.PREDICTION_DOUBLE_ONE_VS_REST else self.n_binpred_rounds

            if is_new_sequence:
                # IF new sequence (same or new video, new annotation session)
                self.pred_metrics = {'mean_j_raw':[]}
                self.prev_scribble_dict = None
                self.curr_scribble_dict = None
                self.davis_state_dict = {'prev_preds': None, 'seed_hist': [], 'seed_prop_hist': []}
                self.pred_label_im_to_submit = None

            self.prev_scribble_dict = self.curr_scribble_dict
            self.curr_scribble_dict = scribble

            self.curr_annot_fr_idx, self.curr_scribbles = DavisUtils.get_new_scribbles(vidname, self.prev_scribble_dict,\
                                                                             self.curr_scribble_dict, (480, 854))
            self.curr_fr_idx = self.curr_annot_fr_idx   # jump to annotated frame


        except StopIteration:
            pass
        #

    def _update_label_predictions(self):
        '''
        Updates self.pred_metrics
        '''
        # set davis session state of prediction generator in label estimation, predict new labels, then fetch updated session state
        curr_videodata = self.videodata_dict[self.curr_vidname]

        self.label_estimation.reset_prediction_generator()
        self.label_estimation.set_prediction_davis_state(self.curr_annot_fr_idx, self.curr_scribbles, \
                    self.davis_state_dict['prev_preds'], self.davis_state_dict['seed_hist'], \
                                                         self.davis_state_dict['seed_prop_hist'])
        self.davis_state_dict['prev_preds'] = self.label_estimation.predict_all(return_probs=True)
        davis_state_prev_preds_am = np.argmax(self.davis_state_dict['prev_preds'], axis=-1)

        prev_gs = self.label_estimation.prev_pred_gs
        self.prev_input_feats = np.stack([prev_g.ndata['fs'].numpy() for prev_g in prev_gs], axis=0)   # (n_samples, n_nodes, n_fs)
        self.prev_binpreds = self.label_estimation.prev_binpred_ys

        _, self.davis_state_dict['seed_hist'], self.davis_state_dict['seed_prop_hist'] = \
                                                        self.label_estimation.get_prediction_davis_state()
        seg_im = curr_videodata.get_seg().get_seg_im(framewise_seg_ids=False)
        self.pred_label_im_to_submit = davis_state_prev_preds_am[seg_im]
        # get metrics
        for metric_name in self.pred_metrics.keys():
            if metric_name == 'mean_j_raw':
                n_labels = curr_videodata.get_data('n_labels')
                true_lab_im = curr_videodata.get_data('annot_im')
                metric_val = ImUtil.compute_pixelwise_labeling_error(true_lab_im, self.pred_label_im_to_submit, n_labels)
            else:
                true_lab_seg = curr_videodata.get_data('annot_seg')
                metric_val = ImUtil.compute_segmentation_labeling_error(true_lab_seg, davis_state_prev_preds_am, \
                                            n_labels, metric_name, seg_sizes=curr_videodata.get_seg().get_seg_sizes())
            self.pred_metrics[metric_name].append(metric_val)


    def _update_constant_imgs(self):
        '''
        Updates all constant image arrays and PhotoImage lists for the current video.
        Sets self.bg_ims_gray,
            self.photo_ims_rgb, self.photo_ims_gray, self.photo_ims_true_labs, self.photo_ims_sp_edges.
        '''
        curr_videodata = self.videodata_dict[self.curr_vidname]
        bg_ims = curr_videodata.get_data('bgr_im')
        assert bg_ims.shape[3:] == (3,)
        assert bg_ims.dtype == np.uint8
        self.bg_ims = bg_ims[...,::-1]  # BGR -> RGB
        self.bg_ims_gray = np.mean(self.bg_ims, axis=-1).astype(np.uint8)

        self.photo_ims_rgb = []
        self.photo_ims_gray = []
        self.photo_ims_true_labs = []
        self.photo_ims_sp_edges = []

        seg_im_fwise = curr_videodata.get_seg().get_seg_im(framewise_seg_ids=True)
        true_lab_seg = curr_videodata.get_data('annot_seg')
        color_arr = np.array(DAVISDebugGUI.LABEL_COLORS_RGB).astype(np.uint8)
        assert color_arr.shape[1:] == (3,)

        for fr_idx in range(seg_im_fwise.shape[0]):

            # create colored & grayscale background photo image
            self.photo_ims_rgb.append(ImageTk.PhotoImage(Image.fromarray(self.bg_ims[fr_idx])))
            self.photo_ims_gray.append(ImageTk.PhotoImage(Image.fromarray(self.bg_ims_gray[fr_idx])))

            # create gray bg + true label image
            offset = curr_videodata.get_seg().get_fr_seg_id_offset(fr_idx)
            end_offset = curr_videodata.get_seg().get_fr_seg_id_end_offset(fr_idx)
            seg_im = seg_im_fwise[fr_idx]
            seg_labels = true_lab_seg[offset:end_offset]
            # im_labs = ImUtil.render_segmentation_labels_RGB(seg_im, seg_labels, DAVISDebugGUI.LABEL_COLORS_RGB, \
            #                  color_alpha=.5, bg=self.bg_ims_gray[fr_idx], seg_is_gt=None, color_alpha_gt=1.)
            im_labs = curr_videodata.get_data('annot_im')[fr_idx,:,:]
            im_labs = color_arr[im_labs,:]

            self.photo_ims_true_labs.append(ImageTk.PhotoImage(Image.fromarray(im_labs)))

            # create SP edges overlay
            im_edges = ImUtil.render_segmentation_edges_BGRA(seg_im, DAVISDebugGUI.BOUNDARY_COLOR_RGB, boundary_alpha=1.)
            im_edges[:,:,3] = 80    # overwriting alpha: PIL.ImageTk.PhotoImage class extremely slow with complicated alpha channels
            self.photo_ims_sp_edges.append(ImageTk.PhotoImage(Image.fromarray(im_edges, 'RGBA')))
        #

    def _generate_feature_images(self, feat_idx, binpred_idx):
        '''
        Generates a list of PhotoImages showing input feature #feat_idx (or bin preds as the last feature) for
                        OvR binary prediction round #binpred_idx and adds it to self.photo_ims_inputfeats_dict.
        '''
        if feat_idx == self.n_input_features:
            feat_arr = self.prev_binpreds[binpred_idx,:]   # (n_nodes,)
        else:
            feat_arr = self.prev_input_feats[binpred_idx, :, feat_idx]   # (n_nodes,)
        vmin, vmax = DAVISDebugGUI.FEATURES_VMIN[feat_idx], DAVISDebugGUI.FEATURES_VMAX[feat_idx]
        feat_arr = (feat_arr - vmin)/(vmax-vmin)

        self.photo_ims_dynamic = []
        curr_videodata = self.videodata_dict[self.curr_vidname]
        seg_im_fwise = curr_videodata.get_seg().get_seg_im(framewise_seg_ids=True)
        for fr_idx in range(seg_im_fwise.shape[0]):
            offset = curr_videodata.get_seg().get_fr_seg_id_offset(fr_idx)
            end_offset = curr_videodata.get_seg().get_fr_seg_id_end_offset(fr_idx)
            seg_im = seg_im_fwise[fr_idx]
            im_features = ImUtil.render_segmentation_floatvalues_RGB(seg_im, feat_arr[offset:end_offset], \
                            DAVISDebugGUI.FEATURE_COLOR_MIN_BGR, DAVISDebugGUI.FEATURE_COLOR_MAX_BGR, color_alpha=1., bg=None)

            self.photo_ims_dynamic.append(ImageTk.PhotoImage(Image.fromarray(im_features)))
        #

    def _generate_new_prediction_image(self):
        '''
        Updates self.photo_ims_pred_labs.
        '''
        self.photo_ims_pred_labs = []
        curr_videodata = self.videodata_dict[self.curr_vidname]
        seg_im_fwise = curr_videodata.get_seg().get_seg_im(framewise_seg_ids=True)
        pred_labels = self.davis_state_dict['prev_preds']
        if pred_labels is None:
            pred_labels = np.full((curr_videodata.get_seg().get_n_segs_total(),), dtype=np.int32, fill_value=-1)
        else:
            pred_labels = np.argmax(pred_labels, axis=-1)

        for fr_idx in range(seg_im_fwise.shape[0]):
            offset = curr_videodata.get_seg().get_fr_seg_id_offset(fr_idx)
            end_offset = curr_videodata.get_seg().get_fr_seg_id_end_offset(fr_idx)
            seg_im = seg_im_fwise[fr_idx]
            seg_labels = pred_labels[offset:end_offset]
            im_labs = ImUtil.render_segmentation_labels_RGB(seg_im, seg_labels, DAVISDebugGUI.LABEL_COLORS_RGB, \
                             color_alpha=.5, bg=self.bg_ims_gray[fr_idx], seg_is_gt=None, color_alpha_gt=1.)

            self.photo_ims_pred_labs.append(ImageTk.PhotoImage(Image.fromarray(im_labs)))

    def _generate_info_text(self):
        '''
        Returns:
            text: str
        '''
        text = []
        text.append("Video name: " + str(self.curr_vidname) + ",   frame# " + str(self.curr_fr_idx))
        iter_idx = len(self.davis_state_dict['seed_hist'])
        iter_idx = iter_idx-1 if self.iter_state in ['show_seeds', 'show_prop_seeds', 'show_prediction'] else iter_idx
        text.append("STATE: " + self.iter_state + ", Iter# " + str(iter_idx))
        if self.bg_display_mode == 'bg_rgb':
            text.append("DISPLAYING color images")
        elif self.bg_display_mode == 'bg_gray+true':
            text.append("DISPLAYING true annotation")
        elif self.bg_display_mode == 'bg_gray+pred':
            text.append("DISPLAYING predicted annotation")
        elif self.bg_display_mode == 'features':
            if self.curr_feat_idx == self.n_input_features:
                text.append("DISPLAYING bin pred, OvR round#" + str(self.curr_binpred_idx))
            else:
                text.append("DISPLAYING input feats#" + str(self.curr_feat_idx) + ", OvR round#" + str(self.curr_binpred_idx))

        annot_fr_idxs = [ann_fr_idx for (ann_fr_idx, _) in self.davis_state_dict['seed_hist']]
        mean_j_raw_metrics = self.pred_metrics['mean_j_raw']
        if self.iter_state in ['show_scribble']:
            annot_fr_idxs.append(self.curr_annot_fr_idx)
        if self.iter_state in ['show_seeds', 'show_prop_seeds']:
            mean_j_raw_metrics = mean_j_raw_metrics[:-1]
        text.append("QUERIES: " + str(annot_fr_idxs))
        mean_j_raw_metrics_str = str([str(item)[:5] for item in mean_j_raw_metrics]) if len(mean_j_raw_metrics) >= 1 else "-"
        text.append("mJraw: " + mean_j_raw_metrics_str)

        return '\n'.join(text)


    # QUERIES

    def _get_bg_photoimg(self, fr_idx):
        '''
        Parameters:
            fr_idx: int
        Returns:
            ImageTk.PhotoImage: the color or grayscale background image at the specified frame
        '''
        assert 0 <= fr_idx < len(self.photo_ims_rgb)
        assert self.bg_display_mode in ['bg_rgb', 'bg_gray+true', 'bg_gray+pred', 'features']
        if self.bg_display_mode == 'bg_rgb':
            return self.photo_ims_rgb[fr_idx]
        elif self.bg_display_mode == 'bg_gray+true':
            return self.photo_ims_true_labs[fr_idx]
        elif self.bg_display_mode == 'bg_gray+pred':
            return self.photo_ims_pred_labs[fr_idx]
        elif self.bg_display_mode == 'features':
            return self.photo_ims_dynamic[fr_idx]

    def _get_fg_edgeimg(self, fr_idx):
        '''
        Parameters:
            fr_idx: int
        Returns:
            None OR ImageTk.PhotoImage: the RGBA label overlay image at the specified frame (None if no overlay is shown)
        '''
        assert 0 <= fr_idx < len(self.photo_ims_true_labs)
        if self.fg_boundaries_toggle:
            return self.photo_ims_sp_edges[fr_idx]
        else:
            return None


    # CONTROL

    def _redraw_bg_img(self):
        '''
        Redraws background image. Deletes drawn user click circles.
        '''
        self.canvas.delete('bg')
        self.canvas.create_image((0,0), image=self._get_bg_photoimg(self.curr_fr_idx), anchor='nw', tags=('bg',))
        self.canvas.tag_lower('bg')

    def _redraw_fg_edge_img(self):
        '''
        Redraws SP edge overlay.
        '''
        if len(self.canvas.find_withtag('fg-edge')) > 0:
            self.canvas.delete('fg-edge')
        edge_im = self._get_fg_edgeimg(self.curr_fr_idx)
        if edge_im is None:
            return
        self.canvas.create_image((0,0), image=edge_im, anchor='nw', tags=('fg-edge',))
        self.canvas.tag_raise('fg-edge')
        # raise infobox Z level
        self.canvas.tag_raise('infobox')

    def _redraw_info_textbox(self):
        '''
        Redraws info textbox.
        '''
        self.canvas.delete('infobox')
        if self.infobox_left_corner_toggle:
            infobox_pos = (20,20)
        else:
            infobox_pos = (420,20)
        info_text = self._generate_info_text()
        FONT = ("Courier", 11, 'bold')
        self.canvas.create_text(infobox_pos, text=info_text, fill='#00FF00', anchor='nw', \
                                width=400, font=FONT, tags=('infobox',))

    def _redraw_scribbles(self):
        '''
        Redraws scribble points.
        '''
        if len(self.canvas.find_withtag('scribble')) > 0:
            self.canvas.delete('scribble')
        if self.curr_annot_fr_idx != self.curr_fr_idx:
            return
        if self.iter_state != 'show_scribble':
            return
        if len(self.canvas.find_withtag('seed')) > 0:
            self.canvas.delete('seed')

        for scribble_lab in self.curr_scribbles.keys():
            for scribble in self.curr_scribbles[scribble_lab]:
                scribble_ps_xy = scribble[:,::-1]   # (n_points, [x,y])
                fill_color_hex = DAVISDebugGUI.COLORTUPLE2HEX(DAVISDebugGUI.LABEL_COLORS_RGB[scribble_lab])
                self.canvas.create_line(*scribble_ps_xy.reshape(-1), fill=fill_color_hex, tags=('scribble',), width=2)

        # raise infobox Z level
        self.canvas.tag_raise('infobox')

    def _redraw_seeds(self):
        '''
        Redraws seed points.
        '''
        if len(self.canvas.find_withtag('seed')) > 0:
            self.canvas.delete('seed')
        if self.iter_state not in ['show_seeds', 'show_prop_seeds']:
            return
        if (self.iter_state == 'show_seeds') and (self.curr_annot_fr_idx != self.curr_fr_idx):
            return
        if len(self.canvas.find_withtag('scribble')) > 0:
            self.canvas.delete('scribble')

        RADIUS = 3
        curr_ann_fr_idx, curr_seed_points = self.davis_state_dict['seed_hist'][-1]
        assert curr_ann_fr_idx == self.curr_annot_fr_idx
        if self.curr_fr_idx == self.curr_annot_fr_idx:
            for (pos_y, pos_x, lab) in curr_seed_points:
                fill_color_hex = DAVISDebugGUI.COLORTUPLE2HEX(DAVISDebugGUI.LABEL_COLORS_RGB[lab])
                self.canvas.create_oval(pos_x-RADIUS, pos_y-RADIUS, pos_x+RADIUS, pos_y+RADIUS, \
                                        fill=fill_color_hex, outline='black', tags=('seed',))

        if self.iter_state == 'show_prop_seeds':
            curr_seed_points_prop = self.davis_state_dict['seed_prop_hist'][-1]
            if self.curr_fr_idx in curr_seed_points_prop.keys():
                curr_frame_prop_points = curr_seed_points_prop[self.curr_fr_idx]
                for (pos_y, pos_x, lab) in curr_frame_prop_points:
                    fill_color_hex = DAVISDebugGUI.COLORTUPLE2HEX(DAVISDebugGUI.LABEL_COLORS_RGB[lab])
                    self.canvas.create_oval(pos_x-RADIUS, pos_y-RADIUS, pos_x+RADIUS, pos_y+RADIUS, \
                                            fill=fill_color_hex, outline='yellow', tags=('seed',))

        # raise infobox Z level
        self.canvas.tag_raise('infobox')


    # ON KEY EVENTS

    def on_keypress_left(self, event):
        if self.curr_fr_idx > 0:
            self.curr_fr_idx -= 1
        self._redraw_bg_img()
        self._redraw_fg_edge_img()
        self._redraw_info_textbox()
        self._redraw_scribbles()
        self._redraw_seeds()
        self.canvas.update()

    def on_keypress_right(self, event):
        if self.curr_fr_idx < len(self.photo_ims_rgb)-1:
            self.curr_fr_idx += 1
        self._redraw_bg_img()
        self._redraw_fg_edge_img()
        self._redraw_scribbles()
        self._redraw_seeds()
        self._redraw_info_textbox()
        self.canvas.update()

    def on_keypress_pageup(self, event):
        self.curr_fr_idx = max(self.curr_fr_idx-5, 0)
        self._redraw_bg_img()
        self._redraw_fg_edge_img()
        self._redraw_scribbles()
        self._redraw_seeds()
        self._redraw_info_textbox()
        self.canvas.update()

    def on_keypress_pagedown(self, event):
        self.curr_fr_idx = min(self.curr_fr_idx+5, len(self.photo_ims_rgb)-1)
        self._redraw_bg_img()
        self._redraw_fg_edge_img()
        self._redraw_scribbles()
        self._redraw_seeds()
        self._redraw_info_textbox()
        self.canvas.update()

    def on_keypress_home(self, event):
        self.curr_fr_idx = 0
        self._redraw_bg_img()
        self._redraw_fg_edge_img()
        self._redraw_scribbles()
        self._redraw_seeds()
        self._redraw_info_textbox()
        self.canvas.update()

    def on_keypress_end(self, event):
        self.curr_fr_idx = len(self.photo_ims_rgb)-1
        self._redraw_bg_img()
        self._redraw_fg_edge_img()
        self._redraw_scribbles()
        self._redraw_seeds()
        self._redraw_info_textbox()
        self.canvas.update()

    def on_keypress_q(self, event):
        if self.curr_feat_idx > 0:
            self.curr_feat_idx -= 1
        self._generate_feature_images(self.curr_feat_idx, self.curr_binpred_idx)
        self._redraw_bg_img()
        self._redraw_fg_edge_img()
        self._redraw_info_textbox()
        self._redraw_scribbles()
        self._redraw_seeds()
        self.canvas.update()

    def on_keypress_e(self, event):
        if self.curr_feat_idx < self.n_input_features:
            self.curr_feat_idx += 1
        self._generate_feature_images(self.curr_feat_idx, self.curr_binpred_idx)
        self._redraw_bg_img()
        self._redraw_fg_edge_img()
        self._redraw_scribbles()
        self._redraw_seeds()
        self._redraw_info_textbox()
        self.canvas.update()

    def on_keypress_a(self, event):
        if self.curr_binpred_idx > 0:
            self.curr_binpred_idx -= 1
        self._generate_feature_images(self.curr_feat_idx, self.curr_binpred_idx)
        self._redraw_bg_img()
        self._redraw_fg_edge_img()
        self._redraw_info_textbox()
        self._redraw_scribbles()
        self._redraw_seeds()
        self.canvas.update()

    def on_keypress_d(self, event):
        if self.curr_binpred_idx < self.n_binpred_rounds-1:
            self.curr_binpred_idx += 1
        self._generate_feature_images(self.curr_feat_idx, self.curr_binpred_idx)
        self._redraw_bg_img()
        self._redraw_fg_edge_img()
        self._redraw_scribbles()
        self._redraw_seeds()
        self._redraw_info_textbox()
        self.canvas.update()

    def on_keypress_1(self, event):
        self.bg_display_mode = 'bg_rgb'
        self._redraw_bg_img()
        self._redraw_info_textbox()
        self.canvas.update()

    def on_keypress_2(self, event):
        self.bg_display_mode = 'bg_gray+true'
        self._redraw_bg_img()
        self._redraw_info_textbox()
        self.canvas.update()

    def on_keypress_3(self, event):
        self.bg_display_mode = 'bg_gray+pred'
        self._redraw_bg_img()
        self._redraw_info_textbox()
        self.canvas.update()

    def on_keypress_4(self, event):
        self.bg_display_mode = 'features'
        self._redraw_bg_img()
        self._redraw_info_textbox()
        self.canvas.update()

    def on_keypress_8(self, event):
        self.fg_boundaries_toggle = not self.fg_boundaries_toggle
        self._redraw_fg_edge_img()
        self.canvas.update()

    def on_keypress_9(self, event):
        self.infobox_left_corner_toggle = not self.infobox_left_corner_toggle
        self._redraw_info_textbox()
        self.canvas.update()

    def on_keypress_space(self, event):
        ''' 
        Prediction is updated and evaluated, predicted segmentation is rendered and shown.
        'self.iter_state' flow diagram:
            show_none -> show_scribble -> show_seeds -> [show_prop_seeds ->] show_prediction -> show_scribble -> ...
        '''
        if self.iter_state == 'show_none':
            # don't step DAVIS iterator as it was stepped on initialization
            self.iter_state = 'show_scribble'
            self._redraw_scribbles()
        elif self.iter_state == 'show_scribble':
            # update prediction here, but do not draw it yet - only show new seeds (generate model input feature images as well)
            self._update_label_predictions()
            self._generate_feature_images(self.curr_feat_idx, self.curr_binpred_idx)
            self.iter_state = 'show_seeds'
            self._redraw_seeds()
        elif self.iter_state == 'show_seeds':
            # show new propagated seeds
            self.iter_state = 'show_prop_seeds'
            self._redraw_seeds()
        elif self.iter_state == 'show_prop_seeds':
            # generate prediction images and show prediction
            self.iter_state = 'show_prediction'
            self._generate_new_prediction_image()
        elif self.iter_state == 'show_prediction':
            # step DAVIS iterator and show new scribble points
            self._step_davis_iterator()
            self.iter_state = 'show_scribble'
            self._redraw_scribbles()
        else:
            assert False

        # redraw display
        self._redraw_bg_img()
        self._redraw_info_textbox()
        self.canvas.update()






