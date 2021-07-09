
# 
# GNN_annot IJCNN 2021 implementation
#   Runner script to run the DAVIS Interactive benchmark on a trained GNN model.
#   @author Viktor Varga
#

import sys
sys.path.append('./davis-interactive-gnn-annot-training')

import os
import numpy as np

from cache_manager import CacheManager
import config as Config
from datasets import DAVIS17
import util.davis_utils as DavisUtils
import util.imutil as ImUtil
import util.util_paper as PaperUtils
import util.featuregen as FeatureGen

from seed_propagation.basic_optflow_seed_propagation import BasicOptflowSeedPropagation
from label_estimation.logreg_label_model import LogRegLabelModel
from label_estimation.gnn_label_estimation import GNNLabelEstimation
from davisinteractive.session import DavisInteractiveSession    # install from pip, see https://interactive.davischallenge.org

if __name__ == '__main__':

    # TODO remove unused utils

    #SET_NAME = 'debug_test (1+1+3)'
    SET_NAME = 'test_only (1+1+30)'

    RENDER_RESULTS_PAPER = False   # TEMP, TODO remove after paper figures are generated
    RENDER_RESULTS_PAPER_PATH = './paper_qual_figs/'
    
    print("CONFIG --->")
    print({var: Config.__dict__[var] for var in dir(Config) if not var.startswith("__")})
    print("<--- CONFIG")

    cache_manager = CacheManager()
    DatasetClass = CacheManager.get_dataset_class()
    test_vidnames = DatasetClass.get_video_set_vidnames(SET_NAME, 'test')
    
    cache_manager.load_videos(test_vidnames)
    videodata_dict = {vidname: cache_manager.get_videodata_obj(vidname) for vidname in test_vidnames}
    max_n_labels = max([videodata_dict[vidname].get_data('n_labels') for vidname in test_vidnames])

    print("Loaded videos & data:", SET_NAME)

    # INIT LabelModel & SeedProp
    lab_model = LogRegLabelModel()
    seedprop_alg = BasicOptflowSeedPropagation(videodata_dict) if Config.USE_SEEDPROP is True else None

    # INIT LabelEstimation, load trained GNN model
    lab_est = GNNLabelEstimation(label_model=lab_model, seedprop_alg=seedprop_alg)
    checkpoint_path = os.path.join(Config.GNN_CHECKPOINT_DIR, Config.GNN_CHECKPOINT_NAME_TO_LOAD)
    lab_est.load_pretrained_gnn(checkpoint_path)

    # ---------------------------- DAVIS Interactive Challenge - benchmark evaluation ----------------------------

    DAVIS_report_dir = './_davis_interactive_reports/'
    os.makedirs(DAVIS_report_dir, exist_ok=True)
    
    # Configuration used in the challenges
    DAVIS_max_nb_interactions = 8 # Maximum number of interactions 
    DAVIS_max_time_per_interaction = 30 # Maximum time per interaction per object (in seconds)

    # Total time available to interact with a sequence and an initial set of scribbles
    DAVIS_max_time = DAVIS_max_nb_interactions * DAVIS_max_time_per_interaction # Maximum time per object

    # Metric to optimize
    DAVIS_metric = 'J'
    assert DAVIS_metric in ['J', 'F', 'J_AND_F']

    DAVIS_frame_query_method = 'default'   # 'default', 'equidistant', 'choose_from_distant'

    with DavisInteractiveSession(host='localhost',
                            user_key=None,
                            davis_root=DAVIS17.DAVIS_ROOT_FOLDER,
                            subset='val',
                            shuffle=False,
                            max_time=DAVIS_max_time,
                            max_nb_interactions=DAVIS_max_nb_interactions,
                            metric_to_optimize=DAVIS_metric,
                            report_save_dir=DAVIS_report_dir) as sess:
                            #report_save_dir=None) as sess:

        curr_vidname = None
        pred_metric_mean_j_im = None
        davis_iter = sess.scribbles_iterator()

        # iterate DAVIS benchmark
        annotated_fr_idxs = []
        seq_idx = 0
        step_idx = -1
        for vidname, scribble, is_new_sequence in davis_iter:

            print("--> DAVIS STEP: ", vidname, is_new_sequence)
            step_idx += 1

            # on new video, load video data
            if vidname != curr_vidname:
                seq_idx = -1
                curr_vidname = vidname
                curr_videodata = videodata_dict[vidname]
                assert is_new_sequence

                # init/reset label estimation, seedprop
                lab_est.set_prediction_video(vidname=vidname, videodata=curr_videodata)
                
                if RENDER_RESULTS_PAPER is True:
                    PaperUtils.save_imgs_with_label_overlay(RENDER_RESULTS_PAPER_PATH, vidname, fname_prefix='gt_', \
                                label_ims=curr_videodata.get_data('annot_im'), bgr_ims=curr_videodata.get_data('bgr_im'), \
                                bgr_saturation=.5, label_alpha=0.7, render_scale=1.)

            # on new sequence (same or new video, restarted labeling session)
            if is_new_sequence:
                prev_scribble_dict = None
                curr_scribble_dict = None
                annotated_fr_idxs = []
                if pred_metric_mean_j_im is not None:
                    print("    -> metrics for previous sequence:", pred_metric_mean_j_im)
                pred_metric_mean_j_im = []
                davis_state_prev_preds = None
                davis_state_seed_hist = []
                davis_state_seed_prop_hist = []
                seq_idx += 1
                step_idx = 0
                prev_pred_label_im = None

            # extract new scribbles
            prev_scribble_dict = curr_scribble_dict
            curr_scribble_dict = scribble
            curr_annot_fr_idx, new_scribbles = DavisUtils.get_new_scribbles(vidname, prev_scribble_dict, curr_scribble_dict, (480, 854))
            annotated_fr_idxs.append(curr_annot_fr_idx)
            print("      fr idx: ", curr_annot_fr_idx)

            if RENDER_RESULTS_PAPER is True:
                PaperUtils.save_img_with_scribble_overlay(RENDER_RESULTS_PAPER_PATH, vidname, \
                            fname_prefix='scrib' + str(seq_idx) + '_' + str(step_idx+1) + '_', \
                            scribbles=(curr_annot_fr_idx, new_scribbles), bgr_ims=curr_videodata.get_data('bgr_im'), \
                            bgr_saturation=.5, scribble_width=3, render_scale=1.)

            # convert new scribbles to seeds, use different algorithm if first step in current sequence
            N_SEEDS_PER_CAT_INITIAL = 100
            N_SEEDS_PER_CAT_LATER = 100

            n_seeds_per_cat = N_SEEDS_PER_CAT_INITIAL if is_new_sequence else N_SEEDS_PER_CAT_LATER
            curr_seed_points = DavisUtils.davis_scribbles2seeds_uniform(new_scribbles, (480, 854), n_seeds_per_cat, \
                                                                       generate_bg=is_new_sequence)  # (n_seeds, 3:[y, x, lab])
            # submit seeds to label estimation, generate predictions
            assert curr_seed_points.shape[1:] == (3,)
            fr_idxs = np.full(curr_seed_points.shape[0], dtype=np.int32, fill_value=curr_annot_fr_idx)
            coords, labels = curr_seed_points[:,:2], curr_seed_points[:,2]

            lab_est.reset_prediction_generator()
            lab_est.set_prediction_davis_state(curr_annot_fr_idx, new_scribbles, \
                                            davis_state_prev_preds, davis_state_seed_hist, davis_state_seed_prop_hist)
            davis_state_prev_preds = lab_est.predict_all(return_probs=True)
            davis_state_prev_preds_am = np.argmax(davis_state_prev_preds, axis=-1)

            _, davis_state_seed_hist, davis_state_seed_prop_hist = lab_est.get_prediction_davis_state()
            seg_im = curr_videodata.get_seg().get_seg_im(framewise_seg_ids=False)
            pred_label_im = davis_state_prev_preds_am[seg_im]

            # get metrics
            n_labels = curr_videodata.get_data('n_labels')
            true_lab_im = curr_videodata.get_data('annot_im')
            pred_metric_mean_j_im.append(ImUtil.compute_pixelwise_labeling_error(true_lab_im, pred_label_im, n_labels))

            # produce frame query list, TODO can remove from paper relase
            assert DAVIS_frame_query_method in ['default', 'equidistant', 'choose_from_distant']
            if DAVIS_frame_query_method == 'default':
                frames_to_query = None
            elif DAVIS_frame_query_method == 'equidistant':
                query_ratios = [0.5, 0.25, 0.75, 0.125, 0.375, 0.625, 0.875, 0.]
                next_fr = int(query_ratios[len(annotated_fr_idxs)-1]*curr_videodata.get_seg().get_n_frames())
                frames_to_query = [min(max(0, next_fr), curr_videodata.get_seg().get_n_frames()-1)]
            elif DAVIS_frame_query_method == 'choose_from_distant':
                frame_dists = np.ones((curr_videodata.get_seg().get_n_frames(),), dtype=np.int32)
                frame_dists[annotated_fr_idxs] = 0
                frame_dists = distance_transform_cdt(frame_dists)
                frames_to_query = list(np.argsort(frame_dists)[(3*frame_dists.shape[0])//4:])
                frames_to_query = frames_to_query
                
            # submit predicted masks to DAVIS benchmark, resize predictions if necessary
            if RENDER_RESULTS_PAPER is True:
                PaperUtils.save_imgs_with_label_overlay(RENDER_RESULTS_PAPER_PATH, vidname, \
                                fname_prefix='pred' + str(seq_idx) + '_' + str(step_idx+1) + '_', \
                                label_ims=pred_label_im, bgr_ims=curr_videodata.get_data('bgr_im'), \
                                bgr_saturation=.5, label_alpha=0.7, render_scale=1.)

            prev_pred_label_im = pred_label_im.copy()
            if curr_vidname in DAVIS17.CUSTOM_IMSIZE_DICT.keys():
                pred_label_im = ImUtil.fast_resize_video_nearest_singlech(pred_label_im, DAVIS17.CUSTOM_IMSIZE_DICT[curr_vidname])

            print("---> DAVIS working...")
            DavisUtils.fix_label_image_error(vidname, pred_label_im)
            sess.submit_masks(pred_label_im, next_scribble_frame_candidates=frames_to_query)

        # Get the global summary
        report_save_fpath = os.path.join(DAVIS_report_dir, 'summary.json')
        summary = sess.get_global_summary(save_file=report_save_fpath)
