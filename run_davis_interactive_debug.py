
#
# GNN_annot IJCNN 2021 implementation
#   Runner script for running the debugging UI for the DAVIS interactive benchmark. 
#   @author Viktor Varga
#

import sys
sys.path.append('./davis-interactive-gnn-annot-training')

import os
import numpy as np
import tkinter as tk

from cache_manager import CacheManager
import config as Config
from datasets import DAVIS17
import util.davis_utils as DavisUtils
import util.imutil as ImUtil
import util.featuregen as FeatureGen

from seed_propagation.basic_optflow_seed_propagation import BasicOptflowSeedPropagation
from label_estimation.logreg_label_model import LogRegLabelModel
from label_estimation.gnn_label_estimation import GNNLabelEstimation
from davisinteractive.session import DavisInteractiveSession    # install from pip, see https://interactive.davischallenge.org
from interactive.davis_debug_gui import DAVISDebugGUI

if __name__ == '__main__':


    #SET_NAME = 'debug_test (1+1+3)'
    SET_NAME = 'test_only (1+1+30)'
    SHUFFLE_BENCHMARK_VIDEOS = True

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

    DAVIS_report_dir = '/home/vavsaai/git/gnn_annot_ijcnn21/temp_results/davis_interactive_reports/'
    os.makedirs(DAVIS_report_dir, exist_ok=True)
    
    # Configuration used in the challenges
    DAVIS_max_nb_interactions = 8 # Maximum number of interactions 
    DAVIS_max_time_per_interaction = 30 # Maximum time per interaction per object (in seconds)

    # Metric to optimize
    DAVIS_metric = 'J'
    assert DAVIS_metric in ['J', 'F', 'J_AND_F']

    with DavisInteractiveSession(host='localhost',
                            user_key=None,
                            davis_root=DAVIS17.DAVIS_ROOT_FOLDER,
                            subset='val',
                            shuffle=SHUFFLE_BENCHMARK_VIDEOS,
                            max_time=2**30,
                            max_nb_interactions=DAVIS_max_nb_interactions,
                            metric_to_optimize=DAVIS_metric,
                            #report_save_dir=DAVIS_report_dir) as sess:
                            report_save_dir=None) as sess:

        # Create GUI window
        root_widget = tk.Tk()
        annotator = DAVISDebugGUI(root_widget, videodata_dict, sess, lab_est=lab_est, seed_prop=seedprop_alg)
        root_widget.mainloop()    # takes over control of the main thread

        # Get the global summary
        #report_save_fpath = os.path.join(DAVIS_report_dir, 'summary.json')
        #summary = sess.get_global_summary(save_file=report_save_fpath)
    #

