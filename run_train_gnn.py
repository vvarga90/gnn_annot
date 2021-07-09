
# 
# GNN_annot IJCNN 2021 implementation
#   Runner script to train a GNN model.
#   @author Viktor Varga
#

import sys
sys.path.append('./davis-interactive-gnn-annot-training')

from cache_manager import CacheManager
import config as Config

from seed_propagation.basic_optflow_seed_propagation import BasicOptflowSeedPropagation
from label_estimation.logreg_label_model import LogRegLabelModel
from label_estimation.gnn_label_estimation import GNNLabelEstimation
from label_estimation.graphs.graph_datagen_davis_bin import GraphTrainingDatasetDAVIS

if __name__ == '__main__':

    #SET_NAME = 'debug_train (3+1+1)'
    #SET_NAME = 'trainval_reduced (30+10+1)'
    SET_NAME = 'full_trainval (45+15+1)'

    print("CONFIG --->")
    print({var: Config.__dict__[var] for var in dir(Config) if not var.startswith("__")})
    print("<--- CONFIG")

    # TODO need to load 'test_vidnames' videos?

    cache_manager = CacheManager()
    DatasetClass = CacheManager.get_dataset_class()
    train_vidnames = DatasetClass.get_video_set_vidnames(SET_NAME, 'train')
    val_vidnames = DatasetClass.get_video_set_vidnames(SET_NAME, 'val')
    test_vidnames = DatasetClass.get_video_set_vidnames(SET_NAME, 'test')
    all_vidnames = set(train_vidnames + val_vidnames + test_vidnames)
    cache_manager.load_videos(all_vidnames)

    videodata_dict = {vidname: cache_manager.get_videodata_obj(vidname) for vidname in all_vidnames}
    max_n_labels = max([videodata_dict[vidname].get_data('n_labels') for vidname in all_vidnames])

    print("Loaded videos & data:", SET_NAME)

    # INIT LabelModel & SeedProp
    lab_model = LogRegLabelModel()
    seedprop_alg = BasicOptflowSeedPropagation(videodata_dict) if Config.USE_SEEDPROP is True else None

    # INIT & TRAIN LabelEstimation
    
    lab_est = GNNLabelEstimation(label_model=lab_model, seedprop_alg=seedprop_alg)

    tr_videodata_dict = {vidname: videodata_dict[vidname] for vidname in train_vidnames}
    val_videodata_dict = {vidname: videodata_dict[vidname] for vidname in val_vidnames}
    lab_model_init_fn = lambda: LogRegLabelModel()

    N_PARALLEL_DAVIS_TR_SESSIONS = 10
    tr_iter = GraphTrainingDatasetDAVIS(videodata_dict=tr_videodata_dict, lab_model_init_fn=lab_model_init_fn, \
                    n_parallel_sessions=N_PARALLEL_DAVIS_TR_SESSIONS, davis_subset='train0', seedprop_alg=seedprop_alg)
    val_iter = GraphTrainingDatasetDAVIS(videodata_dict=val_videodata_dict, lab_model_init_fn=lab_model_init_fn, \
                    n_parallel_sessions=N_PARALLEL_DAVIS_TR_SESSIONS, davis_subset='train1', seedprop_alg=seedprop_alg)
    
    lab_est.pretrain_gnn(tr_iter, val_iter)

