
# 
# GNN_annot IJCNN 2021 implementation
#   Configuration, constant definitions.
#   @author Viktor Varga
#

USE_SEEDPROP = True

TRAINING_DAVIS_N_INTERACTIONS = 6
TRAINING_DAVIS_SESSION_FRAME_QUERY_POLICY = 'random_uniform'   # None OR in ['random_uniform', 'random_linear_distance_prob']
PREDICTION_DOUBLE_ONE_VS_REST = False

GNN_CHECKPOINT_DIR = './_pretrained_model/'
GNN_CHECKPOINT_NAME_TO_LOAD = 'run130_epoch_200.pkl'

GNN_ENABLE_NODE_FEATURE_SESSION_STEP_IDX = False
GNN_HIDDEN_DIM = 20
GNN_N_LAYERS = 12
GNN_DROPOUT = .1
GNN_INFEAT_DROPOUT = .0
GNN_READOUT = "mean"
GNN_GRAPH_NORM = True
GNN_BATCH_NORM = True
GNN_RESIDUAL = True
GNN_BINARY_LOSS = 'bce'   # in ['bce', 'iou_bin']

GNN_INIT_LR = 5e-3
GNN_WEIGHT_DECAY = .0
GNN_LR_REDUCE_FACTOR = 0.6
GNN_LR_SCHEDULE_PATIENCE = 20
GNN_N_EPOCHS = 241
GNN_TR_EPOCH_SIZE = 32
GNN_VAL_EPOCH_SIZE = 8
GNN_N_EPOCHS_CHECKPOINT_FREQ = 20
GNN_MIN_LR = 1e-5
