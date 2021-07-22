# GNN_annot

Publicly released implementation of the GNN based Interactive VOS method described in the IJCNN 2021 paper "Fast Interactive Video Object Segmentation with Graph Neural Networks", <https://arxiv.org/abs/2103.03821>

Performance of the method: 0.759 J AUC, 0.767 J @ 60sec, 0.782 J&F AUC, 0.790 J&F @ 60sec on the DAVIS 2017 validation set.

(These results were achieved with GMA (2021) optical flow estimation. For the evaluations published in the paper FlowNet2 (2017) was used. Performance with FlowNet2: 0.741 J AUC, 0.749 J @ 60sec on the DAVIS 2017 validation set.)

## Prerequisites

The code was developed and tested with Python 3.6.9 and the following packages:

- Pytorch 1.5.0
- Deep Graph Library (DGL) 0.4.2
- NumPy 1.18.5
- SciPy 1.4.1
- Scikit-image 0.17.2
- Scikit-learn 0.24.0
- OpenCV 4.2.0
- h5py 2.10
- GMA optical flow estimation by Jiang et al. (<https://github.com/zacjiang/GMA>)

The repository contains a fork of the DAVIS Interactive Evaluation Framework repository (<https://github.com/albertomontesg/davis-interactive>) as a submodule. Modifications were added to the package in order to be used to train our model against the DAVIS Interactive benchmark. However, for the purpose of evaluation only, the original davis-interactive package can be used instead.

## Installation

- Clone this repository and install the prerequisite Python packages listed above. 

- Clone GMA optical flow estimation from the given repository link. While GMA was developed with a more recent version of Pytorch and other packages, the prerequisite package versions listed above are compatible with GMA as well (tested 22jul2021).

- Download the DAVIS 2017 dataset: <https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip> and unpack it.
- Specify the path to the unpacked contents by setting ```DAVIS_ROOT_FOLDER``` in ```datasets/DAVIS17.py```. 
- Specify the path to the GMA ```/core``` folder by setting ```GMA_OPTICAL_FLOW_CORE_FOLDER_PATH``` in ```preprocessing/create_optflow_gma.py```. 
- Specify the path to the GMA pretrained model by setting ```GMA_MODEL_PATH``` in ```preprocessing/create_optflow_gma.py```. We used the ```.../GMA/checkpoints/gma-sintel.pth``` pretrained model for the evaluation.

## Usage

To speed up training and evaluation, a data preprocessing step is needed. First, run ```preprocessing/create_all_data.py``` to generate the superpixel segmentation and extract feature vectors from images:

```
cd gnn_annot
python preprocessing/create_all_data.py
```

Now, you can either train a new model or use the pretrained model from the ```_pretrained_model``` folder.
You can train a new GNN model by running ```run_train_gnn.py```:

```
python run_train_gnn.py
```

By default the first 45 videos (in alphabetical order) of the DAVIS 2017 training set are used for training and the remaining 15 videos are used for validation. Training a model takes approx. 30 hours on a single GTX 1080 Ti graphics card. Multiple graphics cards are not supported for training currently. For GPUs with a smaller memory, the number of GNN layers (```config.py```) or the graph size (```preprocessing/create_segmentation.py```) might need to be reduced. Keeping the 60 videos of the DAVIS training set in memory at once during the training requires approx. 40 gigabytes of memory. The preprocessed data and the cached graph node and edge features take up about 95 gigabytes of disk space for the DAVIS 2017 training and validation set videos.

Now, you can evaluate the trained model with the DAVIS Interactive benchmark by specifying the model path with ```GNN_CHECKPOINT_DIR``` and ```GNN_CHECKPOINT_NAME_TO_LOAD``` in ```config.py``` and running ```run_davis_interactive_eval.py```:

```
python run_davis_interactive_eval.py
```

Running the benchmark generates the default ```summary.json``` file which summarizes the performance of the method.

Additionally, an interactive, TkInter GUI based tool is available (```run_davis_interactive_debug.py```) to visualize the input node features and binary/multiclass predictions in each step of the interactive benchmark sessions. The tool is not completely bug-free and was only used for debugging purposes.

## Citing

If you used this code in your research or found our paper useful, please cite the following paper:

```
@article{varga2021gnnivos,
  title={Fast Interactive Video Object Segmentation with Graph Neural Networks},
  author={Varga, Viktor and L{\H{o}}rincz, Andr{\'a}s},
  journal={arXiv preprint arXiv:2103.03821},
  year={2021}
}
```
