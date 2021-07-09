# GNN_annot

Publicly released implementation of the GNN based Interactive VOS method described in the IJCNN 2021 paper "Fast Interactive Video Object Segmentation with Graph Neural Networks", <https://arxiv.org/abs/2103.03821>

Performance of the method: 0.741 J AUC, 0.749 J @ 60sec on the DAVIS 2017 validation set.

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
- FlowNet 2

The repository contains a fork of the DAVIS Interactive Evaluation Framework repository (<https://github.com/albertomontesg/davis-interactive>) as a submodule. Modifications were added to the package in order to be used to train our model against the DAVIS Interactive benchmark. However, for the purpose of evaluation only, the original davis-interactive package can be used instead.

FlowNet 2 (<https://github.com/lmb-freiburg/flownet2>) was used to generate optical flow predictions. The script for running FlowNet 2 is currently not integrated into the repository (TODO). We provide the FlowNet 2 results for the DAVIS dataset as .h5 data files instead, temporarily. The format of the data files and the way to generate them are detailed in the "Notes" section, below.

## Installation

First, clone this repository and install the prerequisite Python packages listed above.

Then, download the DAVIS 2017 dataset: <https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip> and unpack it. Specify the path to the unpacked contents by setting ```DAVIS_ROOT_FOLDER``` in ```datasets/DAVIS17.py```.

Our method relies on optical flow estimations (FlowNet 2 by default). Currently the FlowNet runner script is not integrated into the repository. You can either install it and generate optical flow estimations yourself (see format details in the "Notes" section) or download the DAVIS 2017 training and validation set FlowNet2 results from here: <http://gofile.me/5vro8/LkWwkMUl4>.
See ```DATA_FOLDER_OPTFLOWS``` constant in ```datasets/DAVIS17.py```: the downloaded or generated .h5 files must be placed to this location.

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

By default the first 45 videos (in alphabetical order) of the DAVIS 2017 training set are used for training and the remaining 15 videos are used for validation. Training a model takes approx. 36 hours on a single GTX 1080 Ti graphics card. Multiple graphics cards are not supported for training currently. For GPUs with a smaller memory, the number of GNN layers (```config.py```) or the graph size (```preprocessing/create_segmentation.py```) might need to be reduced. Keeping the 60 videos of the DAVIS training set in memory at once during the training requires approx. 40 gigabytes of memory. The preprocessed data and the cached graph node and edge features take up about 95 gigabytes of disk space for the DAVIS 2017 training and validation set videos.

Now, you can evaluate the trained model with the DAVIS Interactive benchmark by specifying the model path with ```GNN_CHECKPOINT_DIR``` and ```GNN_CHECKPOINT_NAME_TO_LOAD``` in ```config.py``` and running ```run_davis_interactive_eval.py```:

```
python run_davis_interactive_eval.py
```

Running the benchmark generates the default ```summary.json``` file which summarizes the performance of the method.

Additionally, an interactive, TkInter GUI based tool is available (```run_davis_interactive_debug.py```) to visualize the input node features and binary/multiclass predictions in each step of the interactive benchmark sessions. The tool is not completely bug-free and was only used for debugging purposes.

## Notes

The FlowNet 2 result data files were generated from the outputs of FlowNet 2 software. The data files can be downloaded from our server. However, we give some details about how to generate them. The .h5 data files contain forward and reversed optical flow and occlusion estimations for a single DAVIS video. Occlusion is derived from the optical flow estimations.
Each data file must contain the following keys and datasets:

- ```'flows'```: shape (n_frames-1, 480, 854, 2), dtype float32 ("\<f4")> - The forward optical flow estimations
- ```'inv_flows'```: shape (n_frames-1, 480, 854, 2), dtype float32 ("\<f4")> - The reversed optical flow estimations
- ```'occls'```: shape (n_frames-1, 480, 854), dtype bool ("|b1")> - The occlusion estimations
- ```'inv_occls'```: shape (n_frames-1, 480, 854), dtype bool ("|b1")> - The reversed occlusion estimations

All videos are resized to 854 x 480 resolution before running FlowNet. The last axis of the ```'flows'``` and ```'inv_flows'``` datasets index coordinates in the y, x order. Occlusion and reversed occlusion of a pixel in a given frame is True, if no optical flow vectors point to that pixel from the adjacent frames in forward and reversed optical flow estimations respectively.

While FlowNet 2 is several years old, we still consider it a state-of-the-art optical flow estimator. Other optical flow software can also be used with our method, but keep in mind that specific optical flow models were trained with specific types of data. For example, optical flow models trained with car dashboard recordings will underperform on the DAVIS dataset.

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
