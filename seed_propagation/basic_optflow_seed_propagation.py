
# 
# GNN_annot IJCNN 2021 implementation
#   Non-parametric optical flow based seed propagation method. 
#       Seed points are propagated in time as long as fwd and bwd optical flow transformations of seeds fall close to each other.
#   @author Viktor Varga
#

import numpy as np

MAX_PROP_LEN = 200   # maximum number of frames to propagate seeds through

# one of the two conditions below must hold
MAX_FWD_BWD_INCONSISTENCY_RATIO = 0.1     # upper limit for |vec_fwd|/|vec_delta|
MAX_FWD_BWD_INCONSISTENCY_PIXELS = 2.     # upper limit for |vec_delta|
#
OPTFLOW_LOCAL_MAX_DEVIANCE_W = 0.1
OPTFLOW_LOCAL_MAX_DEVIANCE_B = 1.

POINT_ADJACENCY_RADIUS_PIX = 5

class BasicOptflowSeedPropagation:

    '''
    Member fields:
        videodata_dict: dict{vidname - str: VideoData}
    '''

    def __init__(self, videodata_dict):
        self.videodata_dict = videodata_dict

    def _transform_points_with_flow(self, flow, points):
        """
        Projects points using optical flow..
        Parameters:
            flow: ndarray(size_y, size_x, 2:[dy,dx]) of float32; output of optical flow algorithms
            points: ndarray(n_points, 2:[y,x]) of float32;
        Returns:
            points_proj: ndarray(n_points, 2:[y,x]) of float32; can be NaN if points are not inside the flow
        """
        box_size = np.asarray(flow.shape[:2], dtype=np.int32)  # (2:[y,x])
        valid_point_idxs = ~np.any(np.isnan(points), axis=1)  # (n_points,)
        inside_box_point_idxs = np.all(points >= 0, axis=1) & np.all(points < box_size-1, axis=1)  # (n_points,)
        valid_point_idxs = valid_point_idxs & inside_box_point_idxs

        # rounding of point coordinates only for indexing, the movement from the flow is added to the original float point values
        points_i = np.around(points).astype(np.int32)
        valid_points = points_i[valid_point_idxs, :]  # (n_valid_points,)

        delta_valid_points = flow[valid_points[:, 0], valid_points[:, 1]]  # (n_valid_points, 2:[y,x])
        delta_points = np.full_like(points, fill_value=np.nan)
        delta_points[valid_point_idxs, :] = delta_valid_points
        points_proj = points + delta_points
        return points_proj

    def propagate(self, vidname, fr_idxs, points, labels):
        '''
        Propagation of seed points based on bidirectional optical flow consistency.
        Parameters:
            vidname: str
            fr_idxs: ndarray(n_seed_annots,) of int32
            points: ndarray(n_seed_annots, 2:[py, px]) of int32
            labels: ndarray(n_seed_annots,) of int32
        Returns:
            prop_points: dict{fr_idx - int: ndarray(n_seeds_in_frame, 3:[py, px, label]) of int32}; original points are not stored
        '''
        assert fr_idxs.shape[0] == points.shape[0] == labels.shape[0]
        if points.shape[0] == 0:
            return {}

        start_fr_idx = fr_idxs[0]
        videodata = self.videodata_dict[vidname]
        assert np.all(start_fr_idx == fr_idxs), "TODO Propagating seeds from multiple frames at once is not implemented."
        optflows_fw = videodata.get_data('of_fw_im')[:-1,:,:,:]   # (n_fr-1, sy, sx, 2:[dy, dx])
        optflows_bw = videodata.get_data('of_bw_im')[1:,:,:,:]    # (n_fr-1, sy, sx, 2:[dy, dx])
        imsize = np.array(videodata.get_seg().get_shape()[1:], dtype=np.float32)
        adjacency = np.array([[0, POINT_ADJACENCY_RADIUS_PIX], [POINT_ADJACENCY_RADIUS_PIX, 0],\
                              [0, -POINT_ADJACENCY_RADIUS_PIX], [-POINT_ADJACENCY_RADIUS_PIX, 0]], dtype=np.int32)  # (4, 2)

        prop_points = {}
        # propagating forward
        for direction in ['fw', 'bw']:
            curr_points = points.astype(np.float32, copy=True)
            curr_labels = labels.copy()
            end_fr_idx = videodata.get_seg().get_n_frames()-1 if direction == 'fw' else 0
            step_size = 1 if direction == 'fw' else -1
            for fr_idx in range(start_fr_idx, end_fr_idx, step_size):
                if curr_points.shape[0] == 0:
                    break
                if abs(fr_idx-start_fr_idx)+1 > MAX_PROP_LEN:
                    break
                flow_to = optflows_fw[fr_idx] if direction == 'fw' else optflows_bw[fr_idx-1]
                flow_back = optflows_bw[fr_idx] if direction == 'fw' else optflows_fw[fr_idx-1]
                points_i = np.round(curr_points).astype(np.int32)   # (n_points, 2)

                # get adjacent points, check optflow differences (TODO only 'to' direction is checked)
                points_adj_i = points_i[:,None,:] + adjacency   # (n_points, 4, 2)
                points_adj_i = np.clip(points_adj_i, [0,0], imsize-1).astype(np.int32)
                flowvecs_to = flow_to[points_i[:,0], points_i[:,1]]               # (n_points, 2)
                flowvecs_to_adj = flow_to[points_adj_i[:,:,0], points_adj_i[:,:,1]]   # (n_points, 4, 2)
                flowdiff_to_lens = np.linalg.norm(flowvecs_to[:,None,:] - flowvecs_to_adj, ord=2, axis=-1)  # (n_points, 4)
                flowvec_to_lens = np.linalg.norm(flowvecs_to, ord=2, axis=-1)  # (n_points,)
                cond1_mask = np.amax(flowdiff_to_lens, axis=-1) <= OPTFLOW_LOCAL_MAX_DEVIANCE_W*flowvec_to_lens \
                                                                            + OPTFLOW_LOCAL_MAX_DEVIANCE_B
                curr_points = curr_points[cond1_mask,:]
                curr_labels = curr_labels[cond1_mask]

                # propagate original points
                points_to = self._transform_points_with_flow(flow_to, curr_points)
                points_ret = self._transform_points_with_flow(flow_back, points_to)
                # filter NaN (lost points) and out of bounds points
                valid_points_mask = ~np.any(np.isnan(points_ret), axis=-1)
                valid_points_mask[valid_points_mask] &= np.all(points_ret[valid_points_mask,:] >= 0., axis=-1) & \
                                                        np.all(points_ret[valid_points_mask,:] < imsize-1, axis=-1)
                curr_points = curr_points[valid_points_mask,:]
                curr_labels = curr_labels[valid_points_mask]
                points_to = points_to[valid_points_mask,:]
                points_ret = points_ret[valid_points_mask,:]
                len_to = np.linalg.norm(points_to - curr_points, ord=2, axis=-1)
                len_ret = np.linalg.norm(points_ret - curr_points, ord=2, axis=-1)
                cond2_mask = len_ret < np.maximum(MAX_FWD_BWD_INCONSISTENCY_RATIO*len_to, MAX_FWD_BWD_INCONSISTENCY_PIXELS)
                curr_points = points_to[cond2_mask,:]
                curr_labels = curr_labels[cond2_mask]

                if curr_points.shape[0] > 0:
                    dict_fr_key = fr_idx+1 if direction == 'fw' else fr_idx-1
                    prop_points[dict_fr_key] = np.concatenate([curr_points.astype(np.int32), curr_labels[:,None]], axis=-1)
            #

        return prop_points
