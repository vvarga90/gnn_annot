
# 
# GNN_annot IJCNN 2021 implementation
#   Class to manage caching (serialization of VideoData objects with precomputed data)
#   @author Viktor Varga
#

import os
import time
import pickle

import datasets.DAVIS17 as Dataset
from video_data import VideoData

class CacheManager():
    
    '''
    Member fields:
        videodata_objs: dict{vidname - str: VideoData instance}

        fully_loaded_vidname: None OR str
    '''

    DATASET_TO_LOAD = 'davis2017'  # 'davis2017'

    if DATASET_TO_LOAD == 'davis2017':
        import datasets.DAVIS17 as Dataset
    else:
        assert False, "Not implemented."

    @staticmethod
    def get_dataset_class():
        return CacheManager.Dataset

    # PUBLIC

    def __init__(self):
        os.makedirs(CacheManager.Dataset.CACHE_FOLDER, exist_ok=True)
        self.videodata_objs = {}
        self.fully_loaded_vidname = None

    def get_videodata_obj(self, vidname):
        return self.videodata_objs[vidname]

    def load_videos(self, vidnames):
        '''
        Loading cached data into the memory for the given videos. To load all data (full feature maps), 
            call load_full_data_for_video() explicitly after loading the cached data for it.
        '''
        vidnames_uncached = [vidname for vidname in vidnames if not os.path.isfile(\
                    os.path.join(CacheManager.Dataset.CACHE_FOLDER, CacheManager.Dataset.DATASET_ID + '_' + vidname + '.pkl'))]
        fpaths_uncached = [os.path.join(CacheManager.Dataset.CACHE_FOLDER, CacheManager.Dataset.DATASET_ID + '_' + vidname + '.pkl') \
                                for vidname in vidnames_uncached]

        # some videos are not cached: cache needs to be created
        if len(vidnames_uncached) > 0:
            print("CacheManager: " + str(len(vidnames_uncached)) + " videos were not found in the cache, creating now...")

            N_PARALLEL_JOBS = 8   # None to disable parallel cache creation
            if N_PARALLEL_JOBS is not None:

                from joblib import Parallel, delayed
                t0 = time.time()
                Parallel(n_jobs=N_PARALLEL_JOBS)(delayed(CacheManager._create_cache_file)(*params) for params in \
                                                                            zip(vidnames_uncached, fpaths_uncached))
                t1 = time.time()
                print("CacheManager: Done creating cache for " + str(len(vidnames_uncached)) + " videos. Total time taken: " \
                                                                        + str(round(t1-t0, 2)) + " seconds.")
            else:
                for vidname, fpath in zip(vidnames_uncached, fpaths_uncached):
                    t0 = time.time()
                    CacheManager._create_cache_file(vidname, fpath)
                    t1 = time.time()
                    print("CacheManager: Video '", vidname, "' cache saved. Time taken: " + str(round(t1-t0, 2)) + " seconds.")

        # load videos from cache
        print("CacheManager: loading " + str(len(vidnames)) + " videos...")
        t0 = time.time()
        for vidname in vidnames:

            fpath = os.path.join(CacheManager.Dataset.CACHE_FOLDER, CacheManager.Dataset.DATASET_ID + '_' + vidname + '.pkl')
            self._load_from_cache(vidname, fpath)
        
        t1 = time.time()
        print("CacheManager: Done loading data from cache. Time taken: " + str(round(t1-t0, 2)) + " seconds.")
        #

    def load_full_data_for_video(self, vidname):
        '''
        Loading full data for the given video (the full feature maps). At a time, only one video can be loaded fully.
        '''
        if self.fully_loaded_vidname is not None:
            print("CacheManager.load_full_data_for_video(): Full data was dropped for sequence", self.fully_loaded_vidname)
            self.unload_full_data()
        self.fully_loaded_vidname = vidname
        self.videodata_objs[vidname].add_data('fmaps_dict', CacheManager.Dataset.get_featuremap_data(vidname))

    def unload_full_data(self):
        '''
        Unload full data for the video self.fully_loaded_vidname. The cached data remains in memory.
        '''
        if self.fully_loaded_vidname is not None:
            self.videodata_objs[self.fully_loaded_vidname].drop_data('fmaps_dict')
            self.fully_loaded_vidname = None

    # PRIVATE

    def _load_from_cache(self, vidname, cache_fpath):
        '''
        Loads preprocessed data from cache.
        '''
        assert vidname in cache_fpath
        assert os.path.isfile(cache_fpath)
        assert vidname not in self.videodata_objs.keys()

        pkl_file = open(cache_fpath, 'rb')
        pkl_dict = pickle.load(pkl_file)
        pkl_file.close()
        pkl_videodata_obj = pkl_dict['videodata_obj']
        self.videodata_objs[vidname] = pkl_videodata_obj

    @staticmethod
    def _create_cache_file(vidname, cache_fpath):
        '''
        Preprocecesses raw data and saves it to a cache file per video.
        '''
        assert vidname in cache_fpath
        assert not os.path.isfile(cache_fpath)
        print("CacheManager: Creating cache for video '", vidname, "'...", )

        imgs_bgr = CacheManager.Dataset.get_img_data(vidname)    # (n_fr, sy, sx, 3) of ui8
        gt_annot = CacheManager.Dataset.get_true_annots(vidname)    # (n_fr, sy, sx) of ui8
        flow_fw, flow_bw, occl_fw, occl_bw = CacheManager.Dataset.get_optflow_occlusion_data(vidname)  
                                # (n_frs, sy, sx, 2:[dy, dx]) of fl16, ..., (n_frs, sy, sx) of bool, ...
        fmaps_dict = CacheManager.Dataset.get_featuremap_data(vidname)     # dict{str: ndarray}, see details in method called
        seg_arr = CacheManager.Dataset.get_segmentation_data(vidname)    # (n_frs, sy, sx) of i32

        videodata_obj = VideoData(vidname=vidname, imgs_bgr=imgs_bgr, gt_annot=gt_annot, flow_fw=flow_fw, flow_bw=flow_bw, \
                                  occl_fw=occl_fw, occl_bw=occl_bw, fmaps_dict=fmaps_dict, seg_arr=seg_arr)
        pkl_file = open(cache_fpath, 'wb')
        pkl_dict = {}
        pkl_dict['videodata_obj'] = videodata_obj
        pkl_data = pickle.dump(pkl_dict, pkl_file)
        pkl_file.close()
        print("CacheManager: Saved cache for video '", vidname, "'.", )

