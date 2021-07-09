
# 
# GNN_annot IJCNN 2021 implementation
#   A collection of general utility functions.
#   @author Viktor Varga
#

import numpy as np

def view_multichannel_i32_as_i64(arr, copy=True):
    '''
    Stores a multichannel (2 channels) int32 array in a single channel int64 array. 
    The last axis is merged into a single value.
    Paramters:
        arr: ndarray(..., n_ch=2) of int32
        copy: bool; IF 'copy' is False, a view of 'arr' is returned
    Returns:
        arr_singlech: ndarray(...) of int64
    '''
    assert arr.ndim >= 2
    assert arr.shape[-1] == 2
    assert arr.dtype == np.int32
    arr_singlech = arr.view(dtype=np.int64)
    if copy:
        arr_singlech = arr_singlech.copy()
    return arr_singlech[..., 0]

def view_multichannel_ui32_as_ui64(arr, copy=True):
    '''
    Reinterprets a multichannel (2 channels) uint32 array as a single channel uint64 array. 
    The last axis is merged into a single value.
    Paramters:
        arr: ndarray(..., n_ch=2) of uint32
        copy: bool; IF 'copy' is False, a view of 'arr' is returned
    Returns:
        arr_singlech: ndarray(...) of uint64
    '''
    assert arr.ndim >= 2
    assert arr.shape[-1] == 2
    assert arr.dtype == np.uint32
    arr_singlech = arr.view(dtype=np.uint64)
    if copy:
        arr_singlech = arr_singlech.copy()
    return arr_singlech[..., 0]

def view_multichannel_ui8_to_ui32(arr, copy=True):
    '''
    Stores a multichannel (up to 4 channels) uint8 array in a single channel uint32 array. 
    The last axis is merged into a single value.
    Paramters:
        arr: ndarray(..., n_ch<=4) of uint8
        copy: bool; IF 'arr' is a 4-channel uint8 array and 'copy' is False, a view of 'arr' is returned
    Returns:
        arr_singlech: ndarray(...) of uint32
    '''
    assert arr.ndim >= 2
    assert arr.shape[-1] >= 2
    assert arr.dtype == np.uint8
    n_ch = arr.shape[-1]
    if (n_ch == 4) and (not copy):
        arr_singlech = arr.view(dtype=np.uint32)
    else:
        arr_singlech = np.empty(arr.shape[:-1] + (4,), dtype=np.uint8)   # (..., 4)
        arr_singlech[...,:n_ch] = arr
        arr_singlech[...,n_ch:] = 0
        arr_singlech = arr_singlech.view(dtype=np.uint32)
    return arr_singlech[..., 0]

def restore_multichannel_i32_from_i64(arr_singlech, copy=True):
    '''
    Restores a 2-channel int32 array from a single channel int64 array.
    Paramters:
        arr_singlech: ndarray(...) of int64
        copy: bool; if False, a view of 'arr_singlech' is returned
    Returns:
        arr: ndarray(..., n_ch) of int32
    '''
    assert arr_singlech.dtype == np.int64
    arr = arr_singlech[..., None].view(dtype=np.int32)
    assert arr.shape == arr_singlech.shape + (2,)
    if copy:
        arr = arr.copy()
    return arr

def restore_multichannel_ui32_from_ui64(arr_singlech, copy=True):
    '''
    Restores a 2-channel uint32 array from a single channel uint64 array.
    Paramters:
        arr_singlech: ndarray(...) of uint64
        copy: bool; if False, a view of 'arr_singlech' is returned
    Returns:
        arr: ndarray(..., n_ch) of uint32
    '''
    assert arr_singlech.dtype == np.uint64
    arr = arr_singlech[..., None].view(dtype=np.uint32)
    assert arr.shape == arr_singlech.shape + (2,)
    if copy:
        arr = arr.copy()
    return arr

def restore_multichannel_ui8_from_ui32(arr_singlech, n_ch, copy=True):
    '''
    Restores a multichannel (up to 4 channels) uint8 array from a single channel uint32 array. TODO fix name.
    Paramters:
        arr_singlech: ndarray(...) of uint32
        n_ch: int; number of output channels
        copy: bool; if False, a view of 'arr_singlech' is returned
    Returns:
        arr: ndarray(..., n_ch) of uint8
    '''
    assert 2 <= n_ch <= 4
    assert arr_singlech.dtype == np.uint32
    arr = arr_singlech[..., None].view(dtype=np.uint8)[..., :n_ch]
    if copy:
        arr = arr.copy()
    return arr

def unique_2chan(arr, unsigned_type, return_index=False, return_inverse=False, return_counts=False):
    '''
    Efficient np.unique on 2 channel ui32 or i32 type array.
    Parameters:
        arr: ndarray(?, 2) of ui32/i32
        unsigned_type: bool; if True, expecting ui32 arr dtype, otherwise i32
        return_index, return_inverse, return_counts: bool; same as np.unique()
    Returns:
        unique: ndarray(n_unique, 2) of ui32/i32
        (OPTIONAL) unique_indices, unique_inverse, unique_counts: same as np.unique()
    '''
    assert arr.shape[-1] == (2,)
    if unsigned_type is True:
        assert arr.dtype == np.uint32
        arr_singlech = view_multichannel_ui32_as_ui64(arr)
    else:
        assert arr.dtype == np.int32
        arr_singlech = view_multichannel_i32_as_i64(arr)
    rets = np.unique(arr_singlech, return_index=return_index, return_inverse=return_inverse, return_counts=return_counts)
    u_arr_singlech = rets[0]
    u_arr = restore_multichannel_ui32_from_ui64(u_arr_singlech) if unsigned_type is True \
                                                              else restore_multichannel_i32_from_i64(u_arr_singlech)
    rets = u_arr + rets[1:]
    return rets


def apply_func_with_groupby_manyIDs(values_arr, ids_arr, func, assume_arange_to=None, empty_val=0):
    '''
    Applies 'func' to all values within ID groups independently.
        Can be used instead of 'apply_ufunc_with_groupby()' when accumulation with a specific ufunc is not possible.
        Note, that the required signature of 'func' is different.
        Use this version if there are many IDs, otherwise use apply_func_with_groupby_fewIDs.
    Parameters:
        values_arr: ndarray(?) of ?
        ids_arr: ndarray(<values_arr.shape>) of int
        func: Callable with signature ndarray(n_items) of <T> -> scalar <T>
        assume_arange_to: None or int; if given, assumes ids_arr to be in a subset of range(0,assume_arange_to);
                                            thus the unique operation is skipped.
        empty_val: scalar <type of 'values_arr'>; if 'assume_arange_to' is given, result for non-existent IDs
                                                                    in range are set to 'empty_val'
    Returns:
        results: ndarray(n_IDs) of ?; for each ID in 'ids_arr', returns the result; IDs are sorted
            IF 'assume_arange_to' is given, return array shape is ndarray(assume_arange_to).
    '''
    if assume_arange_to is None:
        u_ids, ids_arr = np.unique(ids_arr, return_inverse=True)
        assume_arange_to = u_ids.shape[0]
    # algorithm: argsorting IDs and reordering ID indices with it, splitting ID indices where there are skips in sorted IDs
    #   a naive solution (with masks for each ID) would have O(values_arr_size * n_IDs) computation cost
    #   this solution has O(values_arr_size*log(values_arr_size) + values_arr_size) computation cost
    values_arr, ids_arr = values_arr.reshape(-1), ids_arr.reshape(-1)
    id_sorter = np.argsort(ids_arr)
    ids_sorted = ids_arr[id_sorter]
    split_idxs = np.where(ids_sorted[1:] != ids_sorted[:-1])[0]+1
    idx_groups = np.split(id_sorter, split_idxs)
    group_ids = ids_sorted[np.pad(split_idxs, (1,0))]
    assert len(idx_groups) == group_ids.shape[0]
    results = np.full((assume_arange_to,), fill_value=empty_val, dtype=values_arr.dtype)
    for group_id_idx in range(group_ids.shape[0]):
        group_id, group_idxs = group_ids[group_id_idx], idx_groups[group_id_idx]
        if group_idxs.shape[0] == 0:
            continue
        vals_in_group = values_arr[group_idxs]
        results[group_id] = func(vals_in_group)
    return results

def get_meanstd_with_groupby_manyIDs(values_arr, ids_arr, assume_arange_to=None, empty_val=0):
    '''
    An adaptation of apply_func_with_groupby_manyIDs() for the computation of mean & std functions.
    Parameters:
        values_arr: ndarray(?) of ?
        ids_arr: ndarray(<values_arr.shape>) of int
        assume_arange_to: None or int; if given, assumes ids_arr to be in a subset of range(0,assume_arange_to);
                                            thus the unique operation is skipped.
        empty_val: scalar <type of 'values_arr'>; if 'assume_arange_to' is given, result for non-existent IDs
                                                                    in range are set to 'empty_val'
    Returns:
        results: ndarray(n_IDs, 2:[mean, std]) of ?; for each ID in 'ids_arr', returns the mean and std; IDs are sorted
            IF 'assume_arange_to' is given, return array shape is ndarray(assume_arange_to).
    '''
    if assume_arange_to is None:
        u_ids, ids_arr = np.unique(ids_arr, return_inverse=True)
        assume_arange_to = u_ids.shape[0]
    # algorithm: argsorting IDs and reordering ID indices with it, splitting ID indices where there are skips in sorted IDs
    #   a naive solution (with masks for each ID) would have O(values_arr_size * n_IDs) computation cost
    #   this solution has O(values_arr_size*log(values_arr_size) + values_arr_size) computation cost
    values_arr, ids_arr = values_arr.reshape(-1), ids_arr.reshape(-1)
    id_sorter = np.argsort(ids_arr)
    ids_sorted = ids_arr[id_sorter]
    split_idxs = np.where(ids_sorted[1:] != ids_sorted[:-1])[0]+1
    idx_groups = np.split(id_sorter, split_idxs)
    group_ids = ids_sorted[np.pad(split_idxs, (1,0))]
    assert len(idx_groups) == group_ids.shape[0]
    results = np.full((assume_arange_to, 2), fill_value=empty_val, dtype=values_arr.dtype)
    for group_id_idx in range(group_ids.shape[0]):
        group_id, group_idxs = group_ids[group_id_idx], idx_groups[group_id_idx]
        if group_idxs.shape[0] == 0:
            continue
        vals_in_group = values_arr[group_idxs]
        #results[group_id] = func(vals_in_group)
        # computing mean & std of vals_in_group
        assert vals_in_group.ndim == 1
        mean_val = np.mean(vals_in_group)
        diff_vals = vals_in_group - mean_val
        std_val = np.sqrt(np.dot(diff_vals, diff_vals)/float(vals_in_group.size))
        results[group_id,:] = (mean_val, std_val)
        #
    return results
    

def random_sample_balanced(labels, n_samples_per_cat, n_cats=None):
    '''
    Randomly sample idxs from 'labels' with balanced labels.
    Parameters:
        labels: ndarray(n_samples,) of int32; labels from 0..n_cats-1, all labels must be present
        n_samples_per_cat: int
        n_cats: None or int; the number of categories
    Returns:
        idxs: ndarray(n_cats, n_samples_per_cat) of int32
    '''
    assert labels.ndim == 1
    if n_cats is None:
        n_cats = np.amax(labels)+1
    idxs = np.empty((n_cats, n_samples_per_cat), dtype=np.int32)
    for cat in range(n_cats):
        cat_idxs = np.where(labels == cat)[0]
        assert cat_idxs.shape[0] > 0
        idxs[cat,:] = np.random.choice(cat_idxs, size=(n_samples_per_cat,))
    return idxs

def rolling_mean(data1d, window_len):
    '''
    Parameters:
        data1d: array-like(t,)
        window_len: int
    Returns:
        rm: array-like(t,)
    '''
    assert window_len >= 1
    pad0 = window_len // 2
    pad1 = window_len - pad0
    data1d = np.pad(data1d, (pad0, pad1), 'edge')
    return np.convolve(data1d, np.ones(shape=(window_len,), dtype=np.float64), 'valid') / window_len
