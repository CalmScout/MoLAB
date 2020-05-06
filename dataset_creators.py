"""
Module with functions for converting dataset of `.mat` files to form usable by fastai.
"""
from pathlib import Path
import typing
from typing import List
import shutil
import pandas as pd
from pandas import DataFrame
import os
import re
from tqdm import tqdm
import numpy as np
from tools import *
import multiprocessing


def get_hospital_from_filename(fpath:Path) -> str:
    """
    Extracts hospital label from filename. Assume filename in one from two file formats.
    Ignores files with extension different from `.mat`.
    """
    fpath = Path(fpath)
    # we process only `.mat` files
    if fpath.suffix != ".mat":
        return None
    fname = fpath.stem
    # detect one of two possible name conventions
    if fname[3] == '_':
        tmp = fname.split("_")[2]
        label = re.search(r"[a-zA-Z]*", tmp).group()
        return label.upper()
    elif fname[3] == '-':
        label = fname.split("-")[1][:2]
        return label.upper()
    else:
        raise ValueError("Unsupported filename format: {}".format(fpath))


def rename_duplicate_fnames(path:Path) -> None:
    """
    Recursively iterate ove all files in `path`, count them and rename all further <fname>.mat
    occurences to <fname>_copy<copy_n>.mat. There may be files with equal names but different masks.
    """
    d_count = dict()
    for root, _, files in os.walk(path):
        for fname in files:
            if fname in d_count:
                name, ext = os.path.splitext(fname)
                new_fname = name + "_copy_{}".format(d_count[fname]) + ext
                shutil.move(os.path.join(root, fname), os.path.join(root, new_fname))
                d_count[fname] += 1
            else:
                d_count[fname] = 1


def _get_labels_from_imagenet_like_folder(path:Path) -> List:
    """
    Returns names of folders in the folder.
    """
    result = []
    for el in os.listdir(path):
        # if doesn't have extension - folder
        if len(el.split('.')) == 1:
            result.append(el)
    return result


def _create_df_from_folder(path:Path) -> DataFrame:
    """
    Creates dataframe with information about each `.mat` file in the `path` tree.
    """
    d_hospitals = dict()
    labels = _get_labels_from_imagenet_like_folder(path)
    for label in labels:
        for root, _, files in os.walk(path / label):
            # create necessary dictionaries
            for fname in files:
                if Path(fname).suffix == ".mat":
                    key = os.path.splitext(fname)[0]
                    d_hospitals[key] = (get_hospital_from_filename(fname), label, os.path.join(root, fname))
    df = pd.DataFrame.from_dict(d_hospitals, orient='index', columns=['hospital', 'label', 'path'])
    return df


def _split_df_to_train_valid(df:DataFrame, pct:float = 0.8, path_save_csv:Path=None) -> DataFrame:
    """
    Generates DataFrame with new `data_split` column with values from {"train", "valid"}.
    From each hospital for each category randomly assign `pct` labels of "train", the rest are "valid".
    Saves dataframe to `.csv` at `path_save`.
    """
    result = df.copy(deep=True)
    result["data_split"] = [None] * result.shape[0]
    for hospital in result.hospital.unique():
        # split by hospital
        tmp = result[result["hospital"] == hospital]
        for label in tmp.label.unique():
            # split by label in the hospital
            tmp_per_label = tmp[tmp['label'] == label]
            # index till which to split to `train`
            index = np.random.permutation(tmp_per_label.index)
            for idx_el in index[:int(pct * len(index))]:
                # modify the whole dataframe
                result.loc[[idx_el], ['data_split']] = 'train'
            for idx_el in index[int(pct * len(index)):]:
                result.loc[[idx_el], ['data_split']] = 'valid'
    if path_save_csv:
        result.to_csv(path_save_csv)
    return result


def _create_imagenet_folders(path_from:Path, path_to:Path) -> None:
    """
    Creates ImageNet-like folder tree at `path_to` folder, using names of classes at `path_from`.
    train
        |-- class_1
            ...
        |-- class_n
    valid
        |-- class_1
            ...
        |-- class_n
    """
    def mkdir_if_not_exists(path:Path) -> None:
        if os.path.exists(path):
            return None
        os.mkdir(path)
    os.makedirs(path_to)
    data_split_names = ["train", "valid"]
    labels = _get_labels_from_imagenet_like_folder(path_from)
    for data_split in data_split_names:
        mkdir_if_not_exists(path_to / data_split)
        for label in labels:
            mkdir_if_not_exists(path_to / data_split / label)


def _oversample_imagenet_train(path:Path) -> None:
    """
    Balances "train" ImageNet-like dataset by copying random files in undersampled classes.
    """
    labels = _get_labels_from_imagenet_like_folder(path / "train")
    n_max = 0
    for label in labels:
        if len(os.listdir(path / "train" / label)) > n_max:
            n_max = len(os.listdir(path / "train" / label))
    for label in labels:
        n_curr = len(os.listdir(path / "train" / label))
        # if current class is undersampled - perform oversampling
        if  n_curr < n_max:
            n_copy, n_rest = n_max // n_curr, n_max % n_curr
            old_fnames = os.listdir(path / "train" / label)
            for idx_copy in range(n_copy - 1):
                # copy existing all existing files with appendix in file name <copy_<idx_copy+1>>
                for fname in old_fnames:
                    old_name, old_sffx = fname.split('.')
                    new_fname = old_name + "_copy_{}".format(idx_copy + 1) + "." + old_sffx
                    shutil.copy2(path / "train" / label / fname, path / "train" / label / new_fname)
            # choose randomly files and copy
            shuffle(old_fnames)
            for fname in old_fnames[:n_rest]:
                old_name, old_sffx = fname.split('.')
                new_fname = old_name + "_copy_{}".format(n_copy + 1) + "." + old_sffx
                shutil.copy2(path / "train" / label / fname, path / "train" / label / new_fname)
            

def create_imagenet_dataset(path_from:Path, path_to:Path, path_split_csv:Path=None, slice_func:str='mat2bounding_box',
                            pct:float = 0.8, tol_threshold:int=50, oversample_train:bool=False) -> None:
    """
    Creates ImageNet-like dataset.
    path_split_csv - path to earlier generated dataset splitting `.csv`
    """
    if path_split_csv:
        df = pd.read_csv(path_split_csv, index_col=0)
    else:     
        # create basic DataFrame that describes available `.mat` files
        df = _create_df_from_folder(path_from)
        # split rows in dataframe into "train"/"valid"
        df = _split_df_to_train_valid(df, pct=pct, path_save_csv=path_to / "split.csv")
    # create dir tree for new dataset if necessary
    _create_imagenet_folders(path_from, path_to)
    # convert all `.mat` files to set of `.png` files at correct ImageNet dir tree
    for fname in tqdm(df.index):
        s = df.loc[fname]
        fpath_to = path_to / s['data_split'] / s['label']
        if slice_func == "mat2bounding_box":
            mat2bounding_box(s['path'], fpath_to, tol_threshold=tol_threshold)
        elif slice_func == "mat2png":
            mat2png(s["path"], fpath_to, tumor_pixels_threshold=tol_threshold)
        else:
            raise ValueError("Unknown `slice_func` parameter value.")
    # balance classes by oversampling undersampled calsses
    _oversample_imagenet_train(path_to)
