"""
Contains methods for MRI processing.
"""

import os
import numpy as np
import cv2
import scipy.io as sio
import matplotlib.pyplot as plt
import h5py
import pandas as pd
from PIL import Image
from toimage import toimage
from random import shuffle
from tqdm import tqdm
import math
from pathlib import Path
from sklearn.metrics import confusion_matrix
# import for python types convention
from fastai.basic_train import Learner, load_learner
from fastai.vision.data import ImageDataBunch, ImageList
from fastai.vision.transform import get_transforms
from fastai.vision.data import imagenet_stats
from typing import List, Dict, Sequence, Tuple
from my_decorators import timeit
import time


def loadmat(mat_file_path):
    """
    Loads mat file  using scipy.io for versions < 7.3 and using h5py for version 7.3
    :mat_file_path: Posix path or string
    :return: dictionary
    """
    try:
        mat = sio.loadmat(mat_file_path)
    except NotImplementedError:     # in case if this is v7.3 file
        hf = h5py.File(mat_file_path, 'r')
        mat = hf["data"]
    except ValueError as vl_error:
        print(vl_error)
    except FileNotFoundError as fnf_error:
        print(fnf_error)
    return mat


def extract_3d_arrays(mat_file_path):
    """
    Extract 3D images from MRI mat file.
    :return: (image_dicom_3d, mask_contr_3d, mask_total_3d)
    """
    mat = loadmat(mat_file_path)
    # returns as numpy arrays - not lazy readers
    return mat["ImagenesDicom"][()], mat["MascContr"][()], mat["MascTotal"][()]


def calculate_mask(mask_contr, mask_total):
    """
    Creates mask for tissue type 3-class segmentation: 0 - healthy tissue, 1 - cancer, 2 - necrosis.
    :mask_contr: 1-s on tumor areas, 0-s everywhere else
    :mask_total: 1-s on the area where tumor or necrosis are
    :return: mask with 0, 1, 2 segmentation classes.
    """
    mask_contr = (mask_contr > 0).astype(np.uint8)
    mask_total = (mask_total > 0).astype(np.uint8)
    mask_res = 2 * mask_total - mask_contr
    return mask_res


def read_slice(mat_file_path, slice_idx):
    """
    Checks data integrity and returns dicom image with masks
    :param mat_file: MRI in MoLAB's format '.mat' file
    :param slice_idx: index of slice we want to read
    :return: (image_dicom, mask_contr, mask_total)
    """
    # load mat file
    mat = loadmat(mat_file_path)

    # verify data correctness
    if slice_idx < 0 or slice_idx >= mat["ImagenesDicom"].shape[2]:
        raise ValueError("Slice index have to be in range: [0, {}]".format(mat["ImagenesDicom"].shape[2] - 1))
    if mat["ImagenesDicom"].shape != mat["MascContr"].shape or mat["MascContr"].shape != mat["MascTotal"].shape:
        raise ValueError("Dimensions of image and masks must be equal")
    
    # extract 2D arrays to visualize
    image_dicom = mat["ImagenesDicom"][:, :, slice_idx]
    mask_contr = mat["MascContr"][:, :, slice_idx]
    mask_total = mat["MascTotal"][:, :, slice_idx]

    return image_dicom, mask_contr, mask_total


def show_slice_and_masks_mat(mat_file_path, slice_idx):
    """
    Loads '.m' file and shows dicom file and corresponding segmentation masks.
    """
    image_dicom, mask_contr, mask_total = read_slice(mat_file_path, slice_idx)

    # plotting settings
    imsize = 10
    fig = plt.figure(figsize=(imsize, 3*imsize))
    # plot dicom image
    ax1 = fig.add_subplot(311)
    ax1.set_title('DICOM image')
    plt.imshow(image_dicom, cmap=plt.cm.get_cmap('Greys'))
    ax1.set_aspect('equal')
    plt.colorbar(orientation='vertical')
    # plot contrast mask
    ax2 = fig.add_subplot(312)
    ax2.set_title('Contrast mask')
    plt.imshow(mask_contr, cmap=plt.cm.get_cmap('Greys'))
    ax2.set_aspect('equal')
    plt.colorbar(orientation='vertical')
    # plot total mask
    ax2 = fig.add_subplot(313)
    ax2.set_title('Total mask')
    plt.imshow(mask_total, cmap=plt.cm.get_cmap('Greys'))
    ax2.set_aspect('equal')
    plt.colorbar(orientation='vertical')
    fig.show()


def convert_to_pathes(*args):
    """
    Converts non-None parameters to `PosixPathe` objects.
    """
    convert_one = lambda path: Path(path) if path else None
    return tuple(map(convert_one, args))


def save_slice(path_mat, idx, image_dicom, path_to, x_min=0, x_max=None, y_min=0, y_max=None):
    """
    Saves rectangular subpart from `idx` slice of the `path_mat` file from array `image_dicom` to png file at `path_to`.
    (`x_min`, `y_min`) - lower left corner of the rectangular from slice to save
    ('x_max', `y_max`) - upper right corner of the rectangular from slice to save
    """
    if x_max is None:
        x_max = image_dicom.shape[1]
    elif x_max < image_dicom.shape[1]:
        # increase not to loose right boundary during the saving
        x_max += 1
    if y_max is None:
        y_max = image_dicom.shape[2]
    elif y_max < image_dicom.shape[2]:
        y_max += 1
    fname = "{}_{:0>3}.png".format(path_mat.stem, idx)
    im = toimage(image_dicom[idx, x_min:x_max, y_min:y_max])
    im.save(path_to / fname)


def mat2png(path_mat, path_to, path_healthy=None, path_masks=None, tumor_pixels_threshold=None):
    """
    Converts `mat` file to set of `png` images
    :path_mat: str or PosixPath to mat file
    :path_to: str or PosixPath to folder where to save png files
    :path_healthy: str or PosixPath to dir where to solve slices without tumor. `None` value indicate that we will not save healthy slices.
    """
    path_mat, path_to, path_healthy, path_masks = convert_to_pathes(path_mat, path_to, path_healthy, path_masks)
    image_dicom, mask_contr, mask_total = extract_3d_arrays(path_mat)
    mask = calculate_mask(mask_contr, mask_total)
    
    # find layers with non-zero mask
    # direction 0
    for idx in range(mask.shape[0]):
        if np.sum(mask[idx, :, :]) > 0:
            # check if we count number of ill pixels
            if tumor_pixels_threshold is None:
                # flag if we have count number of pixels
                enough_pixels = True
            elif np.sum(mask[idx, :, :] > 0) > tumor_pixels_threshold:
                enough_pixels = True
            else:
                enough_pixels = False
            if enough_pixels:
                # save this slice to appropriate folder
#                 save_slice(path_mat, idx, image_dicom, path_to)
                fname = "{}_0_{:0>3}.png".format(path_mat.stem, idx)
                im = toimage(image_dicom[idx, :, :])
                im.save(path_to / fname)
                # save corresponding mask to appropriate folder if we need them
                if path_masks:
                    save_slice(path_mat, idx, mask, path_masks)
        elif path_healthy:    # if slice is healthy and we want to keep healthy slices
            save_slice(path_mat, idx, image_dicom, path_healthy)
    
    # direction 1
    for idx in range(mask.shape[1]):
        if np.sum(mask[:, idx, :]) > 0:
            # check if we count number of ill pixels
            if tumor_pixels_threshold is None:
                # flag if we have count number of pixels
                enough_pixels = True
            elif np.sum(mask[:, idx, :] > 0) > tumor_pixels_threshold:
                enough_pixels = True
            else:
                enough_pixels = False
            if enough_pixels:
                # save this slice to appropriate folder
#                 save_slice(path_mat, idx, image_dicom, path_to)
                fname = "{}_1_{:0>3}.png".format(path_mat.stem, idx)
                im = toimage(image_dicom[:, idx, :])
                im.save(path_to / fname)
                # save corresponding mask to appropriate folder if we need them
                if path_masks:
                    save_slice(path_mat, idx, mask, path_masks)
        elif path_healthy:    # if slice is healthy and we want to keep healthy slices
            save_slice(path_mat, idx, image_dicom, path_healthy)
    
    # direction 2
    for idx in range(mask.shape[2]):
        if np.sum(mask[:, :, idx]) > 0:
            # check if we count number of ill pixels
            if tumor_pixels_threshold is None:
                # flag if we have count number of pixels
                enough_pixels = True
            elif np.sum(mask[:, :, idx] > 0) > tumor_pixels_threshold:
                enough_pixels = True
            else:
                enough_pixels = False
            if enough_pixels:
                # save this slice to appropriate folder
#                 save_slice(path_mat, idx, image_dicom, path_to)
                fname = "{}_2_{:0>3}.png".format(path_mat.stem, idx)
                im = toimage(image_dicom[:, :, idx])
                im.save(path_to / fname)
                # save corresponding mask to appropriate folder if we need them
                if path_masks:
                    save_slice(path_mat, idx, mask, path_masks)
        elif path_healthy:    # if slice is healthy and we want to keep healthy slices
            save_slice(path_mat, idx, image_dicom, path_healthy)


def get_patch_boundaries(mask_slice, eps=2):
        """
        Computes coordinates of SINGLE patch on the slice. Behaves incorrectly in the case of multiple tumors on the slice.
        :mask_slice: 2D ndarray, contains mask with <0, 1, 2> values of pixels
        :eps: int, number of additional pixels we extract around the actual mask coordinates
        :return: `x_min`, `x_max`, `y_min`, `ymax`
        """
        # check if we work with mask_slice that contains at least one non-zero pixel
        if np.sum(mask_slice[:, :]) <= 0:
            raise ValueError("Slice does not contains any tumors.")
        # smallest index that has something except in its layer
        x_min = None
        for x in range(mask_slice.shape[0]):
            if np.sum(mask_slice[x, :]) > 0:
                # get first from the left index of nonzero 1D slice and break
                x_min = x
                break
        x_max = None
        for x in range(mask_slice.shape[0] - 1, -1, -1):
            if np.sum(mask_slice[x, :]) > 0:
                # get the first from the right index of nonzero 1D slice and break
                x_max = x
                break
        y_min = None
        for y in range(mask_slice.shape[1]):
            if np.sum(mask_slice[:, y]) > 0:
                # get the first from the bottom index of nonzero 1D slice and break
                y_min = y
                break
        y_max = None
        for y in range(mask_slice.shape[1] - 1, -1, -1):
            if np.sum(mask_slice[:, y]) > 0:
                # get the first from the top index of nonzero 1D slice and break
                y_max = y
                break
        # apply `eps` parameter to the actual `min` and `max` values
        x_min = max(x_min - eps, 0)
        x_max = min(x_max + eps, mask_slice.shape[0] - 1)
        y_min = max(y_min - eps, 0)
        y_max = min(y_max + eps, mask_slice.shape[1] - 1)
        return x_min, x_max, y_min, y_max


def get_patches_bounding_boxes(mask, idx, eps=2):
    """
    Find bounding boxes for all tumors that occurs on the slice.
    :mask: 3D ndarray mask[idx, <x_dimension>, <y_dimension>]
    :eps: int, number of additional pixels we extract around the actual mask coordinates
    :return: Tuple of tuples (`x_min`, `y_min`, `x_max`, `y_max`) - bounding boxes for all patches on the `idx`th slice
    """
    pass
    

def mat2patch(path_mat, path_to):
    """
    Extracts from `mat` files patches with tumor and save to `path_to` folder like set of `.png` files.
    :path_mat: str or PosixPath to `mat` file
    :path_to: str or PosixPath where to save patches
    """
    path_mat, path_to = convert_to_pathes(path_mat, path_to)
    image_dicom, mask_contr, mask_total = extract_3d_arrays(path_mat)
    mask = calculate_mask(mask_contr, mask_total)
    
    # find layers with non-zero mask
    for idx in range(mask.shape[0]):
        if np.sum(mask[idx, :, :]) > 0:
            # compute boundaries using mask
            x_min, x_max, y_min, y_max = get_patch_boundaries(mask, idx)
            # save slice of the image from `image_dicom`
            save_slice(path_mat, idx, image_dicom, path_to, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)


def mat2bounding_box(path_mat, path_to, size=224, tol_threshold=0):
    """
    Extracts from `mat` file bounding box with tumor and save to `path_to` folder like set of `.png` files.
    Performes multiplication at mask, so in the case of the multifocal tumor we would have boundidng box with non-zero values
    only at places where tumor is. Passes the `.mat` file three times - makes slices in each of the `x`, `y`, `z` axis.
    :path_mat: str or PosixPath to `mat` file
    :path_to: str or PosixPath where to save images with bounding boxes.
    :size: int, size to which we would pad our patch
    :tol_threshold: int, number of the non-zero pixels we need to have to include the slice to dataset
    """
    # read necessary information from `.mat` files
    path_mat, path_to = convert_to_pathes(path_mat, path_to)
    image_dicom, mask_contr, mask_total = extract_3d_arrays(path_mat)
    mask = calculate_mask(mask_contr, mask_total)
    
    # build slices of the provided `.mat` file in the three directions
    # direction 0
    # find layers with non-zero mask
    for idx in range(mask.shape[0]):
        # location of <idx> defines direction - 0 here
        if np.sum(mask[idx, :, :]) > tol_threshold:
            # compute boundaries using mask
            mask_slice = mask[idx, :, :]
            x_min, x_max, y_min, y_max = get_patch_boundaries(mask_slice)
            # reset to zero non-tumor tissues in multifocal cases
            patch_img = image_dicom[idx, x_min:(x_max + 1), y_min:(y_max + 1)]
            # patch_mask = (mask[idx, x_min:(x_max + 1), y_min:(y_max + 1)] > 0).astype(int)
            patch_mask = mask[idx, x_min:(x_max+1), y_min:(y_max+1)] > 0
            patch_img *= patch_mask
            # zero padding
            patch_res = np.zeros((size, size))
            if max(patch_img.shape[0], patch_img.shape[1]) > size:
                # patch is bigger than required size
                patch_res = cv2.resize(patch_img, dsize=(size, size), interpolation=cv2.INTER_CUBIC)
            else:
                # not too big image, zero padding is required
                x_init = size//2 - patch_img.shape[0]//2
                y_init = size//2 - patch_img.shape[1]//2
                patch_res[x_init : (x_init + patch_img.shape[0]), y_init : (y_init + patch_img.shape[1])] = patch_img
            # save `patch_res` to the file, `magic` 0 - hint that this is from `0` axis slice
            fname = "{}_0_{:0>3}.png".format(path_mat.stem, idx)
            im = toimage(patch_res)
            im.save(path_to / fname)
    
    # direction 1
    # find layers with non-zero mask
    for idx in range(mask.shape[1]):
        # location of <idx> defines direction - 0 here
        if np.sum(mask[:, idx, :]) > tol_threshold:
            # compute boundaries using mask
            mask_slice = mask[:, idx, :]
            x_min, x_max, y_min, y_max = get_patch_boundaries(mask_slice)
            # reset to zero non-tumor tissues in multifocal cases
            patch_img = image_dicom[x_min:(x_max + 1), idx, y_min:(y_max + 1)]
            # patch_mask = (mask[idx, x_min:(x_max + 1), y_min:(y_max + 1)] > 0).astype(int)
            patch_mask = mask[x_min:(x_max+1), idx, y_min:(y_max+1)] > 0
            patch_img *= patch_mask
            # zero padding
            patch_res = np.zeros((size, size))
            if max(patch_img.shape[0], patch_img.shape[1]) > size:
                # patch is bigger than required size
                patch_res = cv2.resize(patch_img, dsize=(size, size), interpolation=cv2.INTER_CUBIC)
            else:
                # not too big image, zero padding is required
                x_init = size//2 - patch_img.shape[0]//2
                y_init = size//2 - patch_img.shape[1]//2
                patch_res[x_init : (x_init + patch_img.shape[0]), y_init : (y_init + patch_img.shape[1])] = patch_img
            # save `patch_res` to the file, `magic` 1 - hint that this is from `1` axis slice
            fname = "{}_1_{:0>3}.png".format(path_mat.stem, idx)
            im = toimage(patch_res)
            im.save(path_to / fname)
    
    # direction 2
    # find layers with non-zero mask
    for idx in range(mask.shape[2]):
        # location of <idx> defines direction - 0 here
        if np.sum(mask[:, :, idx]) > tol_threshold:
            # compute boundaries using mask
            mask_slice = mask[:, :, idx]
            x_min, x_max, y_min, y_max = get_patch_boundaries(mask_slice)
            # reset to zero non-tumor tissues in multifocal cases
            patch_img = image_dicom[x_min:(x_max + 1), y_min:(y_max + 1), idx]
            # patch_mask = (mask[idx, x_min:(x_max + 1), y_min:(y_max + 1)] > 0).astype(int)
            patch_mask = mask[x_min:(x_max+1), y_min:(y_max+1), idx] > 0
            patch_img *= patch_mask
            # zero padding
            patch_res = np.zeros((size, size))
            if max(patch_img.shape[0], patch_img.shape[1]) > size:
                # patch is bigger than required size
                patch_res = cv2.resize(patch_img, dsize=(size, size), interpolation=cv2.INTER_CUBIC)
            else:
                # not too big image, zero padding is required
                x_init = size//2 - patch_img.shape[0]//2
                y_init = size//2 - patch_img.shape[1]//2
                patch_res[x_init : (x_init + patch_img.shape[0]), y_init : (y_init + patch_img.shape[1])] = patch_img
            # save `patch_res` to the file, `magic` 1 - hint that this is from `1` axis slice
            fname = "{}_2_{:0>3}.png".format(path_mat.stem, idx)
            im = toimage(patch_res)
            im.save(path_to / fname)


def delete_hidden_files_recursively(path):
    """
    Deletes recursively all hiden files from folder `path`.
    """
    for root, _, files in os.walk(path):
        for fname in files:
            if fname[0] == ".":
                os.remove(os.path.join(root, fname))


def get_all_mat_files(path, permutate=True):
    """
    Returns list of pathes to the `.mat` files in `path` folder obtained recursively.
    :path: str or PosixPath to folder with `.mat` files
    :permutate: bool, shows if we need to permutate output list
    """
    mat_pathes = []
    for root, _, files in os.walk(path):
        for f in files:
            if f[-3:] == "mat":
                mat_pathes.append(os.path.join(root, f))
    if permutate:
        shuffle(mat_pathes)
    return mat_pathes


def create_camvid_dataset(path_from, path_to, split_train=0.8):
    """
    Reads each `.mat` file in the `path_from` dir and creates segmentation dataset in the `path_to` dir.
    Assumes that `path_from` contains only `.mat` files.
    :path_from: str or PosixPath to folder with `.mat` files
    :path_to: str or PosixPath to folder where to save segmentation dataset
    :split_train: proportion of `train` in whole dataset; proportion of `valid`: (1 - `split_train`)
    """
    # check splitting probability
    if split_train < 0 or split_train > 1:
        raise ValueError("Wrong 'train'/'valid' split proportion, should be in range [0, 1].")
    # convert all inputs to PosixPath format
    path_from, path_to = convert_to_pathes(path_from, path_to)
    # create folders if needed
    for dirname in ["images", "labels"]:
        if not os.path.exists(path_to / dirname):
            os.mkdir(path_to / dirname)
    # convert `mat` files to `png` dataset of slices and masks
    # perutation is needed for further random splitting to "valid"/"test" datasets.
    fnames = get_all_mat_files(path_from, permutate=True)
    for fname in tqdm(fnames):
        mat2png(fname, path_to=(path_to / "images"), path_masks=(path_to / "labels"))
    # create file with segmentation codes: 0 - Healthy, 1 - tumor, 2 - Necrosis
    with open(path_to / "codes.txt", "w") as file:
        file.write("Healthy\nTumor\nNecrosis")
    # create file with filenames for `valid` dataset
    with open(path_to / "valid.txt", "w") as file:
        prefixes_valid = [el.split('/')[-1][:-4] for el in fnames[int(len(fnames) * 0.8):]]
        # split by `.mat` file, not by `.png` slices
        for name_png in os.listdir(path_to / "images"):
            if name_png[:-8] in prefixes_valid:
                file.write(name_png + '\n')
 

# Below are inference related methods
@timeit
def inference_dict(path_to_model, progress_output=True):
    """
    Makes inference on `test` images from `path_to_model/test` subfolder.
    """
    data = (ImageList.from_folder(path_to_model)
        .split_by_folder()
        .label_from_folder()
        .add_test_folder('test')
        .transform(get_transforms(), size=224)
        .databunch()
        .normalize(imagenet_stats))
    # load model from `export.pkl` file
    learn = load_learner(path_to_model)
    # inference on all test images
    res_dict = dict()
    for idx in range(len(data.test_ds)):
        img = data.test_ds[idx][0]
        start_time = time.time()
        label, _, probs = learn.predict(img)
        elapsed_time = time.time() - start_time
        label = str(label)
        fname = data.test_dl.dataset.items[idx].stem
        # create dictionary value (future dataframe row)
        row = [label]
        row.extend([float(p) for p in probs])
        row.extend([elapsed_time])
        res_dict[fname] = row
        if progress_output:
            print("'{}' --> '{:>17}' class with probabilities [{:04.2f}, {:04.2f}, {:04.2f}] inference time: {:04.3} seconds".
                  format(fname, label, probs[0], probs[1], probs[2], elapsed_time))
    # creating columns names for pretty outputs
    prob_names = data.classes
    prob_names = ["p_" + el for el in prob_names]
    columns = ['label']
    columns.extend(prob_names)
    columns.extend(['time'])
    df = pd.DataFrame.from_dict(res_dict, orient='index', columns=columns)
    return df


def agregate_per_patient(df):
    """
    Summarize results of slices classification by patient.
    :df: DataFrame with results of slice predictions
    """
    # get names of all processed files from data frame
    fnames_set = set()
    for name in df.index.values:
        fnames_set.add(name[:-6])
    # extract names of columns with classes probabilities `p_<class_name>`
    classes = []
    take_class_name = lambda col: col[:2] == "p_"
    for col in df.columns.values:
        if take_class_name(col):
            classes.append(col)
    classes.sort()
    # function for calculation probability vector for given fname
    def get_probs_for_prefix(prefix, df, columns):
        """
        Returns tuple of mean values of columns `columns` from dataframe `df` for rows that starts from `prefix` substring.
        """
        take_prefix = lambda index: index[0:len(prefix)] == prefix
        series = df.loc[[take_prefix(el) for el in df.index.values], columns].mean()
        tmp_list = list(series)
        #tmp_list.append(df.index.values[np.argmax(tmp_list)])
        tmp_list.append(classes[np.argmax(tmp_list)][2:])
        return tuple(tmp_list)
    # create dictionary for further DataFrame creation
    res_dict = dict()
    # iterate over all file names we want to agregate predictions
    for name in fnames_set:
        res_dict[name] = get_probs_for_prefix(name, df, classes)
    # create DataFrame from dictionary
    columns = [el[2:] for el in classes]
    columns.append('label')
    res_df = pd.DataFrame.from_dict(res_dict, orient='index', columns=columns)
    return res_df


def plot_confusion_matrix_for_patients(data:ImageDataBunch, learn:Learner, normalize:bool=True) -> None:
    """
    Plot confusion matrix per patient.
    """
    # make inference on `data.valid_ds` and build dictionary:  <fname>:(<actual_label>, <predicted_label>)
    # which contains predictions per slice
    res_dict = dict()
    for idx in range(len(data.valid_ds)):
        img = data.valid_ds[idx][0]
        label_actual = str(data.valid_ds[idx][1])
        label_predict, _, _ = learn.predict(img)
        label_predict = str(label_predict)
        fname = data.valid_dl.dataset.items[idx].stem
        res_dict[fname] = label_actual, label_predict
    # calculate dictionary with dictionary of prediction counters per patient
    nested_patient = dict()
    for key in res_dict:
        label_predict = res_dict[key][1]
        patient_id = key[:-6]
        # if there is no patient with such `patient_id` - create it, othervise - create it
        if patient_id in nested_patient.keys():
            # update current walue, if we have this label for this patient, othervise create it
            if label_predict in nested_patient[patient_id]["predictions"].keys():
                nested_patient[patient_id]["predictions"][label_predict] += 1
            else: # create this label
                nested_patient[patient_id]["predictions"][label_predict] = 1
        else:
            label_actual = res_dict[key][0]
            nested_patient[patient_id] = {"predictions": {label_predict : 1}, "actual": label_actual}
    # modify `nested_patient` dictionary to left prediction with the maximum counter
    for patient_id in nested_patient:
        nested_patient[patient_id]["predictions"] = max(nested_patient[patient_id]["predictions"], key=nested_patient[patient_id]["predictions"].get)
    # create list with labels: predicted and actual
    y_true = []
    y_pred = []
    for patient_id in nested_patient:
        y_true.append(nested_patient[patient_id]["actual"])
        y_pred.append(nested_patient[patient_id]["predictions"])
    
    # make plotting
    classes = learn.data.classes
    # compute confusion matrix(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            # ... and label them with the respective list entries
            xticklabels=classes, yticklabels=classes,
            title="Per patient confusion matrix",
            ylabel='Actual',
            xlabel='Predicted')
    ax.set_ylim(len(classes)- .5, -.5)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
