import os

import nibabel as nib
import numpy as np
import tables

from .training import load_old_model
from .utils import pickle_load
from .utils.patches import reconstruct_from_patches, get_patch_from_3d_data, compute_patch_indices
from .augment import permute_data, generate_permutation_keys, reverse_permute_data
from unet3d.model import unet_model_3d
from train import config 

# config = dict()
# config["pool_size"] = (2, 2, 2)  # pool size for the max pooling operations
# config["image_shape"] = (144, 144, 144)  # This determines what shape the images will be cropped/resampled to.
# config["patch_shape"] = (32, 32, 32)  # switch to None to train on the whole image
# config["labels"] = (1, 2, 4)  # the label numbers on the input image
# config["n_labels"] = len(config["labels"])
# config["all_modalities"] = ["t1", "t1ce", "flair", "t2"]
# config["training_modalities"] = config["all_modalities"]  # change this if you want to only use some of the modalities
# config["nb_channels"] = len(config["training_modalities"])
# if "patch_shape" in config and config["patch_shape"] is not None:
#     config["input_shape"] = tuple([config["nb_channels"]] + list(config["patch_shape"]))
# else:
#     config["input_shape"] = tuple([config["nb_channels"]] + list(config["image_shape"]))
# config["truth_channel"] = config["nb_channels"]
# config["deconvolution"] = True  # if False, will use upsampling instead of deconvolution

# config["batch_size"] = 6
# config["validation_batch_size"] = 4
# config["n_epochs"] = 8 # cutoff the training after this many epochs
# config["patience"] = 10  # learning rate will be reduced after this many epochs if the validation loss is not improving
# config["early_stop"] = 50  # training will be stopped after this many epochs without the validation loss improving
# config["initial_learning_rate"] = 0.00001
# config["learning_rate_drop"] = 0.5  # factor by which the learning rate will be reduced
# config["validation_split"] = 0.8  # portion of the data that will be used for training
# config["flip"] = False  # augments the data by randomly flipping an axis during
# config["permute"] = True  # data shape must be a cube. Augments the data by permuting in various directions
# config["distort"] = None  # switch to None if you want no distortion
# config["augment"] = config["flip"] or config["distort"]
# config["validation_patch_overlap"] = 0  # if > 0, during training, validation patches will be overlapping
# config["training_patch_start_offset"] = (16, 16, 16)  # randomly offset the first patch index by up to this offset
# config["skip_blank"] = True  # if True, then patches without any target will be skipped

# config["data_file"] = os.path.abspath("brats_data.h5")
# config["model_file"] = os.path.abspath("tumor_segmentation_model.h5")
# config["training_file"] = os.path.abspath("training_ids.pkl")
# config["validation_file"] = os.path.abspath("validation_ids.pkl")
# config["overwrite"] = True #False  # If True, will previous files. If False, will use previously written files.


def patch_wise_prediction(model, data, overlap=0, batch_size=1, permute=False):
    """
    :param batch_size:
    :param model:
    :param data:
    :param overlap:
    :return:
    """
    patch_shape = tuple([int(dim) for dim in model.input.shape[-3:]])
    predictions = list()
    indices = compute_patch_indices(data.shape[-3:], patch_size=patch_shape, overlap=overlap)
    batch = list()
    i = 0
    while i < len(indices):
        while len(batch) < batch_size:
            patch = get_patch_from_3d_data(data[0], patch_shape=patch_shape, patch_index=indices[i])
            batch.append(patch)
            i += 1
        prediction = predict(model, np.asarray(batch), permute=permute)
        batch = list()
        for predicted_patch in prediction:
            predictions.append(predicted_patch)
    output_shape = [int(model.output.shape[1])] + list(data.shape[-3:])
    return reconstruct_from_patches(predictions, patch_indices=indices, data_shape=output_shape)


def get_prediction_labels(prediction, threshold=0.5, labels=None):
    n_samples = prediction.shape[0]
    label_arrays = []
    for sample_number in range(n_samples):
        label_data = np.argmax(prediction[sample_number], axis=0) + 1
        label_data[np.max(prediction[sample_number], axis=0) < threshold] = 0
        if labels:
            for value in np.unique(label_data).tolist()[1:]:
                label_data[label_data == value] = labels[value - 1]
        label_arrays.append(np.array(label_data, dtype=np.uint8))
    return label_arrays


def get_test_indices(testing_file):
    return pickle_load(testing_file)


def predict_from_data_file(model, open_data_file, index):
    return model.predict(open_data_file.root.data[index])


def predict_and_get_image(model, data, affine):
    return nib.Nifti1Image(model.predict(data)[0, 0], affine)


def predict_from_data_file_and_get_image(model, open_data_file, index):
    return predict_and_get_image(model, open_data_file.root.data[index], open_data_file.root.affine)


def predict_from_data_file_and_write_image(model, open_data_file, index, out_file):
    image = predict_from_data_file_and_get_image(model, open_data_file, index)
    image.to_filename(out_file)


def prediction_to_image(prediction, affine, label_map=False, threshold=0.5, labels=None):
    if prediction.shape[1] == 1:
        data = prediction[0, 0]
        if label_map:
            label_map_data = np.zeros(prediction[0, 0].shape, np.int8)
            if labels:
                label = labels[0]
            else:
                label = 1
            label_map_data[data > threshold] = label
            data = label_map_data
    elif prediction.shape[1] > 1:
        if label_map:
            label_map_data = get_prediction_labels(prediction, threshold=threshold, labels=labels)
            data = label_map_data[0]
        else:
            return multi_class_prediction(prediction, affine)
    else:
        raise RuntimeError("Invalid prediction array shape: {0}".format(prediction.shape))
    return nib.Nifti1Image(data, affine)


def multi_class_prediction(prediction, affine):
    prediction_images = []
    for i in range(prediction.shape[1]):
        prediction_images.append(nib.Nifti1Image(prediction[0, i], affine))
    return prediction_images


def run_validation_case(data_index, output_dir, model, data_file, training_modalities,
                        output_label_map=False, threshold=0.5, labels=None, overlap=16, permute=False):
    """
    Runs a test case and writes predicted images to file.
    :param data_index: Index from of the list of test cases to get an image prediction from.
    :param output_dir: Where to write prediction images.
    :param output_label_map: If True, will write out a single image with one or more labels. Otherwise outputs
    the (sigmoid) prediction values from the model.
    :param threshold: If output_label_map is set to True, this threshold defines the value above which is 
    considered a positive result and will be assigned a label.  
    :param labels:
    :param training_modalities:
    :param data_file:
    :param model:
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    affine = data_file.root.affine[data_index]
    test_data = np.asarray([data_file.root.data[data_index]])
    for i, modality in enumerate(training_modalities):
        image = nib.Nifti1Image(test_data[0, i], affine)
        image.to_filename(os.path.join(output_dir, "data_{0}.nii.gz".format(modality)))

    test_truth = nib.Nifti1Image(data_file.root.truth[data_index][0], affine)
    test_truth.to_filename(os.path.join(output_dir, "truth.nii.gz"))

    patch_shape = tuple([int(dim) for dim in model.input.shape[-3:]])
    if patch_shape == test_data.shape[-3:]:
        prediction = predict(model, test_data, permute=permute)
    else:
        prediction = patch_wise_prediction(model=model, data=test_data, overlap=overlap, permute=permute)[np.newaxis]
    prediction_image = prediction_to_image(prediction, affine, label_map=output_label_map, threshold=threshold,
                                           labels=labels)
    if isinstance(prediction_image, list):
        for i, image in enumerate(prediction_image):
            image.to_filename(os.path.join(output_dir, "prediction_{0}.nii.gz".format(i + 1)))
    else:
        prediction_image.to_filename(os.path.join(output_dir, "prediction.nii.gz"))


def run_validation_cases(validation_keys_file, model_file, training_modalities, labels, hdf5_file,
                         output_label_map=False, output_dir=".", threshold=0.5, overlap=16, permute=False, isDeeper = False, depth = 4):
    validation_indices = pickle_load(validation_keys_file)

    model = None 

    if not isDeeper:
    #Recreated the unet_model_3d 
        model = unet_model_3d(input_shape=config["input_shape"],
                                  pool_size=config["pool_size"],
                                  n_labels=config["n_labels"],
                                  initial_learning_rate=config["initial_learning_rate"],
                                  deconvolution=config["deconvolution"])#load_old_model(model_file)
    else:
         model = unet_model_3d(input_shape=config["input_shape"],
                                  pool_size=config["pool_size"],
                                  n_labels=config["n_labels"],
                                  initial_learning_rate=config["initial_learning_rate"],
                                  deconvolution=config["deconvolution"], depth = depth)

    #Loading the weights to the model 
    model.load_weights(model_file)
    data_file = tables.open_file(hdf5_file, "r")
    for index in validation_indices:
        if 'subject_ids' in data_file.root:
            case_directory = os.path.join(output_dir, data_file.root.subject_ids[index].decode('utf-8'))
        else:
            case_directory = os.path.join(output_dir, "validation_case_{}".format(index))
        run_validation_case(data_index=index, output_dir=case_directory, model=model, data_file=data_file,
                            training_modalities=training_modalities, output_label_map=output_label_map, labels=labels,
                            threshold=threshold, overlap=overlap, permute=permute)
    data_file.close()


def predict(model, data, permute=False):
    if permute:
        predictions = list()
        for batch_index in range(data.shape[0]):
            predictions.append(predict_with_permutations(model, data[batch_index]))
        return np.asarray(predictions)
    else:
        return model.predict(data)


def predict_with_permutations(model, data):
    predictions = list()
    for permutation_key in generate_permutation_keys():
        temp_data = permute_data(data, permutation_key)[np.newaxis]
        predictions.append(reverse_permute_data(model.predict(temp_data)[0], permutation_key))
    return np.mean(predictions, axis=0)
