import os

from train import config
from unet3d.prediction import run_validation_cases


def main():
    prediction_dir = os.path.abspath("prediction")
    print("The prediction process has started")

    # second_model_file - second model file, deeper model
    # model_file - usual architecture, usual model

    # data_file - file 

    run_validation_cases(validation_keys_file=config["validation_file"],
                         model_file=config["second_model_file"],
                         training_modalities=config["training_modalities"],
                         labels=config["labels"],
                         hdf5_file=config["data_file"],
                         output_label_map=True,
                         output_dir=prediction_dir, isDeeper = True, depth = 5)


if __name__ == "__main__":
    main()
