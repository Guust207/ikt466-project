import re


def filename_to_metadata(file_names: list[str], return_labels=False):
    """
    Gets dataset type, fold, source name, label and take from .wav filename.

    :param file_names: List of .wav filename strings
    :param return_labels: Bool, where True means returning only a list of labels.
    :return: List of dictionaries ["dataset_type", "fold", "source_name", "label", "take"] or strings.
    """
    # stft-1-137-A-32.wav
    # stft --> Dataset type
    # 1 --> Fold
    # 137 --> Source name
    # A --> Take
    # 32 --> Label

    metadata_list = []

    for filename in file_names:
        filename = filename.replace(".wav", "")
        dataset_type, fold, source_name, take, label = re.split(r"-", filename)

        data_dict = {"dataset_type": dataset_type, "fold": int(fold), "source_name": source_name, "take": take,
                     "label": int(label)}

        if return_labels:
            metadata_list.append(data_dict["label"])
        else:
            metadata_list.append(data_dict)

    return metadata_list