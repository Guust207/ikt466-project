import re


def filename_to_metadata(file_names: list[str], return_labels=False):
    """
    Gets fold, source name, label and take from .wav filename.

    :param file_names: List of .wav filename strings
    :param return_labels: Bool, where True means returning only a list of labels.
    :return: List of dictionaries ["fold", "source_file", "label", "take"] or strings.
    """
    # 1-137-A-32.wav
    # 1 --> Fold
    # 137 --> Source file
    # A --> Take
    # 32 --> Label

    metadata_list = []

    for filename in file_names:
        filename = filename.replace(".wav", "")
        fold, source_file, take, label = re.split(r"-", filename)

        data_dict = {"fold": fold, "source_file": source_file, "take": take, "label": label}

        if return_labels:
            metadata_list.append(data_dict["label"])
        else:
            metadata_list.append(data_dict)

    return metadata_list