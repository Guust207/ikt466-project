def filename_to_label_name(file_names: list[str]):
    # 1-137-A-32.wav
    # 1 --> Fold
    # 137 --> Source file
    # 32 --> Label
    # A --> Take

    # Can be edited to include the all
    label_list = []

    for filename in file_names:
        filename = filename.replace(".wav", "")
        fold, name, take, label = re.split(r"[-]", filename)

        label_list.append(int(label))

    return label_list