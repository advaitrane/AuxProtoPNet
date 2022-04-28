import os
import numpy as np
import pandas as pd

# cub_data_dir = "/content/drive/MyDrive/CS_566_DDL/CBM/CUB_200_2011"

def load_attributes_file(cub_data_dir):
    attributes_file = "attributes/attributes.txt"
    attributes_df = pd.read_csv(
        os.path.join(cub_data_dir, attributes_file), 
        sep = ' ',
        names = ['id', 'attribute']
        )
    attributes_dict = dict(zip(attributes_df['id'], attributes_df['attribute']))
    num_attributes = len(attributes_df)
    return attributes_df, attributes_dict

def load_class_file(cub_data_dir):
    class_file = "classes.txt"
    class_df = pd.read_csv(
        os.path.join(cub_data_dir, class_file), 
        sep = ' ',
        names = ['id', 'class']
        )
    class_dict = dict(zip(class_df['id'], class_df['class']))
    return class_df, class_dict

def load_class_attribute_file(cub_data_dir, attributes_df, class_df):
    class_attributes_file = "attributes/class_attribute_labels_continuous.txt"
    class_attributes_df = pd.read_csv(
        os.path.join(cub_data_dir, class_attributes_file), 
        sep = ' ',
        names = attributes_df['id']
        )
    class_attributes_df.index = class_df['id']
    return class_attributes_df

def denoise(class_attributes_df, proportion_threshold = 47.5):
    class_attributes = class_attributes_df.values
    class_attr = (class_attributes >= proportion_threshold).astype(int)

    class_attr_proc = []
    attr_proc = []
    num_class, num_attr = class_attr.shape
    min_class_threshold = 10
    for col in range(num_attr):
        if class_attr[:, col].sum() >= min_class_threshold:
            attr_proc.append(col+1)
            class_attr_proc.append(class_attr[:, col].reshape(-1,1))

    class_attr_proc = np.concatenate(class_attr_proc,1)
    return attr_proc, class_attr_proc

def load_aux_metadata(cub_data_dir, proportion_threshold=47.5):
    attributes_df, attributes_dict = load_attributes_file(cub_data_dir)
    class_df, class_dict = load_class_file(cub_data_dir)
    class_attributes_df = load_class_attribute_file(
        cub_data_dir,
        attributes_df, 
        class_df
        ) 
    attr_proc, class_attr_proc = denoise(class_attributes_df, proportion_threshold)
    return attr_proc, class_attr_proc, attributes_dict, class_dict
