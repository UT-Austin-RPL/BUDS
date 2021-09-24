"""Hierarchical agglomoration"""

import os
import argparse
import h5py
import numpy as np
import simplejson as json

from sklearn.cluster import SpectralClustering

from PIL import Image
import cv2

import shutil
import pickle

from collections import namedtuple

import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import init_path
from skill_discovery.hierarchical_agglomoration_utils import Node, HierarchicalAgglomorativeTree, save_agglomorative_tree

from models.args_utils import get_common_args

import hydra
from omegaconf import OmegaConf, DictConfig
import yaml
from easydict import EasyDict

def filter_labels(labels):
    for i in range(len(labels)):
        # In the beginning
        if i < 3:
            if labels[i+1] == labels[i+2] == labels[i+3] and labels[i] != labels[i+1]:
                labels[i] = labels[i+1]
        # At tail
        elif len(labels)-3 < i < len(labels) - 1:
            if labels[i-1] == labels[i-2] == labels[i-3] and labels[i] != labels[i-1]:
                labels[i] = labels[i-1]
        elif 3 <= i <= len(labels) - 3:
            # label = find_most_frequent_element(labels)
            if (labels[i-1] == labels[i-2] == labels[i+1] or labels[i-1] == labels[i+1] == labels[i+2]) and (labels[i-1] != labels[i]):
                labels[i] = labels[i-1]
    return labels

@hydra.main(config_path="../conf", config_name="config")
def main(hydra_cfg: DictConfig):
    yaml_config = OmegaConf.to_yaml(hydra_cfg, resolve=True)
    cfg = EasyDict(yaml.load(yaml_config))
    
    print(f"Footprint mode: {cfg.agglomoration.footprint}, Dist mode: {cfg.agglomoration.dist}")
    # if args.skip:
    #     if os.path.exists(f"skill_classification/trees/{args.dataset_name}_trees_{modality_str}_{args.footprint}_{args.dist}.pkl"):
    #         print("Already constructed, skipping")
    #         exit()


    modalities = cfg.repr.modalities
    modality_str = modalities[0]
    for modality in modalities[1:]:
        modality_str += f"_{modality}"
    modality_str += f"_{cfg.repr.alpha_kl}"    

    if cfg.repr.no_skip:
        modality_str += "_no_skip"
    output_parent_dir = f"datasets/{cfg.data.dataset_name}/"
    demo_file = f"{output_parent_dir}/demo.hdf5"
    h5py_file = h5py.File(demo_file, "r")

    print(demo_file)
    num_eps = h5py_file["data"].attrs["num_eps"]

    embedding_hdf5_path = os.path.join(output_parent_dir, f"embedding_{modality_str}_{cfg.repr.z_dim}.hdf5")
    embedding_h5py_f = h5py.File(embedding_hdf5_path, "r")

    save_data = {"dataset_name": cfg.data.dataset_name}
    
    total_len = 0

    X = []
    image_names_list = []
    ep_indices_list = []
    indices_list = []
    step = cfg.agglomoration.agglomoration_step

    try:
        shutil.rmtree("skill_classification/initial_clustering")
    except:
        pass
        
    initial_segments = {}

    init_X = []
    X = []
    X_indices = []


    num_segments = 0

    trees = {"trees": {}}

    for ep_idx in range(num_eps):

        embeddings = embedding_h5py_f[f"data/ep_{ep_idx}/embedding"][()]
        agentview_image_names_list = h5py_file[f"data/ep_{ep_idx}/agentview_image_names"][()]
        # print(embeddings.shape)
        image_names_list.append(agentview_image_names_list)
        init_X.append(embeddings)

        agglomorative_tree = HierarchicalAgglomorativeTree()

        agglomorative_tree.agglomoration(embeddings, step,
                                         footprint_mode=cfg.agglomoration.footprint,
                                         dist_mode=cfg.agglomoration.dist)
        agglomorative_tree.create_root_node()

        trees["trees"][ep_idx] = agglomorative_tree

        # Visualization
        save_agglomorative_tree(agglomorative_tree, agentview_image_names_list, ep_idx, cfg.data.dataset_name,
                                footprint_mode=cfg.agglomoration.footprint,
                                dist_mode=cfg.agglomoration.dist,
                                modality_mode=modality_str)
        
    trees["info"] = {"dataset_name": cfg.data.dataset_name, "num_eps": num_eps}

    with open(f"skill_classification/trees/{cfg.data.dataset_name}_trees_{modality_str}_{cfg.agglomoration.footprint}_{cfg.agglomoration.dist}.pkl", "wb") as f:
        pickle.dump(trees, f)

    h5py_file.close()
    embedding_h5py_f.close()

if __name__ == "__main__":
    main()
