import pickle
import argparse
import h5py
import os
from sklearn import cluster
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

import init_path
from models.args_utils import get_common_args, update_json_config

import hydra
from omegaconf import OmegaConf, DictConfig
import yaml
from easydict import EasyDict
from hydra.experimental import compose, initialize


DEL_COST = INS_COST = 1.0
SUB_COST = 1.0
def ComputeEditDistance(a: list, b: list):

    a_len = len(a) + 1
    b_len = len(b) + 1
    
    dist = np.zeros((a_len, b_len))

    for i in range(b_len):
        dist[0, i] = i
    for i in range(a_len):
        dist[i, 0] = i
    
    for i in range(1, a_len):
        for j in range(1, b_len):
            a_idx = i
            b_idx = j
            min_val = 10000
            cost = 0
            if b_idx > 0 and min_val > dist[a_idx][b_idx - 1]:
                min_val = dist[a_idx][b_idx-1]
                cost = DEL_COST
            if a_idx > 0 and b_idx > 0 and min_val > dist[a_idx-1][b_idx-1]:
                min_val = dist[a_idx-1][b_idx-1]
                cost = SUB_COST
                if a[i - 1] == b[j - 1]:
                    cost = 0
                
            if a_idx > 0 and min_val > dist[a_idx-1][b_idx]:
                min_val = dist[a_idx-1][b_idx]
                cost = INS_COST
            dist[a_idx][b_idx] = min_val + cost
    return dist[-1][-1]

def take_start_idx(elem):
    return elem[0]

def seg_start_idx(elem):
    return elem.start_idx

def segment_footprint(start_idx, end_idx, embeddings, mode="mean"):
    centroid_idx = (start_idx + end_idx) // 2
    if mode == "mean":
        return np.mean([embeddings[start_idx],
                        embeddings[centroid_idx],
                        embeddings[end_idx]], axis=0)

    elif mode == "head":
        return embeddings[start_idx]

    elif mode == "tail":
        return embeddings[end_idx]

    elif mode == "centroid":
        return embeddings[centroid_idx]

    elif mode == "concat_1":
        return np.concatenate([embeddings[start_idx],
                               embeddings[centroid_idx],
                               embeddings[end_idx]], axis=1)
    elif mode == "concat_2":
        return np.concatenate([embeddings[start_idx],
                               embeddings[(start_idx + centroid_idx) // 2],
                               embeddings[centroid_idx],
                               embeddings[(centroid_idx + end_idx) // 2],
                               embeddings[end_idx]], axis=1)
    
class Segment():
    def __init__(self, ep_idx, start_idx, end_idx, label=None):
        self.ep_idx = ep_idx
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.label = label

def assert_wrong_labeling(ep_subtasks_seq):
    for ep_idx in ep_subtasks_seq:
        ep_subtasks_seq[ep_idx].sort(key=seg_start_idx)

    for ep_idx in ep_subtasks_seq:
        for i in range(len(ep_subtasks_seq[ep_idx])-1):
            if ep_subtasks_seq[ep_idx][i].end_idx != ep_subtasks_seq[ep_idx][i+1].start_idx:
                print(f"Ep idx: {ep_idx}, seg idx: {i}")
                import pdb; pdb.set_trace()

@hydra.main(config_path="../conf", config_name="config")
def main(hydra_cfg):
    yaml_config = OmegaConf.to_yaml(hydra_cfg, resolve=True)
    cfg = EasyDict(yaml.load(yaml_config))
    
    print(f"Footprint: {cfg.agglomoration.footprint}, Dist: {cfg.agglomoration.dist}, Segment: {cfg.agglomoration.segment_footprint}, K: {cfg.agglomoration.K}, Affinity: {cfg.agglomoration.affinity}")

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
    num_eps = h5py_file["data"].attrs["num_eps"]

    embedding_hdf5_path = os.path.join(output_parent_dir, f"embedding_{modality_str}_{cfg.repr.z_dim}.hdf5")
    embedding_h5py_f = h5py.File(embedding_hdf5_path, "r")

    with open(f"results/trees/{cfg.data.dataset_name}_trees_{modality_str}_{cfg.agglomoration.footprint}_{cfg.agglomoration.dist}.pkl", "rb") as f:
        trees = pickle.load(f)

    X = []
    locs = []
    init_depth = 0

    if cfg.repr.no_skip:
        scale = cfg.agglomoration.scale
    else:
        scale = 1.0
    
    for (ep_idx, tree) in trees["trees"].items():
        embeddings = embedding_h5py_f[f"data/ep_{ep_idx}/embedding"][()]

        depth = init_depth
        node_list = []
        prev_node_list_len = -1
        while len(node_list) < cfg.agglomoration.K  * cfg.agglomoration.segment_scale and prev_node_list_len < len(node_list):
            # node_list = tree.find_children_nodes(tree.root_node.idx, depth=depth)
            prev_node_list_len = len(node_list)
            node_list = tree.find_midlevel_abstraction(tree.root_node.idx, depth=depth, min_len=0)
            depth += 1


        for node_idx in node_list:
            node = tree.nodes[node_idx]
            embedding = segment_footprint(node.start_idx, node.end_idx, embeddings, cfg.agglomoration.segment_footprint) * scale
            X.append(embedding.squeeze())
            locs.append((ep_idx, node_idx))

    K = cfg.agglomoration.K
    colors = ['r', 'b', 'g', 'y', 'k', 'C0', 'C1', 'C2', 'magenta', 'lightpink', 'deepskyblue', 'lawngreen'] + list(mcolors.CSS4_COLORS.values())
    
    clustering = cluster.SpectralClustering(n_clusters=K,
                                            assign_labels="discretize",
                                            affinity=cfg.agglomoration.affinity
                                            ).fit(X)
    labels = clustering.labels_
    print(clustering.get_params())
    
    loc_dict = {}
    for (loc, label) in zip(locs, labels):
        if loc[0] not in loc_dict:
            loc_dict[loc[0]] = []
        loc_dict[loc[0]].append((loc[1], label))

    merge_nodes = []
    for ep_idx in loc_dict:
        previous_label = None
        start_idx = None
        end_idx = None
        for (loc, label) in loc_dict[ep_idx]:
            node = trees["trees"][ep_idx].nodes[loc]
            
            if previous_label is None:
                previous_label = label
                start_idx = node.start_idx
                end_idx = node.end_idx
            else:
                if previous_label == label and node.end_idx > start_idx:
                    end_idx = node.end_idx
                else:
                    merge_nodes.append([ep_idx, start_idx, end_idx, previous_label])
                    previous_label = label
                    start_idx = node.start_idx
                    end_idx = node.end_idx

        merge_nodes.append([ep_idx, start_idx, end_idx, label])

    # ----------------Process start -----------------        
    ep_subtasks_seq = {}

    max_len = 0
    X = []
    locs = []
    for ep_idx, start_idx, end_idx, label in merge_nodes:
        if ep_idx not in ep_subtasks_seq:
            ep_subtasks_seq[ep_idx] = []
        ep_subtasks_seq[ep_idx].append(Segment(ep_idx, start_idx, end_idx, label))
        if end_idx > max_len:
            max_len = end_idx

    for ep_idx in ep_subtasks_seq:
        ep_subtasks_seq[ep_idx].sort(key=seg_start_idx)

    for ep_idx in ep_subtasks_seq:
        for seg in ep_subtasks_seq[ep_idx]:
            locs.append((seg.ep_idx, seg.start_idx, seg.end_idx))
            embedding = segment_footprint(seg.start_idx, seg.end_idx, embedding_h5py_f[f"data/ep_{ep_idx}/embedding"][()], cfg.agglomoration.segment_footprint) * scale
            X.append(embedding.squeeze())

    clustering = cluster.SpectralClustering(n_clusters=K,
                                               assign_labels="discretize",
                                               affinity=cfg.agglomoration.affinity).fit(X)

    # If there is a cluster whose average length is shorter than a
    # threshold, merge the segments and decrease the number of
    # cluster, redo clustering again

    # Reorder label
    label_mapping = []
    labels = clustering.labels_    
    for (loc, label) in zip(locs, labels):
        if label not in label_mapping and len(label_mapping) < K:
            label_mapping.append(label)
    new_labels = []
    for label in labels:
        new_labels.append(label_mapping.index(label))
    labels = new_labels

    subtask_len_clusters = {}
    for label in range(K):
        subtask_len_clusters[label] = []

    cluster_indices = []
    for (loc, label) in zip(locs, labels):
        subtask_len_clusters[label].append(loc[2] - loc[1])
        cluster_indices.append(label)

    cluster_indices = np.array(cluster_indices)

    counter = 0
    for ep_idx in ep_subtasks_seq:
        for seg in ep_subtasks_seq[ep_idx]:
            seg.label = labels[counter]
            counter += 1

    min_len_thresh = cfg.agglomoration.min_len_thresh
    remove_labels = []
    for (label, c) in subtask_len_clusters.items():
        print(label, np.mean(c))
        if np.mean(c) < min_len_thresh:
            remove_labels.append(label)
            print(colors[label])
    print(remove_labels)

    has_removed_labels = False

    while not has_removed_labels:
        new_ep_subtasks_seq = {}
        for ep_idx, ep_subtask_seq in ep_subtasks_seq.items():
            new_ep_subtask_seq = []
            previous_label = None
            for idx in range(len(ep_subtask_seq)):
                if previous_label is None:
                    previous_label = ep_subtask_seq[idx].label
                    start_idx = ep_subtask_seq[idx].start_idx
                    end_idx = ep_subtask_seq[idx].end_idx
                else:
                    if previous_label == ep_subtask_seq[idx].label and ep_subtask_seq[idx].end_idx > start_idx:
                        end_idx = ep_subtask_seq[idx].end_idx
                    else:
                        new_ep_subtask_seq.append(Segment(ep_idx, start_idx, end_idx, previous_label))
                        previous_label = ep_subtask_seq[idx].label
                        start_idx = ep_subtask_seq[idx].start_idx
                        end_idx = ep_subtask_seq[idx].end_idx

            new_ep_subtask_seq.append(Segment(ep_idx, start_idx, end_idx, ep_subtask_seq[idx].label))
            new_ep_subtasks_seq[ep_idx] = new_ep_subtask_seq

        ep_subtasks_seq = new_ep_subtasks_seq
        new_ep_subtasks_seq = {}
        for ep_idx, ep_subtask_seq in ep_subtasks_seq.items():
            new_ep_subtask_seq = []
            idx = 0
            embeddings = embedding_h5py_f[f"data/ep_{ep_idx}/embedding"][()]
            while idx < len(ep_subtask_seq):
                d1 = None
                d2 = None
                if idx == 0:
                    d2 = 0
                elif idx == len(ep_subtask_seq) - 1:
                    d1 = 0
                else:
                    d1 = trees["trees"][ep_idx].compute_distance(segment_footprint(ep_subtask_seq[idx].start_idx, ep_subtask_seq[idx].end_idx, embeddings, cfg.agglomoration.segment_footprint),
                                                                 segment_footprint(ep_subtask_seq[idx-1].start_idx, ep_subtask_seq[idx-1].end_idx, embeddings, cfg.agglomoration.segment_footprint))

                    d2 = trees["trees"][ep_idx].compute_distance(segment_footprint(ep_subtask_seq[idx].start_idx, ep_subtask_seq[idx].end_idx, embeddings, cfg.agglomoration.segment_footprint),
                                                                 segment_footprint(ep_subtask_seq[idx+1].start_idx, ep_subtask_seq[idx+1].end_idx, embeddings, cfg.agglomoration.segment_footprint))
                if ep_subtask_seq[idx].label in remove_labels: #  or (ep_subtask_seq[idx].end_idx - ep_subtask_seq[idx].start_idx) <= min_len_thresh:
                    # Merge
                    if d1 is None:
                        # Merge with after
                        new_seg = Segment(ep_idx, ep_subtask_seq[idx].start_idx, ep_subtask_seq[idx+1].end_idx, ep_subtask_seq[idx+1].label)
                        new_ep_subtask_seq.append(new_seg)
                        step = 2
                    elif d2 is None:
                        # Merge with before
                        new_seg = Segment(ep_idx, ep_subtask_seq[idx-1].start_idx, ep_subtask_seq[idx].end_idx, ep_subtask_seq[idx-1].label)
                        if new_ep_subtask_seq[-1].start_idx == new_seg.start_idx:
                            new_ep_subtask_seq.pop()
                        new_ep_subtask_seq.append(new_seg)
                        step = 1
                    else:
                        if d1 < d2:
                            new_seg = Segment(ep_idx, ep_subtask_seq[idx-1].start_idx, ep_subtask_seq[idx].end_idx, ep_subtask_seq[idx-1].label)
                            if new_ep_subtask_seq[-1].end_idx > new_seg.start_idx:
                                new_seg.start_idx = new_ep_subtask_seq[-1].end_idx
                            if new_ep_subtask_seq[-1].start_idx == new_seg.start_idx:
                                new_ep_subtask_seq.pop()
                            new_ep_subtask_seq.append(new_seg)
                            step = 1
                        else:
                            new_seg = Segment(ep_idx, ep_subtask_seq[idx].start_idx, ep_subtask_seq[idx+1].end_idx, ep_subtask_seq[idx+1].label)
                            new_ep_subtask_seq.append(new_seg)
                            step = 2
                else:
                    new_ep_subtask_seq.append(ep_subtask_seq[idx])                    
                    step = 1
                
                idx += step

            # new_ep_subtasks_seq[ep_idx] = new_ep_subtask_seq
            new_ep_subtasks_seq[ep_idx] = []
            previous_label = None
            start_idx = None
            end_idx = None
            
            for i in range(len(new_ep_subtask_seq)):
                label = new_ep_subtask_seq[i].label
                if previous_label is None:
                    start_idx = new_ep_subtask_seq[i].start_idx
                    end_idx = new_ep_subtask_seq[i].end_idx
                    previous_label = new_ep_subtask_seq[i].label
                else:
                    if previous_label == label and new_ep_subtask_seq[i].end_idx > start_idx:
                        end_idx = new_ep_subtask_seq[i].end_idx
                    else:
                        new_ep_subtasks_seq[ep_idx].append(Segment(ep_idx, start_idx, end_idx, previous_label))
                        previous_label = label
                        start_idx = new_ep_subtask_seq[i].start_idx
                        end_idx = new_ep_subtask_seq[i].end_idx
            new_ep_subtasks_seq[ep_idx].append(Segment(ep_idx, start_idx, end_idx, label))
            
        ep_subtasks_seq = new_ep_subtasks_seq

        has_removed_labels = True
        for (ep_idx, ep_subtask_seq) in ep_subtasks_seq.items():
            for idx in range(len(ep_subtask_seq)):
                if ep_subtask_seq[idx].label in remove_labels:
                    has_removed_labels = False
                    break
            if not has_removed_labels:
                break
        
    K = K - len(remove_labels)
    print(ep_subtasks_seq.keys())
    # ----------------Process finished -----------------
    for (ep_idx, ep_subtask_seq) in ep_subtasks_seq.items():
        for seg in ep_subtask_seq:
            if h5py_file[f"data/ep_{ep_idx}"].attrs["task_id"] in [5, 6, 8]:
                plt.plot([seg.start_idx, seg.end_idx], [-seg.ep_idx, -seg.ep_idx], colors[seg.label], linewidth=3)
            else:
                plt.plot([seg.start_idx, seg.end_idx], [seg.ep_idx, seg.ep_idx], colors[seg.label], linewidth=3)

    plt.xlabel("Sequence")
    plt.ylabel("No. Demo")

    os.makedirs(f"skill_data_vis/{cfg.data.dataset_name}", exist_ok=True)
    if not cfg.agglomoration.visualization:
        plt.savefig(f"skill_data_vis/{cfg.data.dataset_name}/{modality_str}_{cfg.agglomoration.footprint}_{cfg.agglomoration.dist}_{cfg.agglomoration.segment_footprint}_K{K}_{cfg.agglomoration.affinity}.png")

    plt.show()

    from PIL import Image
    mod_idx = 0
    while mod_idx < 5:
        mod_idx += 1
        fig = plt.figure(figsize=(100, 50))
        scale = 3.
        plt.xlim([0, (max_len) * scale])
        plt.ylim([-5, num_eps + 5]) 
       # plt.tight_layout()
        ax = plt.gca()
        trans = ax.transData.transform
        trans2 = fig.transFigure.inverted().transform
        img_size = 0.10

        image_info = []
        for (ep_idx, ep_subtask_seq) in ep_subtasks_seq.items():
            if ep_idx % (num_eps // 5) != mod_idx:
                continue
            for seg in ep_subtask_seq:
                start_idx, end_idx, label = seg.start_idx, seg.end_idx, seg.label
                point = plt.plot([start_idx * scale, end_idx * scale], [ep_idx, ep_idx], colors[label], linewidth=3)
                xa, ya = trans2(trans([point[0].get_data()[0][0], point[0].get_data()[1][0]]))
                image_info.append((xa, ya, start_idx, ep_idx))

            xa, ya = trans2(trans([point[0].get_data()[0][1], point[0].get_data()[1][1]]))
            image_info.append((xa, ya, end_idx, ep_idx))



        for info in image_info:
            xa, ya, start_idx, ep_idx = info
            agentview_image = np.array(Image.open(h5py_file[f"data/ep_{ep_idx}/agentview_image_names"][()][start_idx]))
            new_axis = plt.axes([xa - img_size / 2, ya + img_size / 50, img_size, img_size])
            new_axis.imshow(agentview_image)
            new_axis.set_aspect('equal')
            new_axis.axis('off')



        plt.xlabel("Sequence")
        plt.ylabel("No. Demo")
        ax.axis('off')

        plt.show()

    if cfg.agglomoration.visualization:
        exit()


    os.makedirs("results/skill_data", exist_ok=True)
    subtask_file_name = f"results/skill_data/{cfg.data.dataset_name}_subtasks_{modality_str}_{cfg.repr.z_dim}_{cfg.agglomoration.footprint}_{cfg.agglomoration.dist}_{cfg.agglomoration.segment_footprint}_K{K}_{cfg.agglomoration.affinity}.hdf5"
    subtask_file = h5py.File(subtask_file_name, "w")

    grp = subtask_file.create_group("subtasks")
    grp.attrs["num_subtasks"] = K

    subtasks_grps = []
    subtasks = {}

    label_mapping = []
    for i in range(cfg.agglomoration.K):
        if i not in remove_labels:
            label_mapping.append(i)
    for i in range(K):
        subtasks_grps.append(grp.create_group(f"subtask_{i}"))
        subtasks[i] = []

    for ep_idx, ep_subtask_seq in ep_subtasks_seq.items():
        for seg in ep_subtask_seq:
            subtasks[label_mapping.index(seg.label)].append([ep_idx, seg.start_idx, seg.end_idx])

    # print(ep_subtasks_seq[1])
    for i in range(K):
        subtasks_grps[i].create_dataset("segmentation", data=subtasks[i])

    ep_strings = []
    for ep_idx in ep_subtasks_seq:
        subtask_seq = ep_subtasks_seq[ep_idx]
        subtask_seq.sort(key=seg_start_idx)

        ep_string = []
        for subtask in subtask_seq:
            ep_string.append(subtask.label)
        ep_strings.append(ep_string)
        # print(f"{ep_idx}: {ep_string}")
    assert_wrong_labeling(ep_subtasks_seq)
    

    for ep_idx in ep_subtasks_seq:
        ep_subtasks_seq[ep_idx].sort(key=seg_start_idx)
        saved_ep_subtasks_seq = []
        prev_seg = None
        for seg in ep_subtasks_seq[ep_idx]:
            saved_ep_subtasks_seq.append([seg.start_idx, seg.end_idx, label_mapping.index(seg.label)])
            if prev_seg is not None:
                assert(seg.start_idx == prev_seg.end_idx)
            prev_seg = seg
        grp.create_dataset(f"ep_subtasks_seq_{ep_idx}", data=saved_ep_subtasks_seq)

    grp.attrs["num_eps"] = num_eps
    
    for ep_idx in ep_subtasks_seq:
        for i in range(len(ep_subtasks_seq[ep_idx])-1):
            if ep_subtasks_seq[ep_idx][i].end_idx != ep_subtasks_seq[ep_idx][i+1].start_idx:
                print(f"Ep idx: {ep_idx}, seg idx: {i}")
                import pdb; pdb.set_trace()
    
        
    print(f"Final K: {K}")

    score = 0.

    num_pairs = 0
    for i in range(len(ep_strings)):
        for j in range(len(ep_strings)):
            if i >= j:
                continue
            dist = ComputeEditDistance(ep_strings[i], ep_strings[j])
            score += dist
            num_pairs += 1

    score /= ((len(ep_strings) * (len(ep_strings) - 1)) / 2)
    print(score)
    grp.attrs["score"] = score
    
    subtask_file.close()
    h5py_file.close()
    embedding_h5py_f.close()

if __name__ == "__main__":
    main()
