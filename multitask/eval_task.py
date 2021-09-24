"""Use meta policy mechanism for running the policy."""

import os
import argparse
import h5py
import random
import ast

import init_path
from models.model_utils import safe_cuda
from models.conf_utils import *
from models.torch_utils import *
from skill_discovery.hierarchical_agglomoration_utils import Node, HierarchicalAgglomorativeTree, save_agglomorative_tree


from policy_learning.models import *
from policy_learning.path_templates import *

from PIL import Image, ImageDraw
from matplotlib import cm
import numpy  as np
import robosuite as suite
from robosuite import load_controller_config
import robosuite.utils.transform_utils as T
import cv2
from robosuite_task_zoo.environments.manipulation import *
from demonstration_tools import postprocess_model_xml
import imageio
from easydict import EasyDict
import json
from torch.utils.tensorboard import SummaryWriter
# from sklearn.manifold import TSNE

import hydra
from omegaconf import OmegaConf, DictConfig
import yaml
from easydict import EasyDict
from hydra.experimental import compose, initialize
import pprint
from pathlib import Path

import robosuite.utils.macros as macros
macros.IMAGE_CONVENTION = "opencv"

USE_SPACEMOUSE = False
device = None
if USE_SPACEMOUSE:
    from robosuite.utils.input_utils import input2action
    from robosuite.devices import SpaceMouse
    device = SpaceMouse(9583, 50734, pos_sensitivity=2.0, rot_sensitivity=0.1)
    device.start_control()

OUR_MODEL = "ours"


def offscreen_visualization(env, use_eye_in_hand=True):    
    img = env.sim.render(height=512, width=512, camera_name="agentview")[::-1]
    
    if use_eye_in_hand:
        new_img = np.ones((512, 512 + 512, 3)).astype(np.uint8) * 255
        new_img[0:512, 0:512, :3] = img
        # if goal_img is None:
        new_img[200:328, 512:512+128, :3] = env.sim.render(height=128, width=128, camera_name="robot0_eye_in_hand")[::-1]
        img = new_img
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
    return img

def get_subtask_id(num_step, demo_subtask_seq):
    for (start_idx, end_idx, subtask_id) in demo_subtask_seq:
        if start_idx <= num_step < end_idx:
            return subtask_id

    # If num_step larger then end_idx
    largest_end_idx = 0
    subtask_id = 0
    for (start_idx, end_idx, subtask_id) in demo_subtask_seq:
        if end_idx > largest_end_idx:
            largest_end_idx = end_idx
            subtask_id = subtask_id
    return subtask_id

class EvalMetaPolicy():
    def __init__(self,
                 subtask_seq,
                 subtask_policies,
                 meta_policy,
                 policy_type):

        self.subtask_seq = subtask_seq
        self.current_subtask_idx = 0

        self.subtask_policies = subtask_policies
        self.meta_policy = meta_policy

        self.policy_type = policy_type

        self.skill_finished_counter = 0
        # Make sure it is stable for the first 10 steps of each skill
        self.skill_starting_counter = 0
        self._id_buffer = []

        self.rnn_init_state = None

    def reset(self):
        self.current_subtask_idx = 0
        self.skill_finished_counter = 0
        self.skill_starting_counter = 0
        self.current_subtask_id = 0
        self._id_buffer = []
        self.counter = 0
        self.rnn_init_state = None
        self.prev_subgoal_embedding = None
        
        self.keypoins = None

    def get_skill_info(self, meta_state):
        prediction = self.meta_policy.predict(meta_state)

        return prediction["subtask_id"][0], prediction["embedding"]
        
    def step(self, policy_state, meta_state, ext_embedding=None, freq=1, subgoal_embeddings=None):

        if self.counter % freq == 0:
            subtask_id, subgoal_embedding = self.get_skill_info(meta_state)
            self.prev_subgoal_embedding = subgoal_embedding
            if self.current_subtask_id != subtask_id:
                self.current_subtask_id = subtask_id
        self.counter += 1

        if ext_embedding is not None:
            self.prev_subgoal_embedding = safe_cuda(torch.from_numpy(ext_embedding).unsqueeze(0))
        if self.policy_type == PolicyType.NORMAL_SUBGOAL:
            policy_state["subgoal_embedding"] = self.prev_subgoal_embedding

        actions = []
        action = self.subtask_policies[self.current_subtask_id].get_action(policy_state)
        action = action.detach().cpu().numpy()[0]

        return action, self.current_subtask_id

    @property
    def subgoal(self):
        return self.prev_subgoal_embedding.detach().cpu().numpy()

@hydra.main(config_path="../conf", config_name="config")
def main(hydra_cfg):
    yaml_config = OmegaConf.to_yaml(hydra_cfg, resolve=True)
    cfg = EasyDict(yaml.load(yaml_config))
    pp = pprint.PrettyPrinter(indent=4)
    if cfg.verbose:
        pp.pprint(cfg)
    
    print("Evaluating in mode: ", cfg.eval.mode)

    modalities = cfg.repr.modalities
    modality_str = get_modalities_str(cfg)
    suffix_str = ""
    goal_str = get_goal_str(cfg)

    data_modality_str = get_data_modality_str(cfg)
    # Get controller config
    controller_config = load_controller_config(default_controller=cfg.env.controller)
    env_name = cfg.data.env
    # Create argument configuration
    config = {
        "robots": cfg.env.robots,
        "controller_configs": controller_config,
    }
    demo_path = cfg.data.folder
    demo_file_path = os.path.join(demo_path, "demo.hdf5")
    f = h5py.File(demo_file_path, "r")

    # OSC_POSITION action dimension
    action_dim = 4
    proprio_dim = 0

    subtask_policies = []
    subtask_terminations = []
    if "proprio" in cfg.skill_training.data_modality:
        proprio_dim = 0
        if cfg.skill_training.use_gripper:
            # past 5 frames of joint readings for gripper
            proprio_dim += 2 * 5
        if cfg.skill_training.use_joints:
            proprio_dim += 7
        

    demos_subtask_sequence = {}
    num_subtasks = cfg.skill_training.agglomoration.K

    if cfg.eval.mode == OUR_MODEL:
        if cfg.eval.singletask_baseline:
            subtask_file_name=f"results/skill_data/{cfg.data.dataset_name}_SingleTask_{cfg.multitask.training_task_id}_subtasks_{modality_str}_{cfg.repr.z_dim}_{cfg.agglomoration.footprint}_{cfg.agglomoration.dist}_{cfg.agglomoration.segment_footprint}_K{cfg.skill_training.agglomoration.K}_{cfg.agglomoration.affinity}.hdf5"
        else:
            subtask_file_name=f"results/skill_data/{cfg.data.dataset_name}_subtasks_{modality_str}_{cfg.repr.z_dim}_{cfg.agglomoration.footprint}_{cfg.agglomoration.dist}_{cfg.agglomoration.segment_footprint}_K{cfg.skill_training.agglomoration.K}_{cfg.agglomoration.affinity}.hdf5"
        subtask_file = h5py.File(subtask_file_name, "r")

        num_subtasks = subtask_file["subtasks"].attrs["num_subtasks"]
        for subtask_id in range(num_subtasks):
            subtask_segmentation = subtask_file["subtasks"][f"subtask_{subtask_id}"]["segmentation"][()]
            for (i, start_idx, end_idx) in subtask_segmentation:
                if i not in demos_subtask_sequence:
                    demos_subtask_sequence[i] = []

                demos_subtask_sequence[i].append((start_idx, end_idx, subtask_id))

    if cfg.skill_subgoal_cfg is not None:
        subgoal_visual_feature_dimension = cfg.skill_subgoal_cfg.visual_feature_dimension
    else:
        raise ValueError

    if cfg.skill_training.policy_type == "no_subgoal":
        policy_type = PolicyType.NO_SUBGOAL
    elif cfg.skill_training.policy_type == "normal_subgoal":
        policy_type = PolicyType.NORMAL_SUBGOAL
    elif cfg.skill_training.policy_type == "vae_subgoal":
        policy_type = PolicyType.VAE_SUBGOAL

    if cfg.eval.mode == OUR_MODEL:
        for i in range(num_subtasks):
            if cfg.eval.singletask_baseline:
                template = single_subskill_path_template(cfg, subtask_id=i)
            else:
                template = subskill_path_template(cfg, subtask_id=i)                
            output_dir = template.output_dir
            model_checkpoint_name = template.model_checkpoint_name
            state_dict, policy_cfg = torch_load_model(model_checkpoint_name)
            if cfg.verbose:
                print(model_checkpoint_name)
            policy = safe_cuda(BCPolicy(action_dim=action_dim,
                                        state_dim=policy_cfg.skill_training.state_dim,
                                        proprio_dim=proprio_dim,
                                        data_modality=policy_cfg.skill_training.data_modality,
                                        use_eye_in_hand=policy_cfg.skill_training.use_eye_in_hand,
                                        use_subgoal_eye_in_hand=policy_cfg.skill_subgoal_cfg.use_eye_in_hand,
                                        use_subgoal_spatial_softmax=policy_cfg.skill_subgoal_cfg.use_spatial_softmax,
                                        activation=policy_cfg.skill_training.activation,
                                        z_dim=policy_cfg.repr.z_dim,
                                        num_kp=policy_cfg.skill_training.num_kp,
                                        img_h=policy_cfg.skill_training.img_h,
                                        img_w=policy_cfg.skill_training.img_w,
                                        visual_feature_dimension=policy_cfg.skill_training.visual_feature_dimension,
                                        subgoal_visual_feature_dimension=subgoal_visual_feature_dimension,
                                        action_squash=policy_cfg.skill_training.action_squash,
                                        policy_layer_dims=policy_cfg.skill_training.policy_layer_dims,
                                        policy_type=policy_type,
                                        subgoal_type=cfg.skill_subgoal_cfg.subgoal_type))

            policy.load_state_dict(state_dict)
            policy.eval()
            subtask_policies.append(policy)

    # Collect ep_idx information
    keys = list(f["data"].keys())
    demos = []
    for key in keys:
        if "ep" in key:
            demos.append(key)

    # Create environment
    env = eval(cfg.env.environment)(
        **config,
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_camera_obs=True,
        camera_names=["robot0_eye_in_hand",
                      "agentview",
                      # "frontview",
                      ],
        reward_shaping=True,
        control_freq=20,
        camera_heights=128,
        camera_widths=128,
        task_id=cfg.multitask.task_id,
    )
    # Load single behavior cloning policy

    embedding_dim = cfg.repr.z_dim

    if cfg.eval.mode == OUR_MODEL:
        if cfg.eval.singletask_baseline:
            meta_template = singletask_multitask_meta_path_template(cfg)
        else:
            meta_template = multitask_meta_path_template(cfg)
        meta_output_dir = meta_template.output_dir
        meta_model_name = meta_template.model_name
        meta_state_dict, meta_cfg = torch_load_model(meta_model_name)
        if cfg.verbose:
            print(meta_model_name)
        if "use_spatial_softmax" in meta_cfg.meta:
            use_meta_spatial_softmax = meta_cfg.meta.use_spatial_softmax
        else:
            use_meta_spatial_softmax = cfg.meta.use_spatial_softmax

        if cfg.meta_cvae_cfg.enable:
            meta_policy = safe_cuda(MetaCVAEPolicy(num_subtasks=meta_cfg.num_subtasks,
                                                   subgoal_embedding_dim=meta_cfg.subgoal_embedding_dim,
                                                   id_layer_dims=meta_cfg.meta.id_layer_dims,
                                                   embedding_layer_dims=meta_cfg.meta.embedding_layer_dims,
                                                   use_eye_in_hand=meta_cfg.meta.use_eye_in_hand,
                                                   subgoal_type=meta_cfg.skill_subgoal_cfg.subgoal_type,
                                                   use_spatial_softmax=use_meta_spatial_softmax,
                                                   policy_type=cfg.skill_training.policy_type,
                                                   latent_dim=meta_cfg.meta_cvae_cfg.latent_dim,
                                                   activation=meta_cfg.meta.activation))
        
        meta_policy.load_state_dict(meta_state_dict)
        meta_policy.eval()

    if cfg.eval.mode == "ours":
        if cfg.eval.singletask_baseline:
            subgoal_embedding_file_name = singletask_subgoal_embedding_path_template(cfg, modality_str)
        else:
            subgoal_embedding_file_name = subgoal_embedding_path_template(cfg, modality_str)
        subgoal_embedding_file = h5py.File(subgoal_embedding_file_name, "r")

    num_success = 0

    if cfg.eval.mode == OUR_MODEL:
        eval_policy = EvalMetaPolicy([],
                                     subtask_policies,
                                     meta_policy,
                                     policy_type=policy_type)
    else:
        template = single_skill_path_template(cfg)
        model_checkpoint_name = template.model_checkpoint_name
        policy_state_dict, policy_cfg = torch_load_model(model_checkpoint_name)
        eval_policy = safe_cuda(BaselineBCPolicy(action_dim=action_dim,
                                   state_dim=policy_cfg.skill_training.state_dim,
                                   proprio_dim=proprio_dim,
                                   data_modality=policy_cfg.skill_training.data_modality,
                                   use_eye_in_hand=policy_cfg.skill_training.use_eye_in_hand,
                                   use_subgoal_eye_in_hand=policy_cfg.skill_subgoal_cfg.use_eye_in_hand,
                                   activation=policy_cfg.skill_training.activation,
                                   z_dim=policy_cfg.repr.z_dim,
                                   num_kp=policy_cfg.skill_training.num_kp,
                                   img_h=policy_cfg.skill_training.img_h,
                                   img_w=policy_cfg.skill_training.img_w,
                                   visual_feature_dimension=policy_cfg.skill_training.visual_feature_dimension,
                                   subgoal_visual_feature_dimension=subgoal_visual_feature_dimension,
                                   action_squash=policy_cfg.skill_training.action_squash,
                                   policy_layer_dims=policy_cfg.skill_training.policy_layer_dims))
        eval_policy.load_state_dict(policy_state_dict)
        eval_policy.eval()


    demo_task_goal = None

    summary_writer_name = f"results/eval/{env_name}_{cfg.eval.mode}_{cfg.multitask.task_id}"
    summary_writer = SummaryWriter(summary_writer_name)

    if not cfg.eval.testing:
        ep_indices = f["data/task"].attrs[f"{cfg.multitask.task_id}"]
        if cfg.verbose:
            print(ep_indices)
        num_eval = len(ep_indices)
    else:
        num_eval = 100
        if cfg.multitask.training_task_id != -1:
            num_eval = 50
    video_dir = None
    states_dir = None
    for i in range(num_eval):
        prev_skill_id = -1
        min_idx = 0

        record_states = []
        if not cfg.eval.testing:
            demo = f"ep_{ep_indices[i]}"
            ep_idx = int(ep_indices[i])
            print(f"data/{demo}/agentview_image_names")
        else:
            if i % 5 == 0 and cfg.verbose:
                print(f"eps: {i}")
        try:
            saved_ep_subtasks_seq = subtask_file["subtasks"][f"ep_subtasks_seq_{ep_idx}"][()]
            # print(saved_ep_subtasks_seq)
        except:
            pass

        if cfg.eval.mode == OUR_MODEL:
            eval_policy.reset()

        if not cfg.eval.testing:
            actions = f[f"data/{demo}/actions"][()]
            gt_states = f[f"data/{demo}/gt_states"][()]


        env.reset()

        if not cfg.eval.testing:
            model_xml = f[f"data/{demo}"].attrs["model_file"]
            xml = postprocess_model_xml(model_xml, {"robot0_eye_in_hand": {"pos": "0.05 0.0 0.049999", "quat": "0.0 0.707108 0.707108 0.0"}})            

        steps = 5
        
        initial_mjstate = env.sim.get_state().flatten()
        if not cfg.eval.testing:
            model_xml = env.sim.model.get_xml()
            xml = postprocess_model_xml(model_xml, {"robot0_eye_in_hand": {"pos": "0.05 0.0 0.049999", "quat": "0.0 0.707108 0.707108 0.0"}})
            
            env.reset_from_xml_string(xml)

            env.sim.reset()
            env.sim.set_state_from_flattened(gt_states[steps])
            env.step(actions[steps])
            # env.sim.forward()
        else:
            model_xml = env.sim.model.get_xml()
            xml = postprocess_model_xml(model_xml, {"robot0_eye_in_hand": {"pos": "0.05 0.0 0.049999", "quat": "0.0 0.707108 0.707108 0.0"}})
            env.reset_from_xml_string(xml)
            env.sim.reset()
            env.sim.set_state_from_flattened(initial_mjstate)
            env.sim.forward()
            for _ in range(5):
                env.step([0.] * 3 + [-1.])

        obs = env._get_observations()

        done = False
        max_steps = cfg.eval.max_steps
        recorded_imgs = []

        gripper_history = []
        seq = []
        while not done and steps < max_steps:
            steps += 1

            eye_in_hand_image = safe_cuda(torch.from_numpy(np.array(obs["robot0_eye_in_hand_image"]).transpose(2, 0, 1)).float() / 255.).unsqueeze(0)
            agentview_image = safe_cuda(torch.from_numpy(np.array(obs["agentview_image"]).transpose(2, 0, 1)).float() / 255.).unsqueeze(0)
            
            if cfg.skill_training.use_eye_in_hand:
                state_image = torch.cat((agentview_image, eye_in_hand_image), dim=1)
            else:
                state_image = agentview_image

            # Initialize gripper state
            if gripper_history == []:
                for _ in range(5):
                    gripper_history.append(obs["robot0_gripper_qpos"])

            gripper_state = np.array(gripper_history).reshape(-1)
            if cfg.skill_training.use_gripper and cfg.skill_training.use_joints:
                proprio_state = np.hstack((obs["robot0_joint_pos"], gripper_state))
            elif cfg.skill_training.use_joints:
                proprio_state = np.array(obs["robot0_joint_pos"])
            elif cfg.skill_training.use_gripper:
                proprio_state = gripper_state

            gripper_history.pop(0)
            gripper_history.append(obs["robot0_gripper_qpos"])

            policy_state = {"state_image": state_image,
                            "proprio": safe_cuda(torch.from_numpy(proprio_state)).float().unsqueeze(0)}


            if cfg.eval.mode == OUR_MODEL:
                if meta_cfg.meta.use_eye_in_hand:
                    meta_state = {"state_image": state_image}
                else:
                    meta_state = {"state_image": agentview_image}
                
                action, skill_id = eval_policy.step(policy_state=policy_state,
                                                    meta_state=meta_state,
                                                    freq=cfg.eval.meta_freq,
                                                    subgoal_embeddings=None)
                if prev_skill_id != skill_id:
                    prev_skill_id = skill_id
                    seq.append((steps, skill_id))


            elif cfg.eval.mode == "single_skill":
                action = eval_policy.get_action(policy_state).detach().cpu().numpy()[0]
            elif cfg.eval.mode == "gti":
                raise NotImplementedError
            else:
                raise NotImplementedError

            if cfg.record_states:
                state = env.sim.get_state().flatten()
                record_states.append(state)
            
            obs, reward, done, info = env.step(action)
            done = env._check_success()
            if cfg.eval.visualization:
                img = offscreen_visualization(env, use_eye_in_hand=cfg.skill_training.use_eye_in_hand)
                
            if cfg.eval.video:
                video_image = env.sim.render(height=512, width=512, camera_name="agentview")[::-1]
                env.sim.render(height=128, width=128, camera_name="robot0_eye_in_hand")[::-1]
                recorded_imgs.append(video_image)

        if done:
            num_success += 1
            if cfg.eval.video:
                # Record few more states for better visualization
                for _ in range(20):
                    env.step([0.] * 3 + [-1.])
                    video_image = env.sim.render(height=512, width=512, camera_name="agentview")[::-1]
                    recorded_imgs.append(video_image)

            if cfg.record_states:
                for _ in range(20):
                    state = env.sim.get_state().flatten()
                    record_states.append(state)
                    env.step([0.] * 3 + [-1.])
                    
        
        summary_writer.add_scalar("Num Success", num_success)
                    
        if cfg.meta_cvae_cfg.enable:
            video_name_suffix = f"{cfg.meta_cvae_cfg.kl_coeff}_{cfg.meta_cvae_cfg.latent_dim}_False_cvae"
        else:
            video_name_suffix = f"{cfg.meta_cvae_cfg.kl_coeff}_{cfg.meta_cvae_cfg.latent_dim}_False"

        if cfg.record_states and states_dir is None:
            states_dir = f"states/{cfg.data.folder}_{cfg.eval.mode}/{cfg.skill_training.run_idx}_{video_name_suffix}__{cfg.multitask.task_id}"
            experiment_id = 0
            for path in Path(states_dir).glob('run_*'): 
                if not path.is_dir():
                    continue
                try:
                    folder_id = int(str(path).split('run_')[-1])
                    if folder_id > experiment_id:
                        experiment_id = folder_id
                except BaseException:
                    pass
            experiment_id += 1
            states_dir += f"/run_{experiment_id}"
            os.makedirs(states_dir, exist_ok=True)
            
        if not cfg.eval.testing:

            if video_dir is None:
                video_dir = f"videos/training/{cfg.data.folder}_{cfg.eval.mode}/{cfg.skill_training.run_idx}_{video_name_suffix}_task_{cfg.multitask.task_id}_{cfg.skill_subgoal_cfg.horizon}"

                experiment_id = 0
                for path in Path(video_dir).glob('run_*'): 
                    if not path.is_dir():
                        continue

                    try:
                        folder_id = int(str(path).split('run_')[-1])
                        if folder_id > experiment_id:
                            experiment_id = folder_id
                    except BaseException:
                        pass
                experiment_id += 1

                video_dir += f"_run_{experiment_id}"
                os.makedirs(video_dir, exist_ok=True)            
            video_name = f"{video_dir}/{modality_str}_demo_{demo}_video_{done}"
        else:
            if video_dir is None:
                video_dir = f"videos/testing/{cfg.data.folder}_{cfg.eval.mode}/{cfg.skill_training.run_idx}_{video_name_suffix}_task_{cfg.multitask.task_id}_{cfg.skill_subgoal_cfg.horizon}"

                if cfg.eval.singletask_baseline:
                    video_dir += "_baseline"
                experiment_id = 0
                for path in Path(video_dir).glob('run_*'):

                    if not path.is_dir():
                        continue
                    try:
                        folder_id = int(str(path).split('run_')[-1])
                        if folder_id > experiment_id:
                            experiment_id = folder_id
                    except BaseException:
                        pass

                experiment_id += 1

                video_dir += f"/run_{experiment_id}"

                os.makedirs(video_dir, exist_ok=True)
            video_name = f"{video_dir}/{modality_str}_demo_{i}_video_{done}"

        if cfg.record_states:

            state_file = h5py.File(f"{states_dir}/ep_{i}_{done}.hdf5", "w")

            state_file.attrs["env_name"] = env_name
            state_file.create_dataset("states", data=np.array(record_states))
            state_file.attrs["model_file"] = model_xml
            state_file.close()
            
        video_writer = imageio.get_writer(f"{video_name}.mp4", fps=60)
        for img in recorded_imgs:
            video_writer.append_data(img)
        video_writer.close()

        if i % 5 == 0 and cfg.verbose:
            print(f"Evaluated: {num_success}/{num_eval}")

        if cfg.eval.video:
            video_writer.close()

    print(f"Task: {cfg.multitask.training_task_id}, Baseline? : {cfg.eval.singletask_baseline}, Evaluated: {num_success}/{num_eval}")
            
    f.close()

if __name__ == "__main__":
    main()

