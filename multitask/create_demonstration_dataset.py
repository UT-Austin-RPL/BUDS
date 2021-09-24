"""
A convenience script to playback random demonstrations from
a set of demonstrations stored in a hdf5 file.

Arguments:
    --folder (str): Path to demonstrations
    --use-actions (optional): If this flag is provided, the actions are played back
        through the MuJoCo simulator, instead of loading the simulator states
        one by one.
    --visualize-gripper (optional): If set, will visualize the gripper site

Example:
    $ python playback_demonstrations_from_hdf5.py --folder ../models/assets/demonstrations/SawyerPickPlace/
"""

import os
import h5py
import argparse
import random
import numpy as np
import json

import robosuite
import xml.etree.ElementTree as ET

import time
import init_path
from robosuite_task_zoo.environments.manipulation import *
from robosuite.environments.manipulation.pick_place import PickPlaceCan
from robosuite.environments.manipulation.nut_assembly import NutAssemblySquare
import cv2
from PIL import Image
import robosuite.utils.macros as macros
import matplotlib.pyplot as plt
import robosuite.utils.transform_utils as T
from demonstration_tools import postprocess_model_xml

macros.IMAGE_CONVENTION = "opencv"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder",
        type=str,
        help="Path to your demonstration folder that contains the demo.hdf5 file, e.g.: "
             "'path_to_assets_dir/demonstrations/YOUR_DEMONSTRATION'"
    ),
    parser.add_argument(
        "--use-actions", 
        action='store_true',
    )
    parser.add_argument(
        "--use-camera-obs", 
        action='store_true',
    )
    parser.add_argument(
        '--dataset-path',
        type=str,
        default="demonstration_data/dataset/",
    )

    parser.add_argument(
        '--dataset-name',
        type=str,
        default="training_set",
    )

    parser.add_argument(
        '--no-contact',
        action='store_true'
    )

    parser.add_argument(
        '--no-proprio',
        action='store_true'
    )
    
    args = parser.parse_args()

    demo_path = args.folder
    hdf5_path = os.path.join(demo_path, "demo.hdf5")
    f = h5py.File(hdf5_path, "r")
    # env_name = "BlockStacking"
    env_name = f["data"].attrs["env"]
    print(env_name)
    env_info = json.loads(f["data"].attrs["env_info"])

    # env = eval(env_name)(
    #     **env_info,
    #     has_renderer=not args.use_camera_obs,
    #     has_offscreen_renderer=args.use_camera_obs,
    #     ignore_done=True,
    #     use_camera_obs=args.use_camera_obs,
    #     camera_names=["robot0_eye_in_hand",
    #                   "agentview",
    #                   # "frontview",
    #                   ],
    #     reward_shaping=True,
    #     control_freq=20,
    #     camera_heights=128,
    #     camera_widths=128
    # )
    # list of all demonstrations episodes
    demos = list(f["data"].keys())

    out_parent_dir = f"datasets/{env_name}_{args.dataset_name}"
    os.makedirs(out_parent_dir, exist_ok=True)
    hdf5_path = os.path.join(out_parent_dir, "demo.hdf5")
    print(hdf5_path)
    h5py_f = h5py.File(hdf5_path, "w")

    grp = h5py_f.create_group("data")
    grp.attrs["env"] = env_name

    env = eval(env_name)(
        **env_info,
        has_renderer=not args.use_camera_obs,
        has_offscreen_renderer=args.use_camera_obs,
        ignore_done=True,
        use_camera_obs=args.use_camera_obs,
        camera_names=["robot0_eye_in_hand",
                      "agentview",
                      # "frontview",
                      ],
        reward_shaping=True,
        control_freq=20,
        camera_heights=128,
        camera_widths=128,
    )
    
    
    total_len = 0
    demos = demos

    ft_states_list = []
    contact_states_list = []
    gripper_states_list = []
    joint_states_list = []
    proprio_states_list = []
    agentview_images_list = []
    eye_in_hand_images_list = []
    low_dim_states_list = []

    task_id_ep_mapping = {}
    for i in range(12):
        task_id_ep_mapping[i] = []

    prev_task_id = None
        
    for (i, ep) in enumerate(demos):
        print("Playing back random episode... (press ESC to quit)")

        # # select an episode randomly
        # read the model xml, using the metadata stored in the attribute for this episode
        model_xml = f["data/{}".format(ep)].attrs["model_file"]

        if prev_task_id != f[f"data/{ep}"].attrs["task_id"]:

            del env
            env_info["task_id"] = f[f"data/{ep}"].attrs["task_id"]
            env = eval(env_name)(
                **env_info,
                has_renderer=not args.use_camera_obs,
                has_offscreen_renderer=args.use_camera_obs,
                ignore_done=True,
                use_camera_obs=args.use_camera_obs,
                camera_names=["robot0_eye_in_hand",
                              "agentview",
                              # "frontview",
                              ],
                reward_shaping=True,
                control_freq=20,
                camera_heights=128,
                camera_widths=128,
            )
            prev_task_id = f[f"data/{ep}"].attrs["task_id"]
            print("Task: ", prev_task_id)

        # env.update_task(f[f"data/{ep}"].attrs["task_id"])
        env.reset()

        # if env_name == "PegInHoleEnv":
        #     xml = postprocess_model_xml(model_xml, {"agentview": {"pos": "0.7 0 1.45", "quat": "0.653 0.271 0.271 0.653"}})
        # elif env_name == "MediumPickPlace":
        xml = postprocess_model_xml(model_xml, {"robot0_eye_in_hand": {"pos": "0.05 0.0 0.049999", "quat": "0.0 0.707108 0.707108 0.0"}})            
        # else:
        #     xml = postprocess_model_xml(model_xml, {})

        if not args.use_camera_obs:
            env.viewer.set_camera(0)

        # load the flattened mujoco states
        # states = f["data/{}/states".format(ep)].value
        states = f["data/{}/states".format(ep)][()]
        actions = np.array(f["data/{}/actions".format(ep)][()])
        
        num_actions = actions.shape[0]

        init_idx = 0
        env.reset_from_xml_string(xml)
        env.sim.reset()        
        env.sim.set_state_from_flattened(states[init_idx])
        env.sim.forward()
        xml = env.sim.model.get_xml()

        contact_states = []

        # Last 32 force-torque reading
        ft_states = []
        # Do not repeat the force-torque recordings
        recent_ft_states = []
        proprio_states = []
        gripper_states = []
        joint_states = []
        low_dim_states = []
        agentview_image_names = []
        frontview_image_names = []
        eye_in_hand_image_names = []

        agentview_images = []
        eye_in_hand_images = []
        os.makedirs(f"datasets/{env_name}_{args.dataset_name}/ep_{i}", exist_ok=True)

        idx = 0
        for j, action in enumerate(actions):
            obs, reward, done, info = env.step(action)

            # for ee_force in info["history_ft"]:
            #     contact_states.append(np.linalg.norm(ee_force[:3] -
            #     env.ee_force_bias))

            if j < num_actions - 1:
                # ensure that the actions deterministically lead to the same recorded states
                state_playback = env.sim.get_state().flatten()
                # assert(np.all(np.equal(states[j + 1], state_playback)))
                err = np.linalg.norm(states[j + 1] - state_playback)

                if err > 0.01:
                    print(f"[warning] playback diverged by {err:.2f} for ep {ep} at step {j}")

            # Skip recording because the force sensor is not stable in
            # the beginning
            if j < 5:
                continue


            if not args.no_proprio:
                if "robot0_gripper_qpos" in obs:
                    gripper_states.append(obs["robot0_gripper_qpos"])

                joint_states.append(obs["robot0_joint_pos"])

                if env_info["controller_configs"]["type"] == "OSC_POSITION":
                    proprio_states.append(np.hstack((obs["robot0_eef_pos"], obs["robot0_gripper_qpos"])))
                else:
                    proprio_states.append(np.hstack((obs["robot0_eef_pos"], T.quat2axisangle(obs["robot0_eef_quat"]))))

            if not args.no_contact:
                contact_states.append(int(obs["robot0_contact"]))                
                ft_states.append(info["history_ft"].transpose())
                recent_ft_states.append(info["recent_ft"].transpose())

            low_dim_states.append(env.get_state_vector(obs))
            eye_in_hand_image_name = f"datasets/{env_name}_{args.dataset_name}/ep_{i}/eye_in_hand_{idx}.png"
            agentview_image_name = f"datasets/{env_name}_{args.dataset_name}/ep_{i}/agentview_{idx}.png"
            # frontview_image_name = f"datasets/{env_name}_{args.dataset_name}/ep_{i}/frontview_{idx}.png"
            idx += 1
            eye_in_hand_image_names.append(eye_in_hand_image_name)
            agentview_image_names.append(agentview_image_name)
            # frontview_image_names.append(frontview_image_name)

            # obs = env._get_observations()
            # contact_sparse_states.append(np.linalg.norm(env.robots[0].ee_force - env.ee_force_bias))

            if args.use_camera_obs:
                eye_in_hand_image = cv2.cvtColor(obs["robot0_eye_in_hand_image"], cv2.COLOR_BGR2RGB)
                cv2.imwrite(eye_in_hand_image_name, eye_in_hand_image)
                agentview_image = cv2.cvtColor(obs["agentview_image"], cv2.COLOR_BGR2RGB)
                cv2.imwrite(agentview_image_name, agentview_image)
                # cv2.imwrite(frontview_image_name, cv2.cvtColor(obs["frontview_image"], cv2.COLOR_BGR2RGB))

                assert(np.all(obs["agentview_image"] == np.array(Image.open(agentview_image_name))))
                
                agentview_images.append(obs["agentview_image"].transpose(2, 0, 1))
                eye_in_hand_images.append(obs["robot0_eye_in_hand_image"].transpose(2, 0, 1))
                
               
            else:
                env.render()

        actions = actions[5:]
        assert(len(actions) == len(agentview_images))
        print(len(actions))
        # post_process contact

        if not args.no_contact:
            for j in range(len(contact_states)):
                if j == len(contact_states) - 1 and contact_states[j-1] and not contact_states[j]:
                    contact_states[j] = True
                elif not contact_states[j] and contact_states[j-1] and contact_states[j+1]:
                    contact_states[j] = True
                elif j - 2 >= 0 and j + 2 < len(contact_states):
                    if contact_states[j-2] and contact_states[j+2]:
                        contact_states[j-1] = True
                        contact_states[j+1] = True
                        contact_states[j] = True
                elif j - 5 >= 0 and j + 5 < len(contact_states):
                    if np.sum(contact_states[j-5:j+5].astype(np.int8)) > 6:
                        contact_states[j] = True

            ft_states_list.append(np.stack(ft_states, axis=0))
            contact_states_list.append(np.stack(contact_states, axis=0))

        if not args.no_proprio:
            proprio_states_list.append(np.stack(proprio_states, axis=0))
        agentview_images_list.append(np.stack(agentview_images, axis=0))
        eye_in_hand_images_list.append(np.stack(eye_in_hand_images, axis=0))
        low_dim_states_list.append(np.stack(low_dim_states, axis=0))

        ep_data_grp = grp.create_group(f"ep_{i}")

        if not args.no_contact:
            ep_data_grp.create_dataset("ft_states", data=np.stack(ft_states, axis=0))
            ep_data_grp.create_dataset("recent_ft_states", data=np.stack(recent_ft_states, axis=0))        
            ep_data_grp.create_dataset("contact_states", data=np.stack(contact_states, axis=0))
        if not args.no_proprio:
            ep_data_grp.create_dataset("gripper_states", data=np.stack(gripper_states, axis=0))
            ep_data_grp.create_dataset("joint_states", data=np.stack(joint_states, axis=0))
            ep_data_grp.create_dataset("proprio_states", data=np.stack(proprio_states, axis=0))
        ep_data_grp.create_dataset("agentview_image_names", data=agentview_image_names)
        # ep_data_grp.create_dataset("frontview_image_names", data=frontview_image_names)
        ep_data_grp.create_dataset("eye_in_hand_image_names", data=eye_in_hand_image_names)
        ep_data_grp.create_dataset("actions", data=actions)
        ep_data_grp.create_dataset("gt_states", data=states)
        ep_data_grp.create_dataset("low_dim_states", data=np.stack(low_dim_states, axis=0))

        ep_data_grp.attrs["model_file"] = model_xml
        ep_data_grp.attrs["task_id"] = env.task_id

        ep_data_grp.attrs["init_state"] = states[init_idx]

        task_id_ep_mapping[env.task_id].append(i)

        total_len += len(agentview_image_names)
        # np.savez(
        #     os.path.join(f"{out_parent_dir}/ep_{i}/states.npz"),
        #     ft_states=np.stack(ft_states, axis=0),
        #     proprio_states=np.stack(proprio_states, axis=0),
        #     eye_in_hand_image_names=eye_in_hand_image_names,
        #     agentview_image_names=agentview_image_names
        # )

        # print(np.stack(ft_states, axis=0).shape)
        # print(np.stack(proprio_states, axis=0).shape)

        # plt.plot(contact_states)
        # plt.plot(np.array(recent_ft_states)[:, :, -1] * 0.1)
        # # # plt.plot(contact_sparse_states)
        # plt.show()

    if not args.no_contact:
        grp.create_dataset("ft_states", data=np.vstack(ft_states_list))
        grp.create_dataset("contact_states", data=np.hstack(contact_states_list)[..., np.newaxis])
    if not args.no_proprio:
        grp.create_dataset("proprio_states", data=np.vstack(proprio_states_list))
    grp.create_dataset("agentview_images", data=np.vstack(agentview_images_list))
    grp.create_dataset("eye_in_hand_images", data=np.vstack(eye_in_hand_images_list))
    grp.create_dataset("low_dim_states", data=np.concatenate(low_dim_states_list, axis=0))
    print(np.vstack(agentview_images_list).shape)
    grp.attrs["num_eps"] = len(demos)
    grp.attrs["len"] = total_len
    print("Task ep mapping:", task_id_ep_mapping)

    task_grp = grp.create_group("task")
    for (task_id, ep_mapping) in task_id_ep_mapping.items():
        task_grp.attrs[f"{task_id}"] = ep_mapping

    h5py_f.close()        
    f.close()
