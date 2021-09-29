# Bottom-Up Skill Discovery from Unsegmented Demonstrations for Long-Horizon Robot Manipulation
[Yifeng Zhu](https://www.cs.utexas.edu/~yifengz), [Peter Stone](https://www.cs.utexas.edu/~pstone), [Yuke Zhu](https://www.cs.utexas.edu/~yukez/)


[Project](https://ut-austin-rpl.github.io/rpl-BUDS/)  <!-- | [arxiv](http://arxiv.org/abs/2109.13841)  -->


## Introduction
We tackle real-world long-horizon robot manipulation tasks through skill discovery. We present a bottom-up approach to learning a library of reusable skills from unsegmented demonstrations and use these skills to synthesize prolonged robot behaviors. Our method starts with constructing a hierarchical task structure from each demonstration through agglomerative clustering. From the task structures of multi-task demonstrations, we identify skills based on the recurring patterns and train goal-conditioned sensorimotor policies with hierarchical imitation learning. Finally, we train a meta controller to compose these skills to solve long-horizon manipulation tasks. The entire model can be trained on a small set of human demonstrations collected within 30 minutes without further annotations, making it amendable to real-world deployment. We systematically evaluated our method in simulation environments and on a real robot. Our method has shown superior performance over state-of-the-art imitation learning methods in multi-stage manipulation tasks. Furthermore, skills discovered from multi-task demonstrations boost the average task success by 8% compared to those discovered from individual tasks.

## Dependencies
- [robosuite](https://github.com/ARISE-Initiative/robosuite/)
- pytorch
- sklearn

## Simulation Environments
We implemented four simulated environments for evaluation, including three single-task ones and a
multi-task one. They are <tt>Hammer-Place</tt>, <tt>Tool-Use</tt>,
<tt>Kitchen</tt>, <tt>Multitask-Kitchen</tt>. The implementations of these environments can be found in 
[robosuite-task-zoo](https://github.com/ARISE-Initiative/robosuite-task-zoo). 


## Demonstration Data
We provide the demonstration data we used for our experiments. Please download the datasets [here](https://utexas.box.com/shared/static/om0pegpm0hdi12clydau36d3vy0yz516.zip), and extract them in the root folder of this repo, and name the folder `datasets`.


## Instructions on running scripts


### Demonstration collection

We collect demonstrations using 3D spacemouse. The following commands
show how we can collect data in a single-task environment or a
multi-task environment. For multi-task environment, you need to
specify another variable, `task-id`, which is mapped to indiividual
task variant.

1. Single-task


```shell
python data_collection/collect_demonstration_script.py --num-demonstration 100  --pos-sensitivity 1.0 --controller OSC_POSITION --environment ToolUseEnv
```


1. Multi-task

``` shell
python multitask/collect_demonstration_script.py --num-demonstration 40 --pos-sensitivity 1.0 --controller OSC_POSITION --environment MultitaskKitchenDomain --task-id 1
```


### Create dataset

After demonstrations are collected, we want to create a dataset in
hdf5 file format for skill discovery and policy learning.


``` shell
python multitask/create_demonstration_dataset.py --folder demonstration_data/MultitaskKitchenDomain_training_set_0 --dataset-name training_set_0 --use-actions --use-camera-obs 
```

1. Important keys
   - states
   - actions
   - task-id for every ep


### Training multi-sensory representation

```
python multisensory_repr/train_multimodal.py
```

### Skill segmentations
#### Hierarchical Agglomerative Clustering
1. Single Task
``` shell
python skill_discovery/hierarchical_agglomoration.py data=kitchen
```

2. Multi-task

``` shell
python multitask/hierarchical_agglomoration.py data=multitask_kitchen_domain
```

#### Spectral Clustering for K partitions
1. Single Task

``` shell
python skill_disocvery/agglomoration_script.py data=kitchen agglomoration.visualization=true repr.modalities="[agentview, eye_in_hand, proprio]" agglomoration.min_len_thresh=30 agglomoration.K=8 agglomoration.segment_scale=2 agglomoration.scale=0.01
```

2. Multi-task

``` shell
python skill_discovery/agglomoration_script.py data=kitchen agglomoration.visualization=true repr.modalities="[agentview, eye_in_hand, proprio]" agglomoration.min_len_thresh=30 agglomoration.K=8 agglomoration.segment_scale=2 agglomoration.scale=0.01
```

### Policy Learning



#### Skill policies
1. Single-task 

``` shell
python policy_learning/train_subskills.py skill_training.agglomoration.K=$2 skill_training.run_idx=2 data=kitchen skill_training.batch_size=128 skill_subgoal_cfg.visual_feature_dimension=32 skill_training.subtask_id="[$3]" skill_training.data_modality="[image, proprio]" skill_training.lr=1e-4 skill_training.num_epochs=2001 repr.z_dim=32
```

2. Multi-task

``` shell
python multitask/train_subskills.py skill_training.agglomoration.K=$2 skill_training.run_idx=0 data=multitask_kitchen_domain skill_training.batch_size=128 skill_subgoal_cfg.visual_feature_dimension=32 skill_training.subtask_id="[$3]" skill_training.data_modality="[image, proprio]" skill_training.lr=1e-4 skill_training.num_epochs=2001 repr.z_dim=32 skill_training.policy_layer_dims="[300, 300, 400]" skill_subgoal_cfg.horizon=20
```


#### Meta controllers

1. Single Task
``` shell
python policy_learning/train_meta_controller.py skill_training.agglomoration.K=5 skill_training.run_idx=2 data=hammer_place skill_training.batch_size=128 skill_subgoal_cfg.visual_feature_dimension=32 skill_training.data_modality="[image, proprio]" skill_training.lr=1e-4 skill_training.num_epochs=2001 meta_cvae_cfg.latent_dim=64  meta.use_eye_in_hand=False meta.random_affine=True meta_cvae_cfg.kl_coeff=0.005 repr.z_dim=32
```

2. Multi-task
``` shell
python multitask/train_meta_controller.py skill_training.agglomoration.K=$2 skill_training.run_idx=0 data=multitask_kitchen_domain skill_training.batch_size=128 skill_subgoal_cfg.visual_feature_dimension=32 skill_training.subtask_id="[-1]" skill_training.data_modality="[image, proprio]" skill_training.lr=1e-4 skill_training.num_epochs=4001 repr.z_dim=32 meta_cvae_cfg.latent_dim=64  meta.use_eye_in_hand=False meta.random_affine=True meta_cvae_cfg.kl_coeff=$4 meta.num_epochs=2001 multitask.task_id=$3 skill_training.policy_layer_dims="[300, 300, 400]" skill_subgoal_cfg.horizon=20
```

### Evaluation
1. ToolUse

``` shell
python eval_scripts/eval_task.py skill_training.data_modality="[image, proprio]" data=tool_use skill_training.agglomoration.K=4 meta_cvae_cfg.latent_dim=64 repr.z_dim=32 skill_training.run_idx=0 eval.meta_freq=5 eval.max_steps=1500 meta_cvae_cfg.kl_coeff=0.005  meta.use_spatial_softmax=false meta.random_affine=true eval.testing=true eval.mode="BUDS"
```

1. HammerPlace

```shell
python eval_scripts/eval_task.py skill_training.data_modality="[image, proprio]" data=hammer_place skill_training.agglomoration.K=4 meta_cvae_cfg.latent_dim=64 repr.z_dim=32 skill_training.run_idx=0 eval.meta_freq=5 eval.max_steps=1500 meta_cvae_cfg.kl_coeff=0.005  meta.use_spatial_softmax=false meta.random_affine=true eval.testing=true eval.mode="BUDS"
```

1. Kitchen

```  shell
python eval_scripts/eval_task.py skill_training.data_modality="[image, proprio]" data=kitchen skill_training.agglomoration.K=6 meta_cvae_cfg.latent_dim=64 repr.z_dim=32 skill_training.run_idx=0 eval.meta_freq=5 eval.max_steps=5000 meta_cvae_cfg.kl_coeff=0.01  meta.use_spatial_softmax=false meta.random_affine=true eval.testing=true eval.mode="BUDS"
```


1. Multitask Kitchen

``` shell
python multitask/eval_task.py skill_training.data_modality="[image, proprio]" data=multitask_kitchen_domain skill_training.agglomoration.K=8 meta_cvae_cfg.latent_dim=64 repr.z_dim=32 skill_training.run_idx=0 eval.meta_freq=5 eval.max_steps=2500 meta_cvae_cfg.kl_coeff=0.01  meta.use_spatial_softmax=false meta.random_affine=true eval.testing=true multitask.task_id=0 skill_subgoal_cfg.horizon=20 multitask.training_task_id=0 verbose=false
```

## Implementation Details
Please see [this page](implementation_details.ipynb) for more information about our implementation details including training procedures and hyperparameters.
