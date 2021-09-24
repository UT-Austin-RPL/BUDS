# Bottom-up Discovery of Reusable Sensorimotor Skills from Unstructured Demonstrations (BUDS)

## Dependencies
- Robosuite
- pytorch
- sklearn



## Environments
We implemented four environments, three single-task ones and a
multi-task one. They are <tt>Hammer-Place</tt>, <tt>Tool-Use</tt>,
<tt>Kitchen</tt>, <tt>Multitask-Kitchen</tt>. Please refer to
[robosuite-task-zoo](https://github.com/ARISE-Initiative/robosuite-task-zoo). 


## Data
We provide the demonstration data we have.


## Instructions on running scripts


### Demonstration collection

1. Single-task


``` shell
python data_collection/collect_demonstration_script.py --num-demonstration 100  --pos-sensitivity 1.0 --controller OSC_POSITION --environment ToolUseEnvV1
```


1. Multi-task

``` shell
python multitask/collect_demonstration_script.py --num-demonstration 40 --pos-sensitivity 1.0 --controller OSC_POSITION --environment MultitaskKitchenDomain --task-id 1
```


### Create dataset


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
python skill_discovery/hierarchical_agglomoration.py data=kitchen_full
```

2. Multi-task

``` shell
	python multitask/hierarchical_agglomoration.py data=multitask_kitchen_domain
```

#### Spectral Clustering for K partitions
1. Single Task

``` shell
python skill_disocvery/agglomoration_script.py data=kitchen_full agglomoration.visualization=true repr.modalities="[agentview, eye_in_hand, proprio]" agglomoration.min_len_thresh=30 agglomoration.K=8 agglomoration.segment_scale=2 agglomoration.scale=0.01
```

2. Multi-task

``` shell
python skill_classification/agglomoration_script.py data=kitchen_full agglomoration.visualization=true repr.modalities="[agentview, eye_in_hand, proprio]" agglomoration.min_len_thresh=30 agglomoration.K=8 agglomoration.segment_scale=2 agglomoration.scale=0.01
```

### Policy Learning



#### Skill policies
1. Single-task 

``` shell
python policy_learning/train_subskills_hydra.py skill_training.agglomoration.K=$2 skill_training.run_idx=2 data=kitchen_full skill_training.batch_size=128 skill_subgoal_cfg.visual_feature_dimension=32 skill_training.subtask_id="[$3]" skill_training.data_modality="[image, proprio]" skill_training.lr=1e-4 skill_training.num_epochs=2001 repr.z_dim=32
```

2. Multi-task

``` shell
python multitask/train_subskills.py skill_training.agglomoration.K=$2 skill_training.run_idx=0 data=multitask_kitchen_domain skill_training.batch_size=128 skill_subgoal_cfg.visual_feature_dimension=32 skill_training.subtask_id="[$3]" skill_training.data_modality="[image, proprio]" skill_training.lr=1e-4 skill_training.num_epochs=2001 repr.z_dim=32 skill_training.policy_layer_dims="[300, 300, 400]" skill_subgoal_cfg.horizon=20
```


#### Meta controllers

1. Single Task
``` shell
python policy_learning/train_meta_controller_hydra.py skill_training.agglomoration.K=5 skill_training.run_idx=2 data=hammer_sort skill_training.batch_size=128 skill_subgoal_cfg.visual_feature_dimension=32 skill_training.data_modality="[image, proprio]" skill_training.lr=1e-4 skill_training.num_epochs=2001 meta_cvae_cfg.latent_dim=64  meta.use_eye_in_hand=False meta.random_affine=True meta_cvae_cfg.kl_coeff=0.005 repr.z_dim=32
```

2. Multi-task
``` shell
python multitask/train_meta_controller.py skill_training.agglomoration.K=$2 skill_training.run_idx=0 data=multitask_kitchen_domain skill_training.batch_size=128 skill_subgoal_cfg.visual_feature_dimension=32 skill_training.subtask_id="[-1]" skill_training.data_modality="[image, proprio]" skill_training.lr=1e-4 skill_training.num_epochs=4001 repr.z_dim=32 meta_cvae_cfg.latent_dim=64  meta.use_eye_in_hand=False meta.random_affine=True meta_cvae_cfg.kl_coeff=$4 meta.num_epochs=2001 multitask.task_id=$3 skill_training.policy_layer_dims="[300, 300, 400]" skill_subgoal_cfg.horizon=20
```

### Evaluation
1. ToolUse

``` shell
python goal_skill_learning/eval_task.py skill_training.data_modality="[image, proprio]" data=tool_use_v1 skill_training.agglomoration.K=4 meta_cvae_cfg.latent_dim=64 repr.z_dim=32 skill_training.run_idx=0 eval.meta_freq=5 eval.max_steps=1500 meta_cvae_cfg.kl_coeff=0.005  meta.use_spatial_softmax=false meta.random_affine=true eval.testing=true eval.mode="ours"
```



1. Hammer

``` shell
python goal_skill_learning/eval_task.py skill_training.data_modality="[image, proprio]" data=hammer_sort skill_training.agglomoration.K=4 meta_cvae_cfg.latent_dim=64 repr.z_dim=32 skill_training.run_idx=0 eval.meta_freq=5 eval.max_steps=1500 meta_cvae_cfg.kl_coeff=0.005  meta.use_spatial_softmax=false meta.random_affine=true eval.testing=true eval.mode="ours"
```


1. Kitchen

``` shell
python goal_skill_learning/eval_task.py skill_training.data_modality="[image, proprio]" data=kitchen_full skill_training.agglomoration.K=6 meta_cvae_cfg.latent_dim=64 repr.z_dim=32 skill_training.run_idx=0 eval.meta_freq=5 eval.max_steps=5000 meta_cvae_cfg.kl_coeff=0.01  meta.use_spatial_softmax=false meta.random_affine=true eval.testing=true eval.mode="ours"
```


1. Multitask Kitchen

``` shell
python multitask/eval_task.py skill_training.data_modality="[image, proprio]" data=multitask_kitchen_domain skill_training.agglomoration.K=8 meta_cvae_cfg.latent_dim=64 repr.z_dim=32 skill_training.run_idx=0 eval.meta_freq=5 eval.max_steps=2500 meta_cvae_cfg.kl_coeff=0.01  meta.use_spatial_softmax=false meta.random_affine=true eval.testing=true multitask.task_id=0 skill_subgoal_cfg.horizon=20 multitask.training_task_id=0 verbose=false
```

## Implementation
[Details](./implementation_details.ipynb)


## Citing


```

```
