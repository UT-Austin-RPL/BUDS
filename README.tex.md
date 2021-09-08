# Bottom-up Discovery of Reusable Sensorimotor Skills from Unstructured Demonstrations (BUDS)

## Dependencies



## Data



## Instructions on running scripts


## Implementation Details

### Implementation for Multi-sensory Representation Learning
We use the same convolutional network structures as the models from Lee et al.~\cite{lee2020making}, except that we do not have skip connections from the encoder to the decoder parts. We choose $32$ for the latent dimension of the fused representation. For training the fusion model for each task, we use $1000$ training epochs, with a learning rate of $0.001$,  batch size of 128.


### Hyperparameters for Skill Segmentatation.

We present the hyperparameters for the unsupervised clustering step. The maximum number of clusters for each task is: $6$ for \tooluse{} and \hammer{}, $8$ for \kitchen{} and \realrobot{}, $10$ for \multitask{}. And the stopping criteria of the bread-first search is the number of segments of mid-level segments are more than twice the maximum number of clusters. We also use a minimum length threshold to reject a small cluster, the number we choose is: $30$ for \tooluse{}, \realrobot{}, $35$ for \hammer{}, $20$ for \multitask{}. In this work, these values of hyperparameters are tuned heuristically , and how to extend to an end-to-end method is another future direction to look at.


### Model Details for Sensorimotor Policies
BUDS focuses on learning closed-loop sensorimotor skills. The input to each skill is the observations from robot sensors and the latent subgoal vector $\omega_{t}$. Specifically, the observations consist of two RGB images ($128 \times 128$) from the workspace camera and the eye-in-hand camera, and the proprioception of joint and gripper states. For encoding visual inputs, we use ResNet-18~\cite{he2016deep} as the visual encoder, followed by Spatial Softmax~\cite{finn2016deep} to extract keypoints of the feature maps. We then concatenate keypoints with proprioception (joint angles and past five frames of gripper states~\cite{zhang2018deep}), and concatenated vectors are passed through fully connected layers with LeakyReLU activation, outputting end-effector motor commands. The subgoal encoder $E_k$ is a ResNet-18 module with spatial softmax module, and $E_k$ only takes the image from the workspace camera of the subgoal state in demonstration data as inputs. The meta controller $\pi_{H}$ takes the image of current observation from the workspace camera as input, and the visual encoder in $\pi_{H}$ is also a ResNet-18 module. For all ResNet-18 modules, we remove the last two layers compared to the original design, giving us $4\times4$ feature maps. For low-level robot controllers, we use position-based Operational Space Controller (OSC)~\cite{khatib1987unified} with a binary controller for the parallel-jaw gripper, and the controllers take commands at $20$ Hz. During evaluation, we choose the meta controller to operate at $4$ Hz while sensorimotor skills operate at $20$ Hz. 

We choose the dimension for subgoal vector $\omega_t$ to be $32$, the number of 2D keypoints from the output of Spatial Softmax layer to be $64$. We choose $H=30$ for all single-task environments (Both simulation and real robots). We choose $H=20$ for the multitask environment  \multitask{}. This is because skills are relatively short in each task in \multitask{} domain compared to all single-task environments. 

### Training Details for Sensorimotor Policies
To increase the generalization ability of the model, we apply data augmentation~\cite{kostrikov2020image} to images for both training skills and meta controllers. To further increase the robustness of policies $\pi^{(k)}_{L}$, we also add some noise from Gaussian distribution with a standard deviation of $0.1$. 
 
 For all skills, we train for $2001$ epochs with a learning rate of $0.0001$, and the loss function we use is $\ell_{2}$ loss. We use two layers ($300$, $400$ hidden units for each layer) for the fully connected layers in all sing-task environments, while three layers ($300$, $300$, $400$) hidden units for each layer for fully connected layers in \multitask{} domain. For meta controllers, we train $1001$ epochs in all simulated single-task environments, $2001$ epochs in \multitask{} domain, and $3001$ epochs in \realrobot{}. For kl coefficients during cVAE training, we choose $0.005$ for \tooluse{}, \hammer{}, and $0.01$ for all other environments. 
 












## References

