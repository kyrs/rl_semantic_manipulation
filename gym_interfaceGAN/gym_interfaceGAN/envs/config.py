import numpy as np
import torch
from os import mkdir
from os.path import  isdir
import time


def TrueXor(*args):
	## flag to check if only one element in a set is  true
    return sum(args) == 1


############ Important Flag (set it false during testing) ####################
isTrain  =  True

if not isTrain:
	print("==========RUNNING TESTING ===========")
else:
	print("==========RUNNING TRAINING===========")

# input("Press enter to confirm the process or abort and modify the config \n")



################## Model based flag ###################
gan_model_name = 'pggan_celebahq'
batch_size = 1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
latent_dim = 512

###### flags for Typical set #################
typ_set_eps = 6
typ_set_punishment = -25 ## part of reward
assert(typ_set_punishment < 0)

############ manifold approximation flag ###########
step_size = 0.01
weight_clip = 0.5
assert(weight_clip>0)
polyak_param = 0.3
max_step = 10
not_reach_goal_punishment = -1

######## face flag ##################

basis_list = None
age_model_path = None 
eyeglass_model_path = None
gender_model_path = None
smile_model_path = None

###################################
use_age = True
age_reward = None
if use_age:
	age_reward = 2
	age_model_path = "/home/kumar/RL_env/attribute_model/age/age-stage-128-50"
	age_basis_file = "/home/kumar/RL_env/gym_interfaceGAN/gym_interfaceGAN/envs/library/InterFaceGAN/boundaries/pggan_celebahq_age_boundary.npy"
	age_basis = np.load(age_basis_file)
	assert("age" in age_basis_file.split("_"))
	basis_list = np.vstack((age_basis))
#####################################

####### eyeglasses flag #############
use_eyeglass = False
eyeglass_reward = None

if use_eyeglass:
	eyeglass_reward = 2
	eyeglass_model_path = "/home/kumar/RL_env/attribute_model/eyeglasses/eyeglasses_model-1"
	eyeglasses_basis_file = "/home/kumar/RL_env/gym_interfaceGAN/gym_interfaceGAN/envs/library/InterFaceGAN/boundaries/pggan_celebahq_eyeglasses_boundary.npy"
	eyeglasses_basis = np.load(eyeglasses_basis_file)
	assert("eyeglasses" in eyeglasses_basis_file.split("_"))
	basis_list = np.vstack((eyeglasses_basis))
####################################


#######  gender flag #############
use_gender = False
gender_reward = None

if use_gender:
	gender_reward = 2
	gender_model_path = "/home/kumar/RL_env/attribute_model/gender/gender_model-1"
	gender_basis_file = "/home/kumar/RL_env/gym_interfaceGAN/gym_interfaceGAN/envs/library/InterFaceGAN/boundaries/pggan_celebahq_gender_boundary.npy"
	gender_basis = np.load(gender_basis_file)
	assert("gender" in gender_basis_file.split("_"))
	basis_list = np.vstack((gender_basis))
####################################


####### smile flag #############
use_smile = False
smile_reward = None

if use_smile:
	smile_reward = 2
	smile_model_path = "/home/kumar/RL_env/attribute_model/Smiling/smiling_model-1"
	smile_basis_file = "/home/kumar/RL_env/gym_interfaceGAN/gym_interfaceGAN/envs/library/InterFaceGAN/boundaries/pggan_celebahq_smile_boundary.npy"
	smile_basis = np.load(smile_basis_file)
	assert("smile" in smile_basis_file.split("_"))
	basis_list = np.vstack((smile_basis))
####################################

assert(TrueXor( use_smile, use_gender, use_age, use_eyeglass))

############# Identity comparision metric #################
max_identity_limit = 900# P2
prefered_identity_limit = 750 # P1
identity_punishment = -25 ## part of reward
assert(identity_punishment <0) 

###########################Dataset for training and Testing#########################################

reset_train_sample = "/home/kumar/RL_env/attribute_model/random_reset_train.npy"
# reset_test_sample = "/home/kumar/RL_env/attribute_model/random_reset_test.npy"
reset_test_sample = "/mnt/hdd1/shubham/generate_images_for_testing/good_example.npy"


semanticTesting = None
save_adds = None
run_batch_test = None
if not isTrain:
	timestr = time.strftime("%Y%m%d-%H%M%S")
	save_adds = "/home/kumar/RL_env/imgs/ijcai_exp_1_age_reward_neg_50/test3/set1/" + timestr
	run_batch_test = True
	if isdir (save_adds):
		pass
	else:
		mkdir(save_adds)

	semanticTesting = 1 # set 0/1 based on the test of interest 
else:
	pass

# CUDA_VISIBLE_DEVICES="0,1" python -m baselines.run --alg=ppo2 --env=InterfaceGAN-v0 --network=mlp --num_timesteps=1e5 --save_path=./models/test --log_path=./logs/test --ent_coef=0.1 --num_hidden=32 --num_layers=3 
## Test/home/kumar/RL_env/IJCAI/test/ppo2_new/face0
# CUDA_VISIBLE_DEVICES="0,1" python -m baselines.run --alg=ppo2 --env=InterfaceGAN-v0 --network=mlp --num_timesteps=0 --load_path=./models/interface_test_100_step --play 
