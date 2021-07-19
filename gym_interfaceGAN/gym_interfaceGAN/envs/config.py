import torch

gan_model_name = 'pggan_celebahq'
batch_size = 1
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu") 
# device = torch.device("cpu")
latent_dim = 512
basis_path = "/home/gopal/Desktop/shubhamCode/shubham/baselines/gym_interfaceGAN/gym_interfaceGAN/envs/library/basis_vectors.npy"
step_basis = 0.001
hyperplane_path = "/home/gopal/Desktop/shubhamCode/shubham/baselines/gym_interfaceGAN/gym_interfaceGAN/envs/library/InterFaceGAN/boundaries/pggan_celebahq_gender_boundary.npy"
hyperplane_loss_parameter = 0.03 
relative_reward_hyperparameter = 100
eta_parameter = 1
lambda_parameter = 0.01
target_sim = -10
save_adds = "/home/gopal/Desktop/shubhamCode/shubham/baselines/images/continuos_1"
max_step = 200
continuous = True 
save_current_state = True

update_buffer_k = 1000000000
relative_KL_lambda = 100

#CUDA_VISIBLE_DEVICES="1,0" python -m baselines.run --alg=ppo2 --env=InterfaceGAN-v0 --network=mlp --num_timesteps=1e5 --save_path=./models/interface_test_100_step --log_path=./logs/ppo2_100_step --nsteps=128 

##Test
#CUDA_VISIBLE_DEVICES="1,0" python -m baselines.run --alg=ppo2 --env=InterfaceGAN-v0 --network=mlp --num_timesteps=0 --load_path=./models/interface_test_100_step_continuous --play 