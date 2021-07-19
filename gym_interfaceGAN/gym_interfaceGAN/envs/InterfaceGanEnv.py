import numpy as np
import os
import sys
from collections import deque
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding
import torch
sys.path.insert(0, "/home/gopal/Desktop/shubhamCode/shubham/baselines/gym_interfaceGAN/gym_interfaceGAN/envs")

from PIL import Image
from library.InterFaceGAN.models.model_settings import MODEL_POOL
from library.InterFaceGAN.models.pggan_generator import PGGANGenerator
from library.InterFaceGAN.models.stylegan_generator import StyleGANGenerator
from library.InterFaceGAN.utils.logger import setup_logger
from library.InterFaceGAN.utils.manipulator import linear_interpolate
from library.identity_loss import FaceNet, AlexNet
import config
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt


class InterfaceGanEnv(gym.Env):
    def __init__(self, gan_model_name=config.gan_model_name, batch_size=config.batch_size, device=config.device, 
                    latent_dim=config.latent_dim,target_sim=config.target_sim,
                    step_basis=config.step_basis, basis_path=config.basis_path, hyperplane_path=config.hyperplane_path, 
                    hyperplane_loss_parameter=config.hyperplane_loss_parameter,lambda_parameter=config.lambda_parameter,
                    eta_parameter=config.eta_parameter, relative_reward_hyperparameter=config.relative_reward_hyperparameter,
                    continuous=config.continuous, save_current_state_flag=config.save_current_state, update_buffer_k=config.update_buffer_k,
                    relative_KL_lambda=config.relative_KL_lambda):
        ## intializing all the model based parameters 
        super(InterfaceGanEnv, self).__init__()

        self.model_name = gan_model_name
        self.batch_size = batch_size
        self.device = device

        ## loading  the model
        self.model = self._load_model()
        self.latent_dim = latent_dim 
        self.face_model = AlexNet(device=self.device)
        self.basis_path = basis_path 
        self.step_basis = step_basis

        self.lambda_parameter = lambda_parameter
        self.eta_parameter = eta_parameter
        self.relative_reward_hyperparameter = relative_reward_hyperparameter

        self.hyperplane_path = config.hyperplane_path  
        self.hyperplane_loss_parameter = hyperplane_loss_parameter      
        self.act_list = []
        self.target_sim = target_sim
        self.continuous = continuous
        self.save_current_state_flag = save_current_state_flag

        ##  loading the basis vector and corresponding negation of it

        if not self.continuous:
            basis_vec = np.load(self.basis_path)
            for basis in basis_vec:
                self.act_list.append(basis)
                self.act_list.append(-1*basis)

            assert(len(self.act_list) == 2*len(basis_vec))
            assert(np.sum(self.act_list) == 0.0)        

            ## defining the data structuture of openai
            self.action_space = spaces.Discrete(2*len(basis_vec))

        else:
            self.action_space = spaces.Box(low=-1.5, high=1.5, dtype=np.float32, shape=(self.latent_dim,))


        self.observation_space = spaces.Box(low=-3, high=3, dtype=np.float32, shape=(self.latent_dim*2,)) 

        ## loading the hyperplane
        self.hyperplane = self._load_hyperplane( self.hyperplane_path )

        ## keeping a track of current state
        self.current_state = None
        self.base_states = None 
        self.base_image = None
        self.cnt = None 
        self.step_list = []
        self.update_buffer_k = update_buffer_k
        self.relative_KL_lambda = relative_KL_lambda
        self.step_count = -1
        self.buffer = deque(maxlen=self.update_buffer_k)


    def step(self, a):
        if not self.continuous:
            basis_manip = self.act_list[a]
        else:
            basis_manip = a

        old_state = self.current_state
        new_state = old_state + self.step_basis * basis_manip
       
       ########### CODE FOR MAPPING GENEREATED VEC BACK TO NORMAL ANNULUS ########
        # new_state = self.model.preprocess(new_state)
       ######################################################################### 
        new_state_images = self._batch_face_image_sampler(new_state)
        
        self.current_state = new_state

        assert( not np.array_equal(self.current_state,old_state))
        ## extracting facial features 

        new_feature = self.face_model.get_feature_vector(new_state_images)
        reward_val, iden_loss, hyp_loss = self._reward_score( face_feature = new_feature, z_feature_gen = new_state)

        # if(self.step_count % self.update_buffer_k == 0):
        self.buffer.append(new_state)
        # bring back the dimensions to (batch_size, update_buffer_k, len(new_state))
        buffered_states = np.transpose(np.array(list(self.buffer)), (1, 0, 2))
        mu = torch.from_numpy(buffered_states.mean(1))
        log_var = torch.from_numpy(np.log(buffered_states.var(1)))
        kl_loss = self._kl_div_loss(mu, log_var).item()
        print('kl_loss', kl_loss)
        print('reward_val', reward_val)
        reward_val = reward_val - kl_loss
        print('reward_val', reward_val)
        # self.buffer.clear()
        # else:
        #     self.buffer.append(new_state)

        if reward_val > self.target_sim :
            done = True 
        else:
            done = False

        obs = np.concatenate((new_state, self.base_states), axis=None)
        self.step_count += 1
        return obs, reward_val, done, {'episode':{'r':iden_loss, 'l': hyp_loss}}


    @property
    def _n_actions(self):
        if not self.continuous:
            return len(self.act_list)
        else:
            return self.latent_dim


    def reset(self):
        self.buffer.clear()

        states = self.model.easy_sample(self.batch_size)
        # current_state = self.model.easy_sample(self.batch_size) 
        current_state = states
        self.base_states = states
        self.cnt = 0
        image = self._batch_face_image_sampler(states)
        self.base_feature = self.face_model.get_feature_vector(image)
        self.base_image = image
        self.current_state = current_state 

        self.buffer.append(current_state)
        self.step_count = 1

        obs = np.concatenate((current_state, self.base_states), axis=None)
        self.step_list = []
        return obs


    def close(self):
        self.current_state = None
        self.base_states = None 
        self.base_image = None
        return 
        


    def render(self, mode = 'save'):
        
        print(mode)
        self.cnt+=1
        # img = self._get_image()
        if mode == 'array':
            return self.current_state
        elif mode == 'human':
            
            base_image = self.base_image.squeeze(0)
            process_image = self._batch_face_image_sampler(self.current_state).squeeze(0)

            image_fig = plt.figure()
            image1 = image_fig.add_subplot(1,2, 1)
            image1.title.set_text('real image')
            image1.imshow(np.asarray(base_image))
            
            image2 = image_fig.add_subplot(1,2, 2)
            image2.title.set_text('genereated image')
            image2.imshow(np.asarray(process_image))
            plt.show()
        elif mode == 'save':

            base_image = self.base_image.squeeze(0)
            process_image = self._batch_face_image_sampler(self.current_state).squeeze(0)

            image_fig = plt.figure()
            image1 = image_fig.add_subplot(1,2, 1)
            image1.title.set_text('real image')
            image1.imshow(np.asarray(base_image))
            
            image2 = image_fig.add_subplot(1,2, 2)
            image2.title.set_text('genereated image')
            image2.imshow(np.asarray(process_image))

            plt.savefig(os.path.join(config.save_adds, "img_" + str(self.cnt) + ".jpg")) 
            
            if (self.save_current_state_flag):
                self.step_list.append(self.current_state)
            else:
                pass

            if (config.max_step < self.cnt):
                if (self.save_current_state_flag):
                    np.save(os.path.join(config.save_adds, "trajectory.npy"), {"step_list" : self.step_list, "hyperplane" : self.hyperplane, "base_state" : self.base_states})
                else:
                    pass
                sys.exit()
            
            
            # print ("current state : ", self.current_state)
            # process_image = self._batch_face_image_sampler(self.current_state).squeeze(0)

            # plt.imshow(np.asarray(process_image))
            
            return self.current_state

    
    def _identity_reward_score(self, feature_vec_inital, feature_vec_final):
         ## l2 norm identity loss
        l2_norm = torch.norm(feature_vec_inital - feature_vec_final, 2, dim=1)
        assert(feature_vec_inital.shape[0] == l2_norm.shape[0])
        # return l2_norm.unsqueeze(1)
        return 1e4-torch.sum(l2_norm).detach().cpu().numpy()


    def _hyperplane_reward_score(self, z_feature_gen):
        ## calculating the score with respect to hyperplane
        print ("norm_vec : ", np.linalg.norm(z_feature_gen))
        eps = torch.tensor(self.hyperplane_loss_parameter)
        l1_norm = torch.norm(torch.from_numpy(z_feature_gen).to(self.device) * self.hyperplane - eps, 1, dim=1)
        return l1_norm.unsqueeze(1).detach().cpu().numpy() 


    def _reward_score(self, face_feature, z_feature_gen):
        ## reward score 
        identity_loss = self._identity_reward_score(face_feature, self.base_feature) 
        hyperplane_loss = self._hyperplane_reward_score(z_feature_gen)

        print("identity", identity_loss, "hyp_loss", hyperplane_loss)
        ## total loss function 

        face_and_hinge_feature = (identity_loss + self.relative_reward_hyperparameter * hyperplane_loss)
        print("face and hinge", face_and_hinge_feature)
        # return self.lambda_parameter * (1 - (self.eta_parameter *face_and_hinge_feature  / torch.exp(self.eta_parameter * face_and_hinge_feature)))
        return self.lambda_parameter * (- self.eta_parameter*face_and_hinge_feature ), identity_loss, hyperplane_loss

    def _kl_div_loss(self, mu, log_var):
        ## calculating the Kl divergence term
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
        assert(mu.shape[0] == kl_div.shape[0])
        return kl_div.unsqueeze(1).detach().cpu().numpy()

    def get_action_meanings(self):
        return 


    def clone_state(self):
        return


    def restore_state(self, state):
        return


    def clone_full_state(self):
        return


    def restore_full_state(self, state):
        return


    def _load_model(self):
        # logger = setup_logger(config.log_output_dir, logger_name='load model')
        
        model = None
        if(self.model_name == 'pggan_celebahq'):
            model = PGGANGenerator(self.model_name, logger=None)
        elif(self.model_name == 'stylegan_celebahq'):
            model =  StyleGANGenerator(self.model_name, logger=None)
        print('Loaded face model:', self.model_name)
        return model


    def _batch_generator(self):
        ## sampling a batch
        latent_vector = self.model.easy_sample(self.batch_size)
        assert latent_vector.shape[0] <= 4
        result = self.model.easy_synthesize(latent_vector)
        assert np.array_equal(result['z'], latent_vector)
        return (torch.Tensor(result['z']).to(self.device), result['image'])

    
    def _batch_face_image_sampler(self, z):
        assert z.shape[0] <= 4 # ensure that the batch size is less than or equal to 4
        # z = z.cpu().detach().numpy()
        result = self.model.easy_synthesize(z)
        assert np.array_equal(result['z'], z)
        # return (torch.Tensor(result['z']).to(self.device), result['image'])
        return result['image']


    def  _load_hyperplane(self,hyperplane_path):
        hyperplane = torch.Tensor(np.load(hyperplane_path))
        hyperplane = hyperplane.to(self.device)
        return hyperplane


if __name__ == "__main__":
    env = InterfaceGanEnv() 
    print(env.action_space)
    print(env.observation_space)
    env.reset()
    for i in range(40):
        action = env.action_space.sample()
        print(i)
        env.render()
        env.step(action)
