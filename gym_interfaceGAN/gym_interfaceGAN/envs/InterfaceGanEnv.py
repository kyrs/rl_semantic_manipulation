import math
import os
import sys
from collections import deque
import numpy as np
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding
import torch
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import tensorflow as tf 
from pathlib import Path

sys.path.insert(0, "/home/kumar/RL_env/gym_interfaceGAN/gym_interfaceGAN/envs/")
print(sys.path)
from PIL import Image
from library.InterFaceGAN.models.model_settings import MODEL_POOL
from library.InterFaceGAN.models.pggan_generator import PGGANGenerator
from library.InterFaceGAN.models.stylegan_generator import StyleGANGenerator
from library.InterFaceGAN.utils.logger import setup_logger
from library.InterFaceGAN.utils.manipulator import linear_interpolate
from library.identity_loss import FaceNet, AlexNet

import config
from age_prediction_model import AgePredictionModel
from eyeglass_prediction_model import EyeglassPredictionModel
from gender_prediction_model  import GenderPredictionModel
from smile_prediction_model import  SmilePredictionModel

from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt

from PIL import Image as PImage

class InterfaceGanEnv(gym.Env):
    def __init__(self, gan_model_name = config.gan_model_name, batch_size = config.batch_size, device = config.device, 
                    latent_dim = config.latent_dim,max_identity_limit = config.max_identity_limit, step_size = config.step_size, 
                    typ_set_eps = config.typ_set_eps,
                    typ_set_punishment = config.typ_set_punishment, basis_list = config.basis_list, 
                     weight_clip = config.weight_clip,
                     identity_punishment = config.identity_punishment,
                    polyak_param = config.polyak_param, max_step =config.max_step, 
                    prefered_identity_limit = config.prefered_identity_limit,
                    reset_train_sample = config.reset_train_sample, 
                    reset_test_sample = config.reset_test_sample,
                    isTrain = config.isTrain,
                    not_reach_goal_punishment = config.not_reach_goal_punishment,
                    semanticTesting = config.semanticTesting,
                    save_adds = config.save_adds,
                    run_batch_test = config.run_batch_test,

                    use_age_flag = config.use_age,
                    age_reward = config.age_reward,
                    age_model_path = config.age_model_path,

                    use_eyeglass_flag = config.use_eyeglass,
                    eyeglass_reward = config.eyeglass_reward,
                    eyeglass_model_path = config.eyeglass_model_path,

                    use_gender_flag = config.use_gender,
                    gender_reward = config.gender_reward,
                    gender_model_path = config.gender_model_path,

                    use_smile_flag = config.use_smile,
                    smile_reward = config.smile_reward,
                    smile_model_path = config.smile_model_path,
                    ):
        ## intializing all the model based parameters 
        super(InterfaceGanEnv, self).__init__()

        self.gan_model_name = gan_model_name
        self.batch_size = batch_size
        self.device = device
        self.typ_set_eps = typ_set_eps
        self.typ_set_punishment = typ_set_punishment

        ## loading  the model
        self.model = self._load_model()
        self.latent_dim = latent_dim 

        self.identity_model = AlexNet(device=self.device)
        self.step_size = step_size

        self.act_list = []
        self.step_list = []

        self.max_identity_limit = max_identity_limit
        self.identity_punishment = identity_punishment
        
        self.basis_list = basis_list
        
        ## keeping a track of current state
        self.current_state = None
        self.base_states = None 
        self.base_image = None
        self.cnt = None # used in render for saving images
        self.number_of_steps = -1 # used in the loss function

        self.reached = False
        self.stepFlag = 0
        self.totalResetCnt = 0
        self.semanticTypeBool = False
        
        self.max_step = max_step

        self.polyak_param = polyak_param

        self.use_age_flag = use_age_flag
        self.use_gender_flag = use_gender_flag
        self.use_smile_flag = use_smile_flag
        self.use_eyeglass_flag = use_eyeglass_flag 
        
        self.isTrain = isTrain
        self.not_reach_goal_punishment = not_reach_goal_punishment

        self.weight_clip = weight_clip
        
        self.semanticTesting = semanticTesting


        if self.use_age_flag:
            self.ageModelObj = AgePredictionModel(ageModelPath = age_model_path, ageReward = age_reward )
            
        elif self.use_gender_flag:
            self.genderModel = GenderPredictionModel(genderModelPath = gender_model_path, genderReward = gender_reward)

        elif self.use_eyeglass_flag:
            self.eyeglassModel = EyeglassPredictionModel(eyeglassModelPath = eyeglass_model_path , eyeglassReward = eyeglass_reward )

        elif self.use_smile_flag:
            self.smileModel = SmilePredictionModel(smileModelPath = smile_model_path, smileReward = smile_reward)
        
        else:
            pass

        ############ base image flag ##################

        self.prefered_identity_limit = prefered_identity_limit
        ##############################################
        
        
        self.reset_train_sample_npz = np.load(reset_train_sample)
        self.reset_test_sample_npz = np.load(reset_test_sample)
        self.epsReward = 0

        self.run_batch_test = run_batch_test
        self.save_adds = save_adds
        
        self.action_space = spaces.Box(low=-1, high=1, dtype=np.float32, shape=(len(self.basis_list) + self.latent_dim + 1,)) # weight for each basis, predicted basis, weight for predicted basis
        self.observation_space = spaces.Box(low=-2, high=2, dtype=np.float32, shape=(self.latent_dim*3,))


        self.idx = 0
        idx = 0
        if self.run_batch_test:
            pathAdd = Path(self.save_adds)
            pathAdd = pathAdd.parent
            batchProcessWriter = os.path.join(pathAdd, "BatchProcessStatus")
            if os.path.isfile(batchProcessWriter):
                with open(batchProcessWriter,"r") as fileReader:
                    idxProTillNow = fileReader.read()
                    self.idx  = int(idxProTillNow) + 1

                with open(batchProcessWriter,"w") as fileWriter:
                    fileWriter.write(str(self.idx ))
            else:
                with open(batchProcessWriter,"w") as fileWriter:
                    fileWriter.write(str(0))
            self.idx 


    def linear_approx (self,w1, basis ):
        
        w1 = np.clip(w1, - self.weight_clip, self.weight_clip)
        # basis = np.clip(basis, -1, 1)        
        return w1*basis


    def step(self, a):
        old_state = self.current_state[:self.latent_dim]
        goal_vec  = self.current_state[self.latent_dim:]

        self.number_of_steps += 1
        done = False

        weight_principal_basis = a[:len(self.basis_list)]
        ########## clip principal basis weight#######
        # weight_principal_basis = np.clip (weight_principal_basis,-self.weight_clip, self.weight_clip)

        predicted_basis = a[len(self.basis_list):]

        weight_principal_basis = np.abs(weight_principal_basis)
        direction = -1 *  (not self.semanticTypeBool) + 1 * self.semanticTypeBool
        shift_principal_vec = direction * (weight_principal_basis) @ self.basis_list
        

        shift_predicted_vec = self.linear_approx(w1 = predicted_basis[0], basis = predicted_basis[1:])
        new_state =  self.polyak_param * (old_state) + (1-self.polyak_param)*(old_state + shift_principal_vec + shift_predicted_vec)
        new_state_norm = np.linalg.norm(new_state)

        print ("=====Detail eps and step =====")
        print(f"total reset cnt :{self.totalResetCnt}")
        print(" ep step : ", self.number_of_steps)
        print()        

        print("###### Start step #########")

        print ("========Prediction==============")
        print(f"direction : {direction}")
        print(f'principal vec   (sample) : {shift_principal_vec[:5]}')
        print(f'principal basis (weight): {weight_principal_basis}')

        print(f'shift predict (sample) : {shift_predicted_vec[:5]}')
        print (f"predicted vec weight : {predicted_basis[0]}")
        print(f'polyak: {self.polyak_param}')
        print()

        print ("===========state status=========")
        print(f'norm of new state vector: {new_state_norm}')
        print ("low limit : ",math.sqrt(self.latent_dim) - self.typ_set_eps)
        print("upper limit : ", math.sqrt(self.latent_dim) + self.typ_set_eps)
        

        self.current_state = np.concatenate((new_state, goal_vec), axis=None)
        assert(not np.array_equal(new_state, goal_vec))
        # self.current_state = new_state

        reward_val = 0
       
        if (self.checkInTypSet(new_state_norm) and (self.number_of_steps < self.max_step)):
            
            new_state = np.expand_dims(new_state, axis=0)
            new_state_images = self._batch_face_image_sampler(new_state)
            
            ## extracting facial features 
            new_feature = self.identity_model.get_feature_vector(new_state_images)
            identity_loss= self._reward_score(face_feature = new_feature)


            if self.use_age_flag:
                reward_val += self.ageModelObj.stepReward(imageMat = new_state_images, latentVec = new_state)

                if len(self.ageModelObj.bucketRemaining)==0:
                    if not self.isTrain:
                        self.reached = True
                        done = False
                    else:
                        done = True ## end episode once all the buckets are filled  
                        pass
                

            elif self.use_gender_flag:
                reward_val += self.genderModel.stepReward(imageMat = new_state_images, latentVec = new_state)

                if len(self.genderModel.bucketRemaining)==0:
                    if not self.isTrain:
                        self.reached = True
                        done = False
                    else:
                        done = True ## end episode once all the buckets are filled  
                        pass
                

            elif self.use_eyeglass_flag:

                reward_val += self.eyeglassModel.stepReward(imageMat = new_state_images, latentVec = new_state)
                
                if len(self.eyeglassModel.bucketRemaining)==0:
                
                    if not self.isTrain:
                        self.reached = True
                        done = False
                    else:
                        done = True ## end episode once all the buckets are filled  
                        pass

            elif self.use_smile_flag:
                self.smileModel.stepReward(imageMat = new_state_images, latentVec = new_state)
                
                if len(self.smileModel.bucketRemaining)==0:
                
                    if not self.isTrain:
                        self.reached = True
                        done = False
                    else:
                        done = True ## end episode once all the buckets are filled  
                        pass
            else:
                pass

            if identity_loss > self.max_identity_limit:
                reward_val = self.identity_punishment ## setting the reward  to theh punishment value 

                done = True

                if not self.isTrain:
                    self.reached = True
                    done = False
                else:
                    pass
            else:
                if identity_loss <= self.prefered_identity_limit:
                    reward_val = reward_val*2 ## more preference if model find age withing prefered identity bound


                        
                
                ## giving the agent negetive reward if it does not figure out required semantic variation
                if reward_val == 0:
                    reward_val = self.not_reach_goal_punishment  
                else:
                    pass       
        else:
            print("condition false")
            done = True 

            if not self.isTrain:
                self.reached = True
                done = False
            else:
                pass
            
            if not self.checkInTypSet(new_state_norm):
                reward_val = self.typ_set_punishment ## typical set based punishment
            else:
                reward_val = self.not_reach_goal_punishment 

        print('reward_val', reward_val)
        self.epsReward += reward_val
        epsReturnLogger = {}

        if not done:
            epsReturnLogger["episode"] = None
        else:
            epsReturnLogger["episode"] = {"r": self.epsReward, 'l': self.number_of_steps}

        print ("######End step###########")
        print()
        return self.current_state, reward_val, done, epsReturnLogger


    @property
    def _n_actions(self):
        return self.latent_dim

    def checkInTypSet(self, new_state_norm):
       return  (math.sqrt(self.latent_dim) - self.typ_set_eps < new_state_norm) and (new_state_norm < math.sqrt(self.latent_dim) + self.typ_set_eps)

    def reset(self):
        self.cnt = 0 # used in render for saving images
        self.number_of_steps = 0 # used in the loss function
        self.epsReward = 0
        semanticBinIndx = None
        
        
        if self.isTrain:
            randInt = np.random.random_integers(0, len(self.reset_train_sample_npz)-1)    
            if (self.totalResetCnt+1)%100 == 0:
                self.semanticTypeBool = not self.semanticTypeBool
            else:
                pass
            
            oneHotEncode = int(self.semanticTypeBool) * np.array([1 for _ in range(self.latent_dim)])
            baseState = self.reset_train_sample_npz[randInt]
            goal = np.concatenate( (baseState, oneHotEncode))
            print()
            print ("*********Start Episode**************")
            print ("one hot vec :", oneHotEncode)
            print()
            print()
            semanticBinIndx = int(self.semanticTypeBool)


        else:
            ### code for testing the model
            if self.semanticTesting is not None:
                semanticBinIndx = self.semanticTesting 
            else:
                raise Exception("ConfigError : issue wuth the semantic testing flag ")
            self.semanticTypeBool = semanticBinIndx

            oneHotEncode = int(semanticBinIndx) * np.array([1 for _ in range(self.latent_dim)])
            idx = 0
            if self.run_batch_test:
               idx = self.idx
            else:
                randInt = np.random.random_integers(0, len(self.reset_test_sample_npz)-1)
                print(randInt)
                print(oneHotEncode[:5])
                print ("random Int in test data: ", randInt)
                print("age bin Idx : ", semanticBinIndx )
                idx = randInt

            print("index : ", idx)
            # input()
            baseState = self.reset_test_sample_npz[idx]
            goal = np.concatenate( (baseState, oneHotEncode))

        if self.use_age_flag:
            self.ageModelObj.reset(binOrder = semanticBinIndx, oneHotEnc = oneHotEncode)

        elif self.use_gender_flag:
          self.genderModel.reset(binOrder = semanticBinIndx, oneHotEnc = oneHotEncode)  
                       

        elif self.use_eyeglass_flag:
            self.eyeglassModel.reset(binOrder = semanticBinIndx, oneHotEnc = oneHotEncode)
            
           
        elif self.use_smile_flag:
            self.smileModel.reset(binOrder = semanticBinIndx, oneHotEnc = oneHotEncode)
        else : 
            pass

        self.current_state = np.concatenate((baseState, goal), axis = None)
        zVecGoal = np.expand_dims(baseState, axis=0)
        start_image = self._batch_face_image_sampler(zVecGoal)
        self.start_image = start_image
        self.start_feature = self.identity_model.get_feature_vector(start_image)
        self.base_image = start_image
        self.step_list = []    
        self.totalResetCnt+=1            
        return self.current_state


    def close(self):
        self.current_state = None
        self.base_states = None 
        self.base_image = None
        return 
        

    def render(self, mode='save'):
        print(mode)
        print("count",self.cnt)
        # img = self._get_image()
        if mode == 'array':
            return self.current_state
        
        elif mode == 'human':    
            base_image = self.base_image.squeeze(0)
            z_vector = self.current_state[:self.latent_dim]
            z_vector = np.expand_dims(z_vector, axis=0)
            print(z_vector.shape)
            process_image = self._batch_face_image_sampler(z_vector).squeeze(0)
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
            z_vector = self.current_state[:self.latent_dim]
            z_vector = np.expand_dims(z_vector, axis=0)
            print(z_vector.shape)
            process_image = self._batch_face_image_sampler(z_vector).squeeze(0)

            
            self.step_list.append(self.current_state)
            
            imgOfInt = Image.fromarray(np.uint8(process_image)).convert('RGB')

            if self.use_age_flag:
                imgOfInt.save(os.path.join(self.save_adds, "img_" + str(self.cnt) +"_age:" +str("{:.2f}".format((self.ageModelObj.currentAge)))  + ".jpg"))

            elif self.use_gender_flag:
                imgOfInt.save(os.path.join(self.save_adds, "img_" + str(self.cnt) +"_gender_prob:" +str("{:.2f}".format((self.genderModel.currentGender)))  + ".jpg"))                

            elif self.use_eyeglass_flag:
                imgOfInt.save(os.path.join(self.save_adds, "img_" + str(self.cnt) +"_eyeglass_prob:" +str("{:.2f}".format((self.eyeglassModel.currentEyeglass)))  + ".jpg"))
            
           
            elif self.use_smile_flag:
                imgOfInt.save(os.path.join(self.save_adds, "img_" + str(self.cnt) +"_smile_prob:" +str("{:.2f}".format((self.smileModel.currentSmile)))  + ".jpg"))
            else : 
                pass
                

            if self.cnt ==0:
                baseImage = Image.fromarray(np.uint8(base_image)).convert('RGB')
                baseImage.save(os.path.join(self.save_adds, "base_img.jpg"))
            else:
                pass

            if (self.max_step < self.cnt) or (self.reached):
                np.save(os.path.join(self.save_adds, "trajectory.npy"), {"step_list": self.step_list, "base_state": self.base_states})
                sys.exit()
        self.cnt += 1
        return self.current_state

    
    def _identity_reward_score(self, feature_vec_inital, feature_vec_final):
         ## l2 norm identity loss
        l2_norm = torch.norm(feature_vec_inital - feature_vec_final, 2, dim=1)
        assert(feature_vec_inital.shape[0] == l2_norm.shape[0])
        return torch.sum(l2_norm).detach().cpu().numpy()


    def _reward_score(self, face_feature):
        ## reward score 
        identity_loss = self._identity_reward_score(face_feature, self.start_feature)
        print ("ide loss", identity_loss)
        return identity_loss


    def _load_model(self):
        model = None
        if(self.gan_model_name == 'pggan_celebahq'):
            model = PGGANGenerator(self.gan_model_name, logger=None)
        elif(self.gan_model_name == 'stylegan_celebahq'):
            model =  StyleGANGenerator(self.model_name, logger=None)
        print('Loaded face model:', self.gan_model_name)
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
        result = self.model.easy_synthesize(z)
        assert np.array_equal(result['z'], z)
        return result['image']


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
