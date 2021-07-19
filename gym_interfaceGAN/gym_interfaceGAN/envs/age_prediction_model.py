"""
__name__ : Kumar shubham
__desc__ : age prediction model
__Date__ : 23 July 2020
"""
import warnings
warnings.filterwarnings('ignore')

from PIL import Image as PImage
from fastai import *
from fastai.vision import *
from scipy.spatial import distance

from eyeglass_prediction_model import EyeglassPredictionModel
from gender_prediction_model  import GenderPredictionModel
from smile_prediction_model import  SmilePredictionModel


class AgePredictionModel(object):
    def __init__(self, ageModelPath = "/home/kumar/RL_env/attribute_model/age/age-stage-128-50",  ageReward = 100, maskSimilarFlag = False, limitPerAgeDv = 1, maskLimit = 0.5):

        """
        args : 
        ageModelPath    : path where model has been saved
        ageReward       : reward for sampling from a given age bucket
        maskSimilarFlag : flag to mask sampling latent vector which are too close
        limitPerAgeDv   : no of images allowed to be sampled from given age bucket
        maskLimit       : distance limit to consider a new latent vector for reward (part of maskSimilar Flag)
        """

        self.ageModelPath = ageModelPath
        self.ageReward = ageReward
       

        path = ""
        data_bunch = (ImageList.from_folder(path)
        .random_split_by_pct()
        .label_const(0, label_cls=FloatList)
        .transform(get_transforms(), size=224)
        .databunch()).normalize(imagenet_stats)

        _ = create_cnn(data_bunch, models.resnet50, pretrained=False) ############## BUG #########################3
        self.ageModel = create_cnn(data_bunch, models.resnet50, pretrained=False)

        self.ageModel.load(self.ageModelPath)

        print ("Model Loaded .....")
        

        self.currentAge = -1
        self.maskSimilarFlag = maskSimilarFlag
        self.maskLimit = maskLimit

        self.limitPerAgeDv = limitPerAgeDv
        self.baseImageAge = None
        self.predictedAgeList = []

    def reset(self, binOrder = None, oneHotEnc = None):
        ## code to reset the model
        self.baseImageAge = None
        self.rewardVecList = []
        self.predictedAgeList = []
        self.ageModel.currentAge = -1
        self.inpDictAge = {(20, 25) : 0, (25, 30) : 0, (30, 35) : 0, (35, 40) : 0, (40, 45) : 0, (45, 50) : 0, (50, 55) : 0, (55, 60) : 0 }
        
        print ("RESETING AGE LIMIT ..")
        self.order = binOrder
        self.oneHotEnc = oneHotEnc
        self.bucketRemaining = []

    def buktRemainEps(self):
        ######### functuion with detail about which buckets are empty in given order ############
        self.bucketRemaining = []
        if self.baseImageAge is not None:
            for i,j in self.inpDictAge:  
                if self.order == 0:
                    if (j <= self.baseImageAge) and (self.inpDictAge[(i,j)] == 0) :
                        print(i,j,self.inpDictAge[(i,j)])
                        self.bucketRemaining.append((i,j))
                    else:
                        pass
                else:
                    if (i >= self.baseImageAge) and (self.inpDictAge[(i,j)] == 0) :
                        print(i,j,self.inpDictAge[(i,j)])
                        self.bucketRemaining.append((i,j))
                    else:
                        pass



    def stepReward(self,imageMat="", latentVec = ""):
       
        # print(imageMat)
        imageMat = np.squeeze(imageMat, axis = 0)
        imageMat = np.transpose(imageMat,(2,0,1))
        c,w,h = imageMat.shape
        assert (c==3) 
        img = Image(torch.from_numpy(imageMat).float()/255.0)
        # print(img.shape)
        # input()
        age = self.ageModel.predict(img)[0]
        
        ### WTF : you need to convert FloatItem to float ###
        age = age.data
        self.currentAge = age[0]
        pushToLtFlag = False

        foundAgeGrp = ()      
        for ageGrp in self.inpDictAge:
            low, high = ageGrp
            if (age>=low) and (age< high):
                countAgeGrp = self.inpDictAge[ageGrp]
                foundAgeGrp = ageGrp
                if countAgeGrp >= self.limitPerAgeDv:
                    pushToLtFlag = False
                    break
                else:
                    pass

                    if len(self.predictedAgeList)== 0:
                        pass
                    else:
                        
                        if self.order == 0:
                            pushToLtFlag = (age <= self.baseImageAge)
                        else:
                            pushToLtFlag = (age > self.baseImageAge)      

                    break
            else:
                continue

        #### adding first age group in the list
        if self.baseImageAge is None and foundAgeGrp!=():
            pushToLtFlag = True
        else:
            pass

        if pushToLtFlag:
            giveRewardFlag = False
            print("Pushing To age List ..")
            if len(self.rewardVecList) == 0:
                self.inpDictAge[foundAgeGrp] += 1
                self.baseImageAge = age 
                ### return reward for first latent vec ###    
                self.rewardVecList.append(latentVec)
                self.predictedAgeList.append(age)
                giveRewardFlag = True
                # return self.reward

            else:
                if self.maskSimilarFlag:
                    eucdDistance = [distance.euclidean(latentVec, ageVec) for ageVec in self.rewardVecList ]
                    minDistance = np.min(eucdDistance) 
                    
                    print("distance min  : ", minDistance)
                    if minDistance  >  self.maskLimit:
                        self.inpDictAge[foundAgeGrp] += 1    
                        self.rewardVecList.append(latentVec)
                        self.predictedAgeList.append(age)
                        giveRewardFlag = True
                        # return self.ageReward
                    else:
                        return 0.0
                else:
                    self.inpDictAge[foundAgeGrp] += 1
                    self.rewardVecList.append(latentVec)
                    self.predictedAgeList.append(age)
                    giveRewardFlag = True
                    # return self.ageReward    
        else:
            giveRewardFlag = False
            # return 0.0


        ############## Printing status ###############
        print("########### age details ################")
        print(f"predicted age : {age}")
        print(f"base image age : {self.baseImageAge}")
        print(f"age found till now : {self.predictedAgeList}")
        print(f"Mask similar Flag : {self.maskSimilarFlag}")
        print(f"self order : {self.order}")
        print(f"remaining age group : ")
        self.buktRemainEps()
        print(f"bucket remaining :{len(self.bucketRemaining)}")
        

        
        

        if giveRewardFlag:
             return self.ageReward
        else:
            return 0.0


    def predictImage(self,imagePth):
        ## code for predicting the age for the enlisted image
        imgObj = open_image(imagePth)
        print(self.ageModel.predict(imgObj))



if __name__ =="__main__":
    obj = AgePredictionModel()
    obj.reset()
    imagePth = "/home/kumar/RL_env/IJCAI/imgs/ijcal_exp_1_eyeglass_reward_neg_50/set0/20201122-014003/base_img.jpg"
    imageMat = PImage.open(imagePth)
    # obj.predictImage(imagePth = "/NewVolume/shubham/attribute_model/img/image_base.jpg")
    data = np.asarray(imageMat)
    print(data)
    print(data.shape)
    print(obj.stepReward(imageMat=data))
