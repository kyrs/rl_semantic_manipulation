"""
__name__ : Kumar shubham
__desc__ : gender prediction model
__Date__ : 14 Nov 2020
"""
import warnings
warnings.filterwarnings('ignore')

from PIL import Image as PImage
from fastai import *
from fastai.vision import *
from scipy.spatial import distance

class GenderPredictionModel(object):
    def __init__(self, genderModelPath = "/mnt/hdd1/shubham/trainClassifier/ckpt/gender/gender_model-1",  genderReward = 100, maskSimilarFlag = False, limitPerGenderDv = 1, maskLimit = 0.5):

        """
        args : 
        genderModelPath    : path where model has been saved
        genderReward    : reward for sampling from a given gender bucket
        maskSimilarFlag : flag to mask sampling latent vector which are too close
        limitPerGenderDv   : no of images allowed to be sampled from given gender bucket
        maskLimit       : distance limit to consider a new latent vector for reward (part of maskSimilar Flag)
        """
        #0: Female 1 : male

        self.genderModelPath = genderModelPath
        self.genderReward = genderReward
       

        path = ""
        
        dataBunch = (ImageList.from_folder("")
        .random_split_by_pct()
        .label_from_folder(classes = ["Female", "Male"])
        # .label_from_folder()
        .transform(get_transforms(), size=224)
        .databunch()).normalize(imagenet_stats);

        # dataBunch.c2i = {'negative': 0, 'positive': 1}

        self.genderModel = create_cnn(dataBunch, models.resnet50, pretrained=False)
        self.genderModel.load(self.genderModelPath)

        print( Path(self.genderModelPath).parents)
        assert ("gender" in self.genderModelPath.split("/"))

        
        print ("Model Loaded .....")
       
        self.currentGender = -1
        self.maskSimilarFlag = maskSimilarFlag
        self.maskLimit = maskLimit

        self.limitPerGenderDv = limitPerGenderDv
        self.baseImageGender = None
        self.predictedGenderList = []
        # self.inpDictGender = {(0, 0.16) : 0, (0.16, 0.32) : 0, (0.32, 0.48) : 0, (0.48, 0.64 ) : 0, (0.64, 0.80) : 0, (0.80, 1) : 0  }
    def reset(self, binOrder = 1, oneHotEnc = None):
        ## code to reset the model
        self.baseImageGender = None
        self.rewardVecList = []
        self.predictedGenderList = []
        self.genderModel.currentGender = -1
        self.inpDictGender = {(0, 0.20) : 0,  (0.80, 1) : 0  }
        
        print ("RESETING Gender LIMIT ..")
        self.order = binOrder
        self.oneHotEnc = oneHotEnc
        self.bucketRemaining = []

    def buktRemainEps(self):
        ######### functuion with detail about which buckets are empty in given order ############
        self.bucketRemaining = []
        if self.baseImageGender is not None:
            for i,j in self.inpDictGender:  
                if self.order == 0:
                    if (j <= self.baseImageGender) and (self.inpDictGender[(i,j)] == 0) :
                        print(i,j,self.inpDictGender[(i,j)])
                        self.bucketRemaining.append((i,j))
                    else:
                        pass
                else:
                    if (i >= self.baseImageGender) and (self.inpDictGender[(i,j)] == 0) :
                        print(i,j,self.inpDictGender[(i,j)])
                        self.bucketRemaining.append((i,j))
                    else:
                        pass



    def stepReward(self,imageMat="", latentVec = ""):
       
        
        imageMat = np.squeeze(imageMat, axis = 0)
        imageMat = np.transpose(imageMat,(2,0,1))
        c,w,h = imageMat.shape
        assert (c==3) 
        img = Image(torch.from_numpy(imageMat).float()/255.0)
        gender = self.genderModel.predict(img)[2][1]
        
        ## WTF : you need to convert FloatItem to float ###
        gender = gender.data
        print (gender)
        self.currentGender = gender
        pushToLtFlag = False

        foundGenderGrp = ()      
        for genderGrp in self.inpDictGender:
            low, high = genderGrp
            if (gender>=low) and (gender< high):
                countGenderGrp = self.inpDictGender[genderGrp]
                foundGenderGrp = genderGrp
                if countGenderGrp >= self.limitPerGenderDv:
                    pushToLtFlag = False
                    break
                else:
                    pass

                    if len(self.predictedGenderList)== 0:
                        pass
                    else:
                        
                        if self.order == 0:
                            pushToLtFlag = (gender <= self.baseImageGender)
                        else:
                            pushToLtFlag = (gender > self.baseImageGender)      

                    break
            else:
                continue

        #### adding first gender group in the list
        if self.baseImageGender is None and foundGenderGrp!=():
            pushToLtFlag = True
        else:
            pass

        if pushToLtFlag:
            giveRewardFlag = False
            print("Pushing To gender List ..")
            if len(self.rewardVecList) == 0:
                self.inpDictGender[foundGenderGrp] += 1
                self.baseImageGender = gender 
                ### return reward for first latent vec ###    
                self.rewardVecList.append(latentVec)
                self.predictedGenderList.append(gender)
                giveRewardFlag = True
                # return self.reward

            else:
                if self.maskSimilarFlag:
                    eucdDistance = [distance.euclidean(latentVec, genderVec) for genderVec in self.rewardVecList ]
                    minDistance = np.min(eucdDistance) 
                    
                    print("distance min  : ", minDistance)
                    if minDistance  >  self.maskLimit:
                        self.inpDictGender[foundGenderGrp] += 1    
                        self.rewardVecList.append(latentVec)
                        self.predictedGenderList.append(gender)
                        giveRewardFlag = True
                        # return self.genderReward
                    else:
                        return 0.0
                else:
                    self.inpDictGender[foundGenderGrp] += 1
                    self.rewardVecList.append(latentVec)
                    self.predictedGenderList.append(gender)
                    giveRewardFlag = True
                    # return self.genderReward    
        else:
            giveRewardFlag = False
            # return 0.0


        ############## Printing status ###############
        print("########### gender details ################")
        print(f"predicted gender : {gender}")
        print(f"base image gender : {self.baseImageGender}")
        print(f"gender prob found till now : {self.predictedGenderList}")
        print(f"Mask similar Flag : {self.maskSimilarFlag}")
        print(f"self order : {self.order}")
        print(f"remaining gender prob group : ")
        self.buktRemainEps()
        print(f"bucket remaining :{len(self.bucketRemaining)}")
        

        
        

        if giveRewardFlag:
             return self.genderReward
        else:
            return 0.0


    def predictImage(self,imagePth):
        ## code for predicting the gender for the enlisted image
        imgObj = open_image(imagePth)
        print(self.genderModel.predict(imgObj))



if __name__ =="__main__":
    obj = GenderPredictionModel()
    obj.reset()
    imagePth = "/mnt/hdd1/shubham/trainClassifier/ckpt/gender/female.jpg"
    imageMat = PImage.open(imagePth)
    # obj.predictImage(imagePth = "/NewVolume/shubham/attribute_model/img/image_base.jpg")
    data = np.asarray(imageMat)
    # print(data)
    # print(data.shape)
    print(obj.stepReward(imageMat=data))