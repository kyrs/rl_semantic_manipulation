"""
__name__ : Kumar shubham
__desc__ : eyeglass prediction model
__Date__ : 14 Nov 2020
"""
import warnings
warnings.filterwarnings('ignore')

from PIL import Image as PImage
from fastai import *
from fastai.vision import *
from scipy.spatial import distance

class EyeglassPredictionModel(object):
    def __init__(self, eyeglassModelPath = "/mnt/hdd1/shubham/trainClassifier/ckpt/eyeglasses/eyeglasses_model-1",  eyeglassReward = 100, maskSimilarFlag = False, limitPerEyeglassDv = 1, maskLimit = 0.5):

        """
        args : 
        eyeglassModelPath    : path where model has been saved
        eyeglassReward    : reward for sampling from a given eyeglass bucket
        maskSimilarFlag : flag to mask sampling latent vector which are too close
        limitPerEyeglassDv   : no of images allowed to be sampled from given eyeglass bucket
        maskLimit       : distance limit to consider a new latent vector for reward (part of maskSimilar Flag)
        """
        #0: no_eyeglass 1 : eyeglass

        self.eyeglassModelPath = eyeglassModelPath
        self.eyeglassReward = eyeglassReward
       

        path = ""
        
        dataBunch = (ImageList.from_folder("")
        .random_split_by_pct()
        .label_from_folder(classes = ["no_eyeglass", "yes_eyeglass"])
        # .label_from_folder()
        .transform(get_transforms(), size=224)
        .databunch()).normalize(imagenet_stats);

        # dataBunch.c2i = {'negative': 0, 'positive': 1}

        self.eyeglassModel = create_cnn(dataBunch, models.resnet50, pretrained=False)
        self.eyeglassModel.load(self.eyeglassModelPath)

        print( Path(self.eyeglassModelPath).parents)
        assert ("eyeglasses" in self.eyeglassModelPath.split("/"))

        
        print ("Model Loaded .....")
       
        self.currentEyeglass = -1
        self.maskSimilarFlag = maskSimilarFlag
        self.maskLimit = maskLimit

        self.limitPerEyeglassDv = limitPerEyeglassDv
        self.baseImageEyeglass = None
        self.predictedEyeglassList = []
        # self.inpDictEyeglass = {(0, 0.16) : 0, (0.16, 0.32) : 0, (0.32, 0.48) : 0, (0.48, 0.64 ) : 0, (0.64, 0.80) : 0, (0.80, 1) : 0  }
    def reset(self, binOrder = 1, oneHotEnc = None):
        ## code to reset the model
        self.baseImageEyeglass = None
        self.rewardVecList = []
        self.predictedEyeglassList = []
        self.eyeglassModel.currentEyeglass = -1
        self.inpDictEyeglass = {(0, 0.20) : 0, (0.80, 1) : 0  }
        
        print ("RESETING Eyeglass LIMIT ..")
        self.order = binOrder
        self.oneHotEnc = oneHotEnc
        self.bucketRemaining = []

    def buktRemainEps(self):
        ######### functuion with detail about which buckets are empty in given order ############
        self.bucketRemaining = []
        if self.baseImageEyeglass is not None:
            for i,j in self.inpDictEyeglass:  
                if self.order == 0:
                    if (j <= self.baseImageEyeglass) and (self.inpDictEyeglass[(i,j)] == 0) :
                        print(i,j,self.inpDictEyeglass[(i,j)])
                        self.bucketRemaining.append((i,j))
                    else:
                        pass
                else:
                    if (i >= self.baseImageEyeglass) and (self.inpDictEyeglass[(i,j)] == 0) :
                        print(i,j,self.inpDictEyeglass[(i,j)])
                        self.bucketRemaining.append((i,j))
                    else:
                        pass



    def stepReward(self,imageMat="", latentVec = ""):
       
        
        imageMat = np.squeeze(imageMat, axis = 0)
        imageMat = np.transpose(imageMat,(2,0,1))
        c,w,h = imageMat.shape
        assert (c==3) 
        img = Image(torch.from_numpy(imageMat).float()/255.0)
        eyeglass = self.eyeglassModel.predict(img)[2][1]
        
        ## WTF : you need to convert FloatItem to float ###
        eyeglass = eyeglass.data
        print (eyeglass)
        self.currentEyeglass = eyeglass
        pushToLtFlag = False

        foundEyeglassGrp = ()      
        for eyeglassGrp in self.inpDictEyeglass:
            low, high = eyeglassGrp
            if (eyeglass>=low) and (eyeglass< high):
                countEyeglassGrp = self.inpDictEyeglass[eyeglassGrp]
                foundEyeglassGrp = eyeglassGrp
                if countEyeglassGrp >= self.limitPerEyeglassDv:
                    pushToLtFlag = False
                    break
                else:
                    pass

                    if len(self.predictedEyeglassList)== 0:
                        pass
                    else:
                        
                        if self.order == 0:
                            pushToLtFlag = (eyeglass <= self.baseImageEyeglass)
                        else:
                            pushToLtFlag = (eyeglass > self.baseImageEyeglass)      

                    break
            else:
                continue

        #### adding first eyeglass group in the list
        if self.baseImageEyeglass is None and foundEyeglassGrp!=():
            pushToLtFlag = True
        else:
            pass

        if pushToLtFlag:
            giveRewardFlag = False
            print("Pushing To eyeglass List ..")
            if len(self.rewardVecList) == 0:
                self.inpDictEyeglass[foundEyeglassGrp] += 1
                self.baseImageEyeglass = eyeglass 
                ### return reward for first latent vec ###    
                self.rewardVecList.append(latentVec)
                self.predictedEyeglassList.append(eyeglass)
                giveRewardFlag = True
                # return self.reward

            else:
                if self.maskSimilarFlag:
                    eucdDistance = [distance.euclidean(latentVec, eyeglassVec) for eyeglassVec in self.rewardVecList ]
                    minDistance = np.min(eucdDistance) 
                    
                    print("distance min  : ", minDistance)
                    if minDistance  >  self.maskLimit:
                        self.inpDictEyeglass[foundEyeglassGrp] += 1    
                        self.rewardVecList.append(latentVec)
                        self.predictedEyeglassList.append(eyeglass)
                        giveRewardFlag = True
                        # return self.eyeglassReward
                    else:
                        return 0.0
                else:
                    self.inpDictEyeglass[foundEyeglassGrp] += 1
                    self.rewardVecList.append(latentVec)
                    self.predictedEyeglassList.append(eyeglass)
                    giveRewardFlag = True
                    # return self.eyeglassReward    
        else:
            giveRewardFlag = False
            # return 0.0


        ############## Printing status ###############
        print("########### eyeglass details ################")
        print(f"predicted eyeglass : {eyeglass}")
        print(f"base image eyeglass : {self.baseImageEyeglass}")
        print(f"eyeglass prob found till now : {self.predictedEyeglassList}")
        print(f"Mask similar Flag : {self.maskSimilarFlag}")
        print(f"self order : {self.order}")
        print(f"remaining eyeglass prob group : ")
        self.buktRemainEps()
        print(f"bucket remaining :{len(self.bucketRemaining)}")
        

        
        

        if giveRewardFlag:
             return self.eyeglassReward
        else:
            return 0.0


    def predictImage(self,imagePth):
        ## code for predicting the eyeglass for the enlisted image
        imgObj = open_image(imagePth)
        print(self.eyeglassModel.predict(imgObj))



if __name__ =="__main__":
    obj = EyeglassPredictionModel()
    obj.reset()
    imagePth = "/mnt/hdd1/shubham/trainClassifier/ckpt/eyeglasses/female.jpg"
    imageMat = PImage.open(imagePth)
    # obj.predictImage(imagePth = "/NewVolume/shubham/attribute_model/img/image_base.jpg")
    data = np.asarray(imageMat)
    # print(data)
    # print(data.shape)
    print(obj.stepReward(imageMat=data))