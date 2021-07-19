"""
__name__ : Kumar shubham
__desc__ : smile prediction model
__Date__ : 14 Nov 2020
"""
import warnings
warnings.filterwarnings('ignore')

from PIL import Image as PImage
from fastai import *
from fastai.vision import *
from scipy.spatial import distance

class SmilePredictionModel(object):
    def __init__(self, smileModelPath = "/mnt/hdd1/shubham/trainClassifier/ckpt/Smiling/smiling_model-1",  smileReward = 100, maskSimilarFlag = False, limitPerSmileDv = 1, maskLimit = 0.5):

        """
        args : 
        smileModelPath    : path where model has been saved
        smileReward    : reward for sampling from a given smile bucket
        maskSimilarFlag : flag to mask sampling latent vector which are too close
        limitPerSmileDv   : no of images allowed to be sampled from given smile bucket
        maskLimit       : distance limit to consider a new latent vector for reward (part of maskSimilar Flag)
        """
        #0: no_smile 1 : smile

        self.smileModelPath = smileModelPath
        self.smileReward = smileReward
       

        path = ""
        
        dataBunch = (ImageList.from_folder("")
        .random_split_by_pct()
        .label_from_folder(classes = ["no_smile", "yes_smile"])
        # .label_from_folder()
        .transform(get_transforms(), size=224)
        .databunch()).normalize(imagenet_stats);

        # dataBunch.c2i = {'negative': 0, 'positive': 1}

        self.smileModel = create_cnn(dataBunch, models.resnet50, pretrained=False)
        self.smileModel.load(self.smileModelPath)

        print( Path(self.smileModelPath).parents)
        assert ("Smiling" in self.smileModelPath.split("/"))

        
        print ("Model Loaded .....")
       
        self.currentSmile = -1
        self.maskSimilarFlag = maskSimilarFlag
        self.maskLimit = maskLimit

        self.limitPerSmileDv = limitPerSmileDv
        self.baseImageSmile = None
        self.predictedSmileList = []
        # self.inpDictSmile = {(0, 0.16) : 0, (0.16, 0.32) : 0, (0.32, 0.48) : 0, (0.48, 0.64 ) : 0, (0.64, 0.80) : 0, (0.80, 1) : 0  }
    def reset(self, binOrder = 1, oneHotEnc = None):
        ## code to reset the model
        self.baseImageSmile = None
        self.rewardVecList = []
        self.predictedSmileList = []
        self.smileModel.currentSmile = -1
        self.inpDictSmile = {(0, 0.20) : 0, (0.80, 1) : 0  }
        
        print ("RESETING Smile LIMIT ..")
        self.order = binOrder
        self.oneHotEnc = oneHotEnc
        self.bucketRemaining = []

    def buktRemainEps(self):
        ######### functuion with detail about which buckets are empty in given order ############
        self.bucketRemaining = []
        if self.baseImageSmile is not None:
            for i,j in self.inpDictSmile:  
                if self.order == 0:
                    if (j <= self.baseImageSmile) and (self.inpDictSmile[(i,j)] == 0) :
                        print(i,j,self.inpDictSmile[(i,j)])
                        self.bucketRemaining.append((i,j))
                    else:
                        pass
                else:
                    if (i >= self.baseImageSmile) and (self.inpDictSmile[(i,j)] == 0) :
                        print(i,j,self.inpDictSmile[(i,j)])
                        self.bucketRemaining.append((i,j))
                    else:
                        pass



    def stepReward(self,imageMat="", latentVec = ""):
       
        
        imageMat = np.squeeze(imageMat, axis = 0)
        imageMat = np.transpose(imageMat,(2,0,1))
        c,w,h = imageMat.shape
        assert (c==3) 
        img = Image(torch.from_numpy(imageMat).float()/255.0)
        smile = self.smileModel.predict(img)[2][1]
        
        ## WTF : you need to convert FloatItem to float ###
        smile = smile.data
        print (smile)
        self.currentSmile = smile
        pushToLtFlag = False

        foundSmileGrp = ()      
        for smileGrp in self.inpDictSmile:
            low, high = smileGrp
            if (smile>=low) and (smile< high):
                countSmileGrp = self.inpDictSmile[smileGrp]
                foundSmileGrp = smileGrp
                if countSmileGrp >= self.limitPerSmileDv:
                    pushToLtFlag = False
                    break
                else:
                    pass

                    if len(self.predictedSmileList)== 0:
                        pass
                    else:
                        
                        if self.order == 0:
                            pushToLtFlag = (smile <= self.baseImageSmile)
                        else:
                            pushToLtFlag = (smile > self.baseImageSmile)      

                    break
            else:
                continue

        #### adding first smile group in the list
        if self.baseImageSmile is None and foundSmileGrp!=():
            pushToLtFlag = True
        else:
            pass

        if pushToLtFlag:
            giveRewardFlag = False
            print("Pushing To smile List ..")
            if len(self.rewardVecList) == 0:
                self.inpDictSmile[foundSmileGrp] += 1
                self.baseImageSmile = smile 
                ### return reward for first latent vec ###    
                self.rewardVecList.append(latentVec)
                self.predictedSmileList.append(smile)
                giveRewardFlag = True
                # return self.reward

            else:
                if self.maskSimilarFlag:
                    eucdDistance = [distance.euclidean(latentVec, smileVec) for smileVec in self.rewardVecList ]
                    minDistance = np.min(eucdDistance) 
                    
                    print("distance min  : ", minDistance)
                    if minDistance  >  self.maskLimit:
                        self.inpDictSmile[foundSmileGrp] += 1    
                        self.rewardVecList.append(latentVec)
                        self.predictedSmileList.append(smile)
                        giveRewardFlag = True
                        # return self.smileReward
                    else:
                        return 0.0
                else:
                    self.inpDictSmile[foundSmileGrp] += 1
                    self.rewardVecList.append(latentVec)
                    self.predictedSmileList.append(smile)
                    giveRewardFlag = True
                    # return self.smileReward    
        else:
            giveRewardFlag = False
            # return 0.0


        ############## Printing status ###############
        print("########### smile details ################")
        print(f"predicted smile : {smile}")
        print(f"base image smile : {self.baseImageSmile}")
        print(f"smile prob found till now : {self.predictedSmileList}")
        print(f"Mask similar Flag : {self.maskSimilarFlag}")
        print(f"self order : {self.order}")
        print(f"remaining smile prob group : ")
        self.buktRemainEps()
        print(f"bucket remaining :{len(self.bucketRemaining)}")
        

        
        

        if giveRewardFlag:
             return self.smileReward
        else:
            return 0.0


    def predictImage(self,imagePth):
        ## code for predicting the smile for the enlisted image
        imgObj = open_image(imagePth)
        print(self.smileModel.predict(imgObj))



if __name__ =="__main__":
    obj = SmilePredictionModel()
    obj.reset()
    imagePth = "/mnt/hdd1/shubham/trainClassifier/ckpt/Smiling/male_not_smile.png"
    imageMat = PImage.open(imagePth)
    # obj.predictImage(imagePth = "/NewVolume/shubham/attribute_model/img/image_base.jpg")
    data = np.asarray(imageMat)
    # print(data)
    # print(data.shape)
    print(obj.stepReward(imageMat=data))