import argparse
import torch
import torch.nn as nn
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
from PIL import Image, ImageDraw
from torchvision import transforms
from matplotlib import cm
import torchvision.models as models
import config

class FaceModel:
    def __init__(self, image_size, device):
        self.image_size = image_size
        self.device = device
    
    def get_feature_vector(self, image_list):
        raise NotImplementedError

class FaceNet(FaceModel):
    def __init__(self, image_size=256, device='cpu'):
        super().__init__(image_size, device)
        self.mtcnn = MTCNN(image_size)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval()
    
    def get_feature_vector(self, image_list):
        list_feature_vector = []
        for i, image in enumerate(image_list):
            image_pil_obj = Image.fromarray(image)
            cropped_image = transforms.ToTensor()(image_pil_obj)
            # cropped_image = self.mtcnn(image_pil_obj, save_path=str(i+1)+'.jpg')
            feature_vector_image = self.resnet(cropped_image.unsqueeze(0))
            list_feature_vector.append(feature_vector_image)
        final_feature_vector = torch.cat(list_feature_vector, dim=0).to(self.device)
        return final_feature_vector

class AlexNet(FaceModel):
    def __init__(self, image_size=256, device='cpu',layer_index = 11):
        super().__init__(image_size, device)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

        alexnet_model =  models.alexnet(pretrained=True).to(self.device)
        self.layer_index = 11
        forward_list = (list(alexnet_model.children())[0][:self.layer_index])
        self.model = nn.Sequential(*forward_list)

        ## setting grad value as false
        for layer in self.model.parameters():
            layer.requires_grad = False


    def get_feature_vector(self, image_list):
        image_list = list(map(Image.fromarray, image_list))
        output_image_list = []
        for img in image_list:
            img = transforms.Scale(256)(img)
            img = transforms.CenterCrop(224)(img)
            img = transforms.ToTensor()(img)
            img = self.normalize(img) 
            output_image_list.append(img.unsqueeze(0))
        final_image_vector = torch.cat(output_image_list, dim=0).to(self.device)
        # print('self.model.device', self.model.device)
        print('final_image_vector.device', final_image_vector.device)
        output = self.model(final_image_vector)
        return output.view(output.shape[0], -1).to(self.device) # output dim: [batch_size, FC_Linear]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='compute identity loss from the facenet model')
    parser.add_argument('--i1', required=True, help='path to image 1')
    parser.add_argument('--i2', required=True, help='path to image 2')
    args = parser.parse_args()

    # device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    print('Running on device', device)

    mtcnn = MTCNN(image_size=1024, device=device)
    resnet = InceptionResnetV1(pretrained='vggface2').eval()

    img1 = Image.open(args.i1)
    img1_cropped = mtcnn(img1)   # get cropped and prewhitened image tensor 

    img2 = Image.open(args.i2)
    img2_cropped = mtcnn(img2)   # get cropped and prewhitened image tensor 

    img1_embedding = resnet(img1_cropped.unsqueeze(0))            # unsqueeze to add the batch dimension
    img2_embedding = resnet(img2_cropped.unsqueeze(0))            # unsqueeze to add the batch dimension

    print('img1_embedding.shape', img1_embedding.shape)
    print('img2_embedding.shape', img2_embedding.shape)

    cos = nn.CosineSimilarity()
    cos_sim = cos(img1_embedding, img2_embedding)

    print('cosine similarity', cos_sim)
