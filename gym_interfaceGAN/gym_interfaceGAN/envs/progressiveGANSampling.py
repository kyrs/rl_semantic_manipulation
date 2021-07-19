"""
__author__ : Kumar Shubham
__date__   : 28 Nov 2020
__Desc__   : code for sampling images from progressiveGAN
"""
import os 
import sys 
sys.path.insert(0, "/home/kumar/RL_env/gym_interfaceGAN/gym_interfaceGAN/envs/")
print(sys.path)
import os
import argparse
import glob
import pickle

import numpy as np
import torch

from PIL import Image
from library.InterFaceGAN.models.model_settings import MODEL_POOL
from library.InterFaceGAN.models.pggan_generator import PGGANGenerator
import tqdm

def batch_face_image_sampler(model, z):
	assert z.shape[0] <= 4 # ensure that the batch size is less than or equal to 4
	result = model.easy_synthesize(z)
	assert np.array_equal(result['z'], z)
	return result['image']



def linear_sampling(starting_z, diff_vec, noSample, stepSize):
	indVSamples = []
	for i in range (-1*noSample, noSample):
		indVSamples.append( starting_z + i*stepSize*diff_vec)
	return indVSamples	


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--input_folder', required = True, help = 'path to the input folder')
	parser.add_argument('--save_folder', required = True)
	parser.add_argument('--attribute', required = True, help = "attribute to be processed")
	parser.add_argument('--noSample', required = True, help = "no of images to sample in a given direction")
	parser.add_argument('--stepSize', required = True, help = "step size for give traj")

	args = parser.parse_args()
	t = args.sub_folder_prefix


	vec = list()
	folder_names = list()
	try:
		for file_name in glob.glob(args.input_folder + f'/{t}*/*.npy'):
			step_list = np.load(file_name, allow_pickle=True).item().get('step_list')
			inp_z_npy = step_list[0][512:1024]
			vec.append(inp_z_npy)
			folder_name = os.path.basename(os.path.dirname(file_name))
			folder_names.append(folder_name)
	except:
		print('error occured: could NOT process all the trajectories')


	model =  PGGANGenerator('pggan_celebahq', logger=None)

	for z_vec, folder_name in tqdm(zip(vec, folder_names)):
		linearValOut = linear_sampling(z_vec, diff_vec, noSample, stepSize)
		for count,indz in tqdm(enumerate( linearValOut)):
			image =  batch_face_image_sampler(model, indz).squeeze(0)
			Image.fromarray(np.uint8(base_image)).convert('RGB')
			baseImage.save(os.path.join(save_adds, f"folder_name{count}.jpg"))