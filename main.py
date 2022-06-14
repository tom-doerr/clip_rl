#!/usr/bin/env python3

from ray import tune
from ray.rllib.agents.ppo import PPOTrainer

# create a custom gym env
import gym, ray
from ray.rllib.agents import ppo
import cv2
import numpy as np
import visdom
import torch


class ImageImprovementEnv(gym.Env):
    def __init__(self, env_config):
        IMAGE_PATH = 'image.png'
        # load image
        self.vis = visdom.Visdom()
        self.image = cv2.imread(IMAGE_PATH)
        # convert to 0 to 255
        self.image 
        # print("        self.image:",         self.image)
        image_tensor = torch.from_numpy(self.image)
        # switch channels
        image_tensor = image_tensor.permute(2, 0, 1)
        self.vis.image(image_tensor, win='image')
        image_resolution = self.image.shape
        print("image_resolution:", image_resolution)
        # image_resolution = (512, 512, 3)
        image_low_res = self.get_observation()
        resolution = image_low_res.shape
        print("resolution:", resolution)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=resolution, dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(2)

    def get_observation(self):
        # target resolution 84x84
        image_low_res = cv2.resize(self.image, (84, 84))
        image_low_res_shape = image_low_res.shape
        image_converted_for_visdom = image_low_res.astype(np.uint8)
        image_low_res_channels_switched = image_converted_for_visdom.transpose(2, 0, 1)
        # image_low_res_color_channels_switched = cv2.cvtColor(image_low_res_channels_switched, cv2.COLOR_BGR2RGB)
        self.vis.image(image_low_res_channels_switched, win='image_low_res')
        return image_low_res

    def reset(self):
        return self.get_observation()

    def step(self, action):
        reward = 0
        done = False
        info = {}
        return self.get_observation(), reward, done, info

ray.init()
trainer = ppo.PPOTrainer(env=ImageImprovementEnv, 
        config={
    "env_config": {},  # config to pass to env class
})

while True:
    print(trainer.train())
