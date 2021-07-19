import gym
import gym_interfaceGAN

env = gym.make('InterfaceGAN-v0') 
env.reset()


for i in range(40):
    action = env.action_space.sample()
    env.step(action)
    env.render()
# for env in gym.envs.registry.all():
#     # TODO: solve this with regexes
#     # env_type = env.entry_point.split(':')[0].split('.')[-1]
#     print (env)
