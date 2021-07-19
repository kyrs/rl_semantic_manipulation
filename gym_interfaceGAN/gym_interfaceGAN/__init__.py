from gym.envs.registration import register

register(id='InterfaceGAN-v0', 
    entry_point='gym_interfaceGAN.envs:InterfaceGanEnv', 
)