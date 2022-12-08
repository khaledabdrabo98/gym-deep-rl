import torch
import gym
from DoubleDQN import DoubleDQN
from random import seed
import numpy as np
from Wrappers import make_env
from PIL import Image
import torchvision.transforms as T

# ENV_NAME = 'CartPole-v1'
HIDDEN_DIM = 64
N_EPISODES = 200 

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import minerl
import environment

env = gym.make("Mineline-v0")
env = make_env(env)

NB_MAX_STEP = env.unwrapped.spec.max_episode_steps
NOOP = None if not hasattr(env.action_space, 'noop') else env.action_space.noop()

resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])

def main():
    state = env.unwrapped.reset()
    state = state['pov']

    print("state[pov]", state)
    print("pov shape", state.shape)
    state = np.ascontiguousarray(state, dtype=np.float32) / 255
    print("np.ascontiguousarray", state)
    state = torch.from_numpy(state.copy())
    print("state tensor", state)
    state_after = resize(state).unsqueeze(1)
    print("state unsqueeze", state_after)

    print(env.action_space['attack'])
    print(env.unwrapped.observation_space['pov'])
    
    for episode in range(N_EPISODES):
        next_state = env.unwrapped.reset()
        # state = env.observation_space
        # state = np.ascontiguousarray(state, dtype=np.float32) / 255
        # state = torch.from_numpy(state)
        # Resize, and add a batch dimension (BCHW)
        # state =  resize(state).unsqueeze(0)

        print(state)
        
        for step in range(NB_MAX_STEP):
            
            state = next_state

            next_state, reward, done, _ = env.step(NOOP)

            print('Episode {:3}  |  Step {:4}  |  Reward {:2}'.format(episode, step, reward))

            if done: break

if __name__ == "__main__":
    main()
