import torch
import gym
from gym.wrappers import RecordVideo
import time
import random
from itertools import count

from src.replay_memory import ReplayMemory
from src.double_dqn import DoubleDQN
from src.my_utils import create_plot, plot_durations


ENV_NAME = 'CartPole-v1'
N_EPISODES = 250
LEARNING_RATE = 0.01
GAMMA = 0.97
EPSILON_START = 0.3
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.9999
BATCH_SIZE = 20
BUFFER_SIZE = 100000
HIDDEN_DIM = 64
TARGET_UPDATE = 20
verbose = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path_video = './episodes/dqn_agent/'
save_model_path = './pretrained_models/dqn_model.pth'
plot_title = 'Deep Q-Network Strategy'

env = gym.make(ENV_NAME, render_mode='rgb_array')
env = RecordVideo(env, path_video, episode_trigger=lambda x: x % 25 == 0, name_prefix=format(ENV_NAME))
env.reset()
# env.render()

# Number of states in observation space
state_dimension = env.observation_space.shape[0]
# Number of actions in action space 
action_dimension = env.action_space.n

dqn = DoubleDQN(state_dimension, action_dimension, hidden_dim=HIDDEN_DIM, learning_rate=LEARNING_RATE).to(device)
memory = ReplayMemory(BUFFER_SIZE)

episode_durations = []
total_rewards = []

# Function that handles updating the epsilon value  
def decrement_epsilon(epsilon):
    # return max(epsilon * EPSILON_DECAY, EPSILON_MIN)
    if epsilon > EPSILON_MIN:
        return epsilon * EPSILON_DECAY
    else:
        return EPSILON_MIN

def main():
    # TRAINING 
    print("Starting with", N_EPISODES, "episodes ...")
    
    episode_counter = 0
    sum_total_replay_time = 0
    epsilon = EPSILON_START

    for i in range(N_EPISODES):
        episode_counter += 1
        
        # Reset state
        state = env.reset()
        done = False
        total = 0
        
        while not done:
            for t in count():
                # ϵ-greedy exploration policy
                # epsilon posibility of choosing a random action,
                # otherwise, we use the Deep Q-Network to obtain
                # the QValues for all possible actions and choose the best
                if random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    qvalues = dqn.predict(state)
                    action = torch.argmax(qvalues).item()
                
                # Apply action on environnement and add reward to total
                next_state, reward, done, _, _ = env.step(action)
                total += reward

                # Needed to make next_state a Tuple (necessary for DQN)
                next_state = (next_state, next_state)
                
                # Update memory 
                memory.push(state, action, next_state, reward, done)
                qvalues = dqn.predict(state).tolist()
                
                if done:
                    total_rewards.append(total)
                    episode_durations.append(t + 1)
                    plot_durations(episode_durations)
                    break
                
                t0 = time.time()
                # Update Policy Network weights using experience replay 
                dqn.replay(memory, BATCH_SIZE, GAMMA)
                t1 = time.time()
                sum_total_replay_time += (t1-t0)

                # Update epsilon
                epsilon = decrement_epsilon(epsilon)

                # Update the target network (by copying all weights and biases in DQN)
                if t % TARGET_UPDATE == 0:
                    dqn.target_update()

                state = next_state
        
        total_rewards.append(total)
        
        if verbose:
            if total >= 500:
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! episode: {}, total reward: {}".format(episode_counter, total))
            else:
                print("episode: {}, total reward: {}, epsilon: {}".format(episode_counter, total, epsilon))
            # print("Average replay time:", sum_total_replay_time/episode_counter)
        

    print('Training complete')

    # env.render()
    env.close()
    # Save model
    torch.save(dqn.state_dict(), save_model_path)
    create_plot(plot_title, total_rewards, N_EPISODES)


if __name__ == "__main__":
    main()
