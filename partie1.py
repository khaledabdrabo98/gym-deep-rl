import gym
from gym.wrappers import RecordVideo
import matplotlib.pyplot as plt
import numpy as np

# Later for Deep QLearning
# import torch


# Random learning search function
def random_search(env, episodes):
    """ Random search strategy implementation."""
    l_rewards = []
    for i in range(episodes):
        state = env.reset()
        done = False
        total = 0
        while not done:
            # Sample a random action from action space
            action = env.action_space.sample()
            # Apply action on environment 
            observation, reward, done, _, _ = env.step(action)
            # Update total reward
            total += reward
            # render environment (needed to record episodes)
            # env.render()
            if done:
                break
        # Add total to the list of rewards
        l_rewards.append(total)
    return l_rewards

# Create and show plot 
def create_plot(title, values, n_episodes):   
    ''' Plot the reward curve and histogram of results over time.'''    
    f, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,5))
    f.suptitle(title)
    ax[0].plot(values, label='score per run')
    ax[0].axhline(195, c='red', ls='--', label='goal')
    ax[0].set_xlabel('Episodes')
    ax[0].set_ylabel('Reward')
    x = range(len(values))
    # Calculate the trend
    try:
        z = np.polyfit(x, values, 1)
        p = np.poly1d(z)
        ax[0].plot(x,p(x),"--", label='trend')
    except:
        print('')
    
    ax[1].hist(values[-n_episodes:])
    ax[1].axvline(195, c='red', label='goal')
    ax[1].set_xlabel("Scores last " + str(n_episodes) + " episodes")
    ax[1].set_ylabel('Frequency')
    plt.show()


# MAIN
env_id = "CartPole-v1"
path_video = './episodes/random_agent/'
plot_title = 'Random Strategy'
n_episodes = 150

# render_mode="human" -> visualize pygame window 
env = gym.make(env_id, render_mode="rgb_array")
env = RecordVideo(env, path_video, episode_trigger=lambda x: x % 25 == 0, name_prefix=format(env_id))
env.action_space.seed(42)

print("Starting with", n_episodes, "episodes ...")

total = random_search(env, n_episodes)
create_plot(plot_title, total, n_episodes)
