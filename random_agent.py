import gym
from gym.wrappers import RecordVideo
from itertools import count

from src.my_utils import create_plot, plot_durations

ENV_NAME = 'CartPole-v1'


# Random search strategy implementation 
def random_search(env, episodes):
    rewards = []
    episode_durations = []
    for i in range(episodes):
        state = env.reset()
        done = False
        total = 0
        while not done:
            for t in count():
                # Sample a random action from action space
                action = env.action_space.sample()
                # Apply action on environment 
                next_state, reward, done, _, _ = env.step(action)
                # Update total reward
                total += reward
                # render environment (needed to record episodes)
                # env.render()
                if done:
                    episode_durations.append(t + 1)
                    plot_durations(episode_durations)
                    break
        # Add total to the list of rewards
        rewards.append(total)
    return rewards

def main():
    # Random agent
    n_episodes = 150
    path_video = './episodes/random_agent/'
    plot_title = 'Random Strategy'
    # visualize pygame window (UI) change render_mode to "human" 
    env = gym.make(ENV_NAME, render_mode="rgb_array")
    env.action_space.seed(42)
    # Record one episode every 25 episodes
    env = RecordVideo(env, path_video, episode_trigger=lambda x: x % 25 == 0, name_prefix=format(ENV_NAME))
    print("Starting with", n_episodes, "episodes ...")
    total = random_search(env, n_episodes)
    create_plot(plot_title, total, n_episodes)

if __name__ == "__main__":
    main()
