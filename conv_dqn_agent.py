import torch
import torch.nn as nn
import torch.optim as optim
import gym
from gym.wrappers import RecordVideo
import random
from itertools import count

from src.replay_memory import ReplayMemory, Experience
from src.conv_dqn import DQCN
from src.my_utils import get_screen, create_plot, plot_durations


ENV_NAME = 'CartPole-v1'
N_EPISODES = 250
GAMMA = 0.999
EPSILON_START = 0.3
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.99
BATCH_SIZE = 20
BUFFER_SIZE = 100000
HIDDEN_DIM = 64
TARGET_UPDATE = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path_video = './episodes/conv_dqn_agent/'
save_model_path = './pretrained_models/conv_dqn_model.pth'
plot_title = 'Deep Q-Conv-Network Strategy'

env = gym.make(ENV_NAME, render_mode='rgb_array')
env = RecordVideo(env, path_video, episode_trigger=lambda x: x % 25 == 0, name_prefix=format(ENV_NAME))
env.reset()
# env.render()

# Get screen size so that we can initialize layers correctly based on shape
# returned from AI gym. Typical dimensions at this point are close to 3x40x90
# which is the result of a clamped and down-scaled render buffer in get_screen()
init_screen = get_screen(env)
_, _, screen_height, screen_width = init_screen.shape

# Get number of actions from gym action space
n_actions = env.action_space.n

policy_net = DQCN(screen_height, screen_width, n_actions, device).to(device)
target_net = DQCN(screen_height, screen_width, n_actions, device).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(BUFFER_SIZE)

episode_durations = []
total_rewards = []

def decrement_epsilon(epsilon):
    # return max(epsilon * EPSILON_DECAY, EPSILON_MIN)
    if epsilon > EPSILON_MIN:
        return epsilon * EPSILON_DECAY
    else:
        return EPSILON_MIN

def select_action(state, epsilon):
    # ϵ-greedy exploration policy
    # epsilon posibility of choosing a random action,
    # otherwise, we use the Deep Q-Network to obtain
    # the QValues for all possible actions and choose the best
    if random.random() < epsilon:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
    else:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)

# Update Target Network with the Policy Network weights
def target_update():
    target_net.load_state_dict(policy_net.state_dict())

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    experiences = memory.sample(BATCH_SIZE)
    batch = Experience(*zip(*experiences))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def main():
    # TRAINING
    print("Starting with", N_EPISODES, "episodes ...")
    epsilon = EPSILON_START

    for i in range(N_EPISODES):
        # Initialize the environment and state
        env.reset()
        last_screen = get_screen(env)
        current_screen = get_screen(env)
        state = current_screen - last_screen
        total = 0

        for t in count():
            # Select and perform an action
            action = select_action(state, epsilon)
            _, reward, done, _, _ = env.step(action.item())
            total += reward
            reward = torch.tensor([reward], device=device)

            # Observe new state
            last_screen = current_screen
            current_screen = get_screen(env)
            if not done:
                next_state = current_screen - last_screen
            else:
                next_state = None

            # Store the experience in memory
            memory.push(state, action, next_state, reward, done)

            # Move to the next state
            state = next_state

            if done:
                total_rewards.append(total)
                print("epsiode", i, ":", total)
                episode_durations.append(t + 1)
                plot_durations(episode_durations)
                break

            # Perform one step of the optimization (on the policy network)
            optimize_model()

            # Update epsilon
            epsilon = decrement_epsilon(epsilon)

            # Update the target network, copying all weights and biases in DQN
            if t % TARGET_UPDATE == 0:
                target_update()

    print('Training complete')
    # env.render()
    env.close()
    torch.save(policy_net.state_dict(), save_model_path)
    create_plot(plot_title, total_rewards, N_EPISODES)
    average_rewards = sum(total_rewards) / len(total_rewards)
    print(average_rewards)
    print(max(total_rewards))


if __name__ == "__main__":
    main()
