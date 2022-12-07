import torch
import gym
from DoubleDQN import DoubleDQN
from random import seed

ENV_NAME = 'CartPole-v1'
HIDDEN_DIM = 64
N_EPISODES = 200  # not really episodes since the model does not learn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pretrained_model_path = './pretrained_models/dqn_model.pth'


def main():
    env = gym.make(ENV_NAME, render_mode='human')

    # Number of states in observation space
    state_dimension = env.observation_space.shape[0]
    # Number of actions in action space
    action_dimension = env.action_space.n

    # Create DoubleDQN model and load it with saved model weights
    model = DoubleDQN(state_dimension, action_dimension).to(device)
    model.load_state_dict(torch.load(pretrained_model_path))
    model.eval()
    episode_counter = 0

    while True:
        state = env.reset()
        interactions = 0
        episode_counter += 1

        for _ in range(N_EPISODES):
            interactions += 1
            env.render()
            qvalues = model.predict(state)
            action = torch.argmax(qvalues).item()
            state, _, done, _, _ = env.step(action)

            if done:
                break

            state = (state, state)
        print("Iteration", episode_counter, "/", N_EPISODES,
              "finished with", interactions, "interactions")


if __name__ == "__main__":
    main()
