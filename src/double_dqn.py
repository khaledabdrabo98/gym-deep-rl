import torch
import copy
from .dqn import DQN
from .replay_memory import Experience


# Implement a Target Network and Experince Replay into DQN class
class DoubleDQN(DQN):
    def __init__(self, state_dim, action_dim, hidden_dim=64, learning_rate=0.01):
        super().__init__(state_dim, action_dim, hidden_dim, learning_rate)
        # Target Network
        self.target = copy.deepcopy(self.model)
        
    # Use target network to make predicitons
    def target_predict(self, state):
        with torch.no_grad():
            return self.target(torch.Tensor(state))
        
    # Update target network with the model weights
    def target_update(self):
        self.target.load_state_dict(self.model.state_dict())
        
    # Use experience replay memory in DQN class
    def replay(self, memory, batch_size, gamma=0.99):
        if len(memory) <= batch_size:
            return
        else:
            # Sample experiences from the agent's memory
            experiences = memory.sample(batch_size)
            batch = Experience(*zip(*experiences))

            batch_states = batch.state
            actions = batch.action
            next_states = batch.next_state
            rewards = batch.reward
            dones = batch.done

            states = []
            targets = []

            # Extract datapoints from the batch
            for i in range(batch_size):

                state = batch_states[i]
                action = actions[i]
                next_state = next_states[i]
                reward = rewards[i]
                done = dones[i]

                states.append(state)
                qvalues = self.predict(state).tolist()
                if done:
                    qvalues[action] = reward
                else:
                    # The only difference between the simple replay is in this line
                    # It ensures that next q values are predicted with the target network.
                    next_qvalues = self.target_predict(next_state)
                    # Bellman Equation
                    qvalues[action] = reward + gamma * torch.max(next_qvalues).item()

                targets.append(qvalues)

            if len(states) == len(targets):
                for i in range(len(states)):
                    self.update(states[i], targets[i])
