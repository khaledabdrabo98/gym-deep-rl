import minerl
import environment
import gym

env = gym.make("Mineline-v0")

NB_MAX_STEP = env.unwrapped.spec.max_episode_steps
NOOP = None if not hasattr(env.action_space, 'noop') else env.action_space.noop()

for episode in range(100):
    next_state = env.reset()

    for step in range(NB_MAX_STEP):
        state = next_state

        next_state, reward, done, _ = env.step(NOOP)

        print('Episode {:3}  |  Step {:4}  |  Reward {:2}'.format(episode, step, reward))

        if done: break
