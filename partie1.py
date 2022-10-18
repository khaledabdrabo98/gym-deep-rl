import gym
import matplotlib.pyplot as plt

env = gym.make('CartPole-v1', render_mode="human")
env.action_space.seed(42)

observation, info = env.reset(seed=42)

def run(nb_ep):
    tab_score = []
    for e in range(nb_ep):
        score = 0
        for i in range(100):

            observation, reward, terminated, truncated, info = env.step(env.action_space.sample())

            if terminated or truncated:
                observation, info = env.reset()
                break

            score=score+reward

        tab_score.append(score)
        print(e)
        print(score)
    return tab_score

episodes = 10
score = run(episodes)


print(score)
plt.plot([i+1 for i in range(0, episodes)], score)
plt.xlabel('Episode no.')
plt.ylabel('Score')
#plt.show()



env.close() 
