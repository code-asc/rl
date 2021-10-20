from dynaqplus import DynaAgentPlus
from dynaq import DynaAgent
import matplotlib.pyplot as plt

EXECUTE_DYNAQ = False
EXECUTE_DYNAQ_PLUS = True

if __name__ == "__main__":

	N_EPISODES = 3000

	agent = DynaAgent(n_steps=5, episodes=N_EPISODES, enable_change_env=True, enable_after=20)
	agent.play()

	steps_episode_5 = agent.steps_per_episode

	agent = DynaAgentPlus(n_steps=5, episodes=N_EPISODES, enable_change_env=True, enable_after=20)
	agent.play()

	steps_episode_5_plus = agent.steps_per_episode


	plt.figure(figsize=[10, 6])
	plt.ylim(0, 2000)
	plt.plot(range(N_EPISODES), steps_episode_5, label="step=5 dynaq")
	plt.plot(range(N_EPISODES), steps_episode_5_plus, label="step=5 dynaq+")

	plt.legend()
	plt.show()