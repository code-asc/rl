from dynaqplus import DynaAgentPlus
from dynaq import DynaAgent
import matplotlib.pyplot as plt

EXECUTE_DYNAQ = False
EXECUTE_DYNAQ_PLUS = True

if __name__ == "__main__":

	N_EPISODES = 100

	if EXECUTE_DYNAQ:
	    # comparison
	    agent = DynaAgent(n_steps=0, episodes=N_EPISODES)
	    agent.play()

	    steps_episode_0 = agent.steps_per_episode

	    agent = DynaAgent(n_steps=5, episodes=N_EPISODES)
	    agent.play()

	    steps_episode_5 = agent.steps_per_episode

	    agent = DynaAgent(n_steps=50, episodes=N_EPISODES)
	    agent.play()

	    steps_episode_50 = agent.steps_per_episode

	    agent = DynaAgent(n_steps=100, episodes=N_EPISODES)
	    agent.play()

	    steps_episode_100 = agent.steps_per_episode

	    plt.figure(figsize=[10, 6])

	    plt.ylim(0, 2000)
	    plt.plot(range(N_EPISODES), steps_episode_0, label="step=0")
	    plt.plot(range(N_EPISODES), steps_episode_5, label="step=5")
	    plt.plot(range(N_EPISODES), steps_episode_50, label="step=50")
	    plt.plot(range(N_EPISODES), steps_episode_100, label="step=100")

	    plt.legend()


	if EXECUTE_DYNAQ_PLUS:
	    # comparison
	    agent = DynaAgentPlus(n_steps=0, episodes=N_EPISODES)
	    agent.play()

	    steps_episode_0 = agent.steps_per_episode

	    agent = DynaAgentPlus(n_steps=5, episodes=N_EPISODES)
	    agent.play()

	    steps_episode_5 = agent.steps_per_episode

	    agent = DynaAgentPlus(n_steps=50, episodes=N_EPISODES)
	    agent.play()

	    steps_episode_50 = agent.steps_per_episode

	    agent = DynaAgentPlus(n_steps=100, episodes=N_EPISODES)
	    agent.play()

	    steps_episode_100 = agent.steps_per_episode

	    plt.figure(figsize=[10, 6])

	    plt.ylim(0, 2000)
	    plt.plot(range(N_EPISODES), steps_episode_0, label="step=0")
	    plt.plot(range(N_EPISODES), steps_episode_5, label="step=5")
	    plt.plot(range(N_EPISODES), steps_episode_50, label="step=50")
	    plt.plot(range(N_EPISODES), steps_episode_100, label="step=100")

	    plt.legend()
	plt.show()


