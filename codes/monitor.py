import numpy as np
from collections import deque
import torch


def interact(env, agent, brain_name, n_episodes, max_t, eps_start, eps_end, eps_decay):
    """Interaction between agent and environment.

    This function define the interaction between the agent and the openai gym
    environment, and printout the partial results

    Args:
        env: openai gym's environment
        agent: class agent to interact with the environment
        brain_name: String. Name of the agent of the unity environment
        n_episodes: Integer. Maximum number of training episodes
        max_t: Integer. Maximum number of time-steps per episode
        eps_start: Float. Starting value of epsilon, for epsilon-greedy action selection
        eps_end: Float. Minimum value of epsilon
        eps_decay: Float. Multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon

    # Loop the define episodes
    for i_episode in range(1, n_episodes+1):

        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        state = env_info.vector_observations[0]             # get the current state
        score = 0

        # Loop over the maximum number of time-steps per episode
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done = agent.env_step(env, action, brain_name)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward

            # Break the loop if it is final state
            if done:
                break

        scores_window.append(score)        # save most recent score
        scores.append(score)               # save most recent score
        eps = max(eps_end, eps_decay*eps)  # decrease epsilon

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")

        if i_episode % 100 == 0:
            print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}')

        if np.mean(scores_window) >= 13.0:
            print(f'\nEnvironment solved in {i_episode-100:d} episodes!\tAverage Score: {np.mean(scores_window):.2f}')

            # Save model
            torch.save(agent.model_local.state_dict(), 'model/checkpoint.pth')

    return scores

