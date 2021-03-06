# Banana's Collector

Unity Machine Learning Agents (ML-Agents) is an open-source Unity plugin that enables games and simulations to serve as environments for training intelligent agents.

For game developers, these trained agents can be used for multiple purposes, including controlling NPC behavior (in a variety of settings such as multi-agent and adversarial), automated testing of game builds and evaluating different game design decisions pre-release.

## The Environment

For this project, the agent was trained to navigate (and collect bananas!) in a large, square world.

![](images/banana.gif)

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

- 0 - move forward.
- 1 - move backward.
- 2 - turn left.
- 3 - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

## Installation

Use the [docker](https://www.docker.com) to test the algorithm.

```bash
docker pull fernandofsilva/banana_collector 
```


## Usage

Run the container to start the jupyter notebook server

```bash
docker run -t -p 8888:8888 fernandofsilva/banana_collector 
```

And you can access the notebook through the link is provided.

To train the agent with different parameters, you just need to access the notebook, change the parameters on the sections
4.1 and 4.2 and check the results on section 4.3.


## Scores

Below, there are the scores during training of the neural network, the environment was solved in the episode 432. 

![](images/scores.png)


## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.


## License

[MIT](https://choosealicense.com/licenses/mit/)