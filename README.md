# Q-map: a Convolutional Approach for Goal-Oriented Reinforcement Learning

Goal-oriented learning has become a core concept in reinforcement learning (RL), extending the reward signal as a sole way to define tasks. However, as parameterizing value functions with goals increases the learning complexity, efficiently reusing past experience to update estimates towards several goals at once becomes desirable but usually requires independent updates per goal. Considering that a significant number of RL environments can support spatial coordinates as goals, such as on-screen location of the character in ATARI or SNES games, we propose a novel goal-oriented agent called Q-map that utilizes an autoencoder-like neural network to predict the minimum number of steps towards each coordinate in a single forward pass. This architecture is similar to Horde with parameter sharing and allows the agent to discover correlations between visual patterns and navigation. For example learning how to use a ladder in a game could be transferred to other ladders later. We show how this network can be efficiently trained with a 3D variant of Q-learning to update the estimates towards all goals at once. While the Q-map agent could be used for a wide range of applications, we propose a novel exploration mechanism in place of epsilon-greedy that relies on goal selection at a desired distance followed by several steps taken towards it, allowing long and coherent exploratory steps in the environment. We demonstrate the accuracy and generalization qualities of the Q-map agent on a grid-world environment and then demonstrate the efficiency of the proposed exploration mechanism on the notoriously difficult Montezuma's Revenge and Super Mario All-Stars games.

The paper can be found on [arXiv](https://arxiv.org/abs/1810.02927) while videos are available on [the website](https://sites.google.com/view/q-map-rl).

<p align="center"> <img src="data/mario_montezuma.gif" width=40%/> </p>

## Installation

First make sure you have [TensorFlow](https://www.tensorflow.org), [Baselines](https://github.com/openai/baselines), [Gym](https://github.com/openai/gym.git) and [Gym Retro](https://github.com/openai/retro) installed. This code was written for versions ```1.11.0```, ```0.1.5```, ```0.10.5``` and ```0.6.0``` of these libraries.

To install this package, run:
```bash
git clone https://github.com/fabiopardo/qmap.git
cd qmap
pip install -e .
```
and copy the ```SuperMarioAllStars-Snes``` folder to the ```retro/data/stable``` directory where Gym Retro is installed.

## Usage

First, go to the directory where you wish to save the results, for example:
```bash
cd ~/Desktop
```
By default the training scripts will create a ```qmap_results``` folder there.

To train the proposed agent on Super Mario Bros. (All-Stars) level 1.1 you can run:
```bash
python -m qmap.train_mario --render
```
Remove ```--render``` to avoid rendering the episodes (videos are saved in the result folder anyway).
To train only DQN or Q-map use ```--no-qmap``` or ```--no-dqn```.
You can also disable both to get a pure random agent.

Similarly, to train the proposed agent on Montezuma's Revenge you can run:
```bash
python -m qmap.train_montezuma --render
```

Or, to learn Q-frames on the proposed grid world use:
```bash
python -m qmap.train_gridworld --render
```

Those scripts produce images, videos and CSV files in the result folder.
To plot the values contained in the CSV files, run:
```bash
python -m qmap.utils.plot
```
PDF files are produced which can be kept open and refreshed every 10 seconds using for example:
```bash
watch -n10 python -m qmap.utils.plot
```
To filter which environments or agents to plot, use ```--witout``` or ```--only```

To load an agent already trained, run for example:
```bash
python -m qmap.train_mario --load qmap_results/ENV/AGENT/RUN/tensorflow/step_STEP.ckpt --level 2.1
```
where ```ENV``` is the environment used to pre-train (for example on level 1.1) and ```AGENT```, ```RUN``` and ```STEP``` have to be specified.

## BibTeX

To cite this repository in publications please use:
```
@article{pardo2018qmap,
  title={Q-map: a Convolutional Approach for Goal-Oriented Reinforcement Learning},
  author={Pardo, Fabio and Levdik, Vitaly and Kormushev, Petar},
  journal={arXiv preprint arXiv:1810.02927},
  year={2018}
}
```
