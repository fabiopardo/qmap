# Scaling All-Goals Updates in Reinforcement Learning Using Convolutional Neural Networks

Being able to reach any desired location in the environment can be a valuable asset for an agent. Learning a policy to navigate between all pairs of states individually is often not feasible. An all-goals updating algorithm uses each transition to learn Q-values towards all goals simultaneously and off-policy. However the expensive numerous updates in parallel limited the approach to small tabular cases so far. To tackle this problem we propose to use convolutional network architectures to generate Q-values and updates for a large number of goals at once. We demonstrate the accuracy and generalization qualities of the proposed method on randomly generated mazes and Sokoban puzzles. In the case of on-screen goal coordinates the resulting mapping from frames to distance-maps directly informs the agent about which places are reachable and in how many steps. As an example of application we show that replacing the random actions in epsilon-greedy exploration by several actions towards feasible goals generates better exploratory trajectories on Montezuma's Revenge and Super Mario All-Stars games.

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
@inproceedings{pardo2020scaling,
  title={Scaling All-Goals Updates in Reinforcement Learning Using Convolutional Neural Networks},
  author={Pardo, Fabio and Levdik, Vitaly and Kormushev, Petar},
  booktitle={Thirty-Fourth AAAI Conference on Artificial Intelligence},
  year={2020}
}
```
