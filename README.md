# AI Saturdays: Reinforcement Learning Track
Code and notes from the AI Saturdays (Madrid, 3rd ed.) Reinforcement Learning Track.


## Session 1 - 2020-02-08
For this first session we had to watch the following lectures from CS 285 at UC Berkeley (Deep Reinforcement Learning):
 * [Lecture 1: Introduction and Course Overview](http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-1.pdf)
 * [Lecture 2: Supervised Learning of Behaviors](http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-2.pdf)
 * [Lecture 4: Introduction to Reinforcement Learning](http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-4.pdf)

We also had to take a general look al [Pytorch](https://pytorch.org/docs/stable/index.html), since this would be the DL framework used throughout the course.

During the session we had a look at some basic Reinforcement Learning concepts:
 * Overview of environments, statuses, observations, policies and actions.
 * Commonly used notation
 * Markov chains
 * V and Q functions
 * Model Free and Model Based Reinforcement Learning
 * Imitation Learning

Afterward, we started to code a couple of small examples in [Gym](https://gym.openai.com/). This included:
 * A script that executes a series of episodes of a game with a random agent, which samples actions uniformly from the action space of the environment.
 * A script that allows the user to play the game and record the gameplay (observations, actions and rewards).


## Session 2 - 2020-02-15
For the second session we decided to review the following pages from OpenAI Spinning Up:
 * [Part 1: Key Concepts in RL](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)
 * [Part 2: Kinds of RL Algorithms](https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html)

We also decided to watch the following lecture from David Silver:
 * [Video](https://www.youtube.com/watch?v=lfHX2hHRMVQ)
 * [Slides](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/MDP.pdf)

Finally, we read the 4th chapter from Deep Reinforcement Learning Hands On, by Maxim Lapan, (Chapter4: The Cross-Entropy Method).

Sadly, I could not attend the session personally, but it consisted on reviewing the Cross-Entropy method, analyzing in which environments it would be most applicable, and implementing them for a few environments.


## Session 4 - 2020-02-22
The third session focused on Tabular Learning, the Value Iteration Method, Deep Q-learning and the Deep Q-Networks.

We read the Chapters 5 (Tabular Learning and the Bellman Equation) and 6 (Deep Q-Networks) and watched to following lecture by David Silver:
 * [Video](https://www.youtube.com/watch?v=UoPei5o4fps)
 * [Slides](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/FA.pdf)

During the class, we reviewed the Value Iteration and Q-learning methods, and applied the first one to the Frozen Lake environment from OpenAI Gym, and the second one to the ATARI Pong game, with great success.
