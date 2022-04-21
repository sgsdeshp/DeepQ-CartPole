# Submitted by:
# Deshpande, Saurabh Nitin - 201549214 - sgsdeshp@liverpool.ac.uk
# Otitoola, Aanuoluwapo Ayodeji - 201531066 - sgaotito@liverpool.ac.uk
# Nawaz, Shahzaad Karim - 201536031 - sganawaz@liverpool.ac.uk
# Shodipo, Ahmed Ajibola - 201537072 - sgashodi@liverpool.ac.uk


# import necessary dependencies
import random
import gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import RMSprop


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

AGENT_ENV = 'CartPole-v1'


NO_OF_EPISODES = 1000           # games played in training phase
MEMORY_SIZE = 2000              # replay memory size

GAMMA = 0.95                    # future reward discount factor
EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01          # probability of the model to select a non-greedy action.
EXPLORATION_DECAY = 0.999       # the factor by which random action probability decreases
BATCH_SIZE = 64                 # No of samples chosen randomly from replay memory

TRAIN_START = 1000
LEARNING_RATE = 0.001


class DeepQNAgent(object):
    """ Deep Q Network class to train reinforcement learning model.

    Attributes
    -----------

    stateSize : int
       size of state data inputs into network
    action_size : list
       number of possible actions to be taken, corresponding to network outputs
    episodes : int
       number of games played in training phase
    memory : int
       replay memory size
    gamma : float
       discount factor.
    epsilon : int
       probability of the model to select a non-greedy action
    epsilonMin : int
       minimum probability of the model to select a non-greedy action
    epsilonDecay : int
       rate of decrease of the number of explorations as agent gets good at playing games
    batchSize : int
       Size of batches to be passed to the underlying network for training.
    model : int
       neural network model

    """

    # initialize network class
    def __init__(self):

        # prepare the OpenAI Cartpole-v1 Environment
        self.env = gym.make(AGENT_ENV)

        print(f'Max. Episode Steps: {self.env.spec.max_episode_steps}')
        print(f'Reward Threshold: {self.env.spec.reward_threshold}')

        # CartPole-v1 has 500 maximum episode steps
        self.stateSize = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.episodes = NO_OF_EPISODES
        self.memory = deque(maxlen=MEMORY_SIZE)

        # discount rate
        self.gamma = GAMMA

        # exploration rate
        self.epsilon = EXPLORATION_MAX
        self.epsilonMin = EXPLORATION_MIN

        self.epsilonDecay = EXPLORATION_DECAY
        self.batchSize = BATCH_SIZE
        self.trainStart = TRAIN_START

        # create main model
        self.model = self.fetchModel(observation_space=(self.stateSize,), action_space=self.action_size)

    @staticmethod
    def fetchModel(observation_space, action_space):
        """ fetches and returns a new Deep Q-Network model

        Parameters
        ----------
        observation_space: input dimension (No. of Elements in State Tuple (4))
        action_space: dimension of the output layer

        Returns
        -------
        model
            object

        """

        X_input = Input(observation_space)

        # Input Layer of state size(4) and 3 Hidden Layers with 512, 256 and 64 nodes respectively
        X_train = Dense(512, input_shape=observation_space, activation="relu", kernel_initializer='he_uniform')(X_input)
        X_train = Dense(256, activation="relu", kernel_initializer='he_uniform')(X_train)
        X_train = Dense(64, activation="relu", kernel_initializer='he_uniform')(X_train)

        # output layer with 2 actions: 2 nodes (left, right)
        X_train = Dense(action_space, activation="linear", kernel_initializer='he_uniform')(X_train)
        model = Model(inputs=X_input, outputs=X_train, name='CartPole_model')

        # compile the model using the RMSprop optimizer with Initial Learning Rate = 0.00025
        model.compile(loss="mse", optimizer=RMSprop(learning_rate=0.00025, rho=0.95, epsilon=0.01), metrics=["accuracy"])
        model.summary()
        return model

    def remember(self, state, action, reward, nextState, terminal):
        """ store previous experiences and observations.

        Parameters
        ----------
        state: state data taken from an environemnt used to generate actions
        action: selected action
        reward: reward
        nextState: observed next state
        terminal: whether or not game is over

        """
        self.memory.append((state, action, reward, nextState, terminal))
        if len(self.memory) > self.trainStart:
            if self.epsilon > self.epsilonMin:
                self.epsilon *= self.epsilonDecay

    def act(self, state):
        """ predict Q-Values from the model

         Parameters
         ----------
         state: environment state
         """

        # query the model for possible actions and corresponding Q-Values depending on the epsilon value
        if np.random.random() <= self.epsilon:
            # take random action (explore)
            return random.randrange(self.action_size)
        else:
            # select the best action using Q-Values received
            return np.argmax(self.model.predict(state))

    def replayBuffer(self):
        """ store and update the Replay Memory """

        if len(self.memory) < self.trainStart:
            return

        # randomly sample minibatch from the memory
        miniBatch = random.sample(self.memory, min(len(self.memory), self.batchSize))

        state = np.zeros((self.batchSize, self.stateSize))
        nextState = np.zeros((self.batchSize, self.stateSize))
        action, reward, terminal = [], [], []

        # Update the Deep Q-Network Model
        for i in range(self.batchSize):
            state[i] = miniBatch[i][0]
            action.append(miniBatch[i][1])
            reward.append(miniBatch[i][2])
            nextState[i] = miniBatch[i][3]
            terminal.append(miniBatch[i][4])

        # batch prediction
        target = self.model.predict(state)
        target_next = self.model.predict(nextState)

        for i in range(self.batchSize):
            # apply correction on the Q value for the action used
            if terminal[i]:
                target[i][action[i]] = reward[i]
            else:
                # Deep Q Network chooses the max Q value among next actions, Q_max = max_a' Q_target(s', a')
                # update the target with future reward discount factor
                target[i][action[i]] = reward[i] + self.gamma * (np.amax(target_next[i]))

        # train Neural Network with batches
        self.model.fit(state, target, batch_size=self.batchSize, verbose=0)

    def load(self, name):
        """ load saved model

        Parameters
        ----------
        name: name of model

        """
        self.model = load_model(name)

    def save(self, name):
        """ save model as file

        Parameters
        ----------
        name: name of model

        """
        self.model.save(name)

    def train(self):
        """ performs Q-Learning training on the networks """

        scores = []
        for e in range(self.episodes):
            state = self.env.reset()
            state = np.reshape(state, [1, self.stateSize])
            terminal = False
            i = 0
            while not terminal:
                self.env.render()
                action = self.act(state)

                # take the selected action and observe next state
                next_state, reward, terminal, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.stateSize])

                if not terminal or i == self.env.spec.max_episode_steps - 1:
                    reward = reward
                else:
                    reward = -100

                self.remember(state, action, reward, next_state, terminal)
                state = next_state
                i += 1

                # if game over
                if terminal:
                    scores.append(i)
                    print("episode:{:>4}/{:<4}, score:{:>4}, e: {:>.2}".format(e, self.episodes, i, self.epsilon))
                    if i == 500:
                        print("Saving trained model as cartpole-dqn.h5")
                        self.save("cartpole-dqn.h5")
                        self.env.close()
                        return scores

                self.replayBuffer()

        return scores

    def test(self):
        """ test agent performance after training """
        scores = []
        self.load("cartPole-DQN.h5")

        for e in range(self.episodes):
            state = self.env.reset()
            state = np.reshape(state, [1, self.stateSize])
            terminal = False
            i = 0
            while not terminal:
                self.env.render()
                action = np.argmax(self.model.predict(state))

                # take the selected action and observe next state
                next_state, reward, terminal, _ = self.env.step(action)
                state = np.reshape(next_state, [1, self.stateSize])
                i += 1

                # if game over
                if terminal:
                    scores.append(i)
                    print("episode:{:>4}/{:<4}, score:{:>4}".format(e, self.episodes, i))
                    break

        self.env.close()
        return scores

    @staticmethod
    def plotScores(self, scores):
        " plot aggregated scores against epochs "
        version = f'3 hidden layers (Nodes: 512, 264, 64), epsilon = {EXPLORATION_MIN}'
        plt.figure(figsize=(10, 4))
        plt.plot(np.arange(len(scores)), scores, color='tab:purple', marker=11)
        plt.xlabel("Epochs")
        plt.ylabel("Scores")
        plt.title(f'Network - {version}')

        plt.show()

    @staticmethod
    def movingAverage(sumList):
        """ returns an array of moving averages for a given array."""
        averages = []
        for i in range(len(sumList)):
            if i < 5:
                averages.append(-100)
            elif i > len(sumList) - 5:
                averages.append(None)
            else:
                mean = sum(sumList[i - 5: i + 5]) / 10
                averages.append(mean)
        return averages


def main():
    agent = DeepQNAgent()
    scores = agent.train()
    agent.plotScores(scores)
    agent.test()


if __name__ == "__main__":
    main()
