import flappy_bird_gym

import random
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import load_model
from tensorflow.keras.models import save_model
from tensorflow.keras.models import Sequential

from tensorflow.keras.optimizers import RMSprop

class DQNAgent:
    def __init__(self):
        self.env = flappy_bird_gym.make("FlappyBird-v0")
        self.episodes = 1000
        self.state_space = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space.n
        self.memory = deque(maxlen = 2000)

        self.gamma = 0.95
        self.epsilon = 1
        self.epsilon_decay = 0.0000
        self.min_epsilon = 0.01
        self.batch_size = 64

        self.train_start = 1000
        self.jump_prob = 0.01

        self.model = NeuralNetwork(input_shape = (self.state_space,),
                                   output_shape = self.action_space)


    def act(self, state):
        if np.random.random() > self.epsilon:
            return np.argmax(self.model.predict(state))

        if np.random.random() < self.jump_prob:
            return 1
        else:
            return 0

    def learn(self, state):
        if len(self.memory) < self.train_start:
            return
        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))

        state = np.zeros((self.batch_size, self.state_space))
        next_state = np.zeros((self.batch_size, self.state_space))
        action, reward, done = [], [], []

        for i in range(self.batch_size):
            state[i] = minibatch[i][0]
            action.append(minibatch[i][1])
            reward.append(minibatch[i][2])
            next_state[i] = minibatch[i][3]
            done.append(minibatch[i][4])

        target = self.model.predict(state)
        target_next = self.model.predict(next_state)

        for i in range(self.batch_size):
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + self.gamma * (np.amax(target_next[i]))

        self.model.fit(state, target, batch_size= self.batch_size, verbose = 0)

    def train(self):
        for i in range(self.episodes):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_space])
            done = False
            score = 0

            if self.epsilon * self.epsilon > self.min_epsilon:
                self.epsilon = self.epsilon * self.epsilon_decay
            else:
                self.epsilon = self.min_epsilon

                while not done:
                    self.env.render()
                    action = self.act(state)
                    next_state, reward, done, info = self.env.step(action)

                    next_state = np.reshape(next_state, [1, self.state_space])
                    score += 1

                    if done:
                        reward -= 100

                    self.memory.append((state, action, reward, next_state, done))
                    state = next_state
                    if done:
                        print('Episode: {}\nScore: {}\nEpsilon: {:.2}'.format(i, score, self.epsilon))

                        if score >= 1000:
                            self.model.save('flappybrain.h5')
                            return
                    self.learn(state)

    def perform(self):
        self.model = load_model('flappybrain.h5')
        while 1:
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_space])
            done = False
            score = 0

            while not done:
                self.env.render()
                action = np.argmax(self.model.predict(state))
                next_state, reward, done, info = self.env.step(action)
                state = np.reshape(next_state, [1, self.state_space])
                score += 1

                print('Current Score: {}'.format(score))

                if done:
                    print("DEAD")
                    break

def NeuralNetwork(input_shape, output_shape):
    model = Sequential()
    model.add(Dense(512,
                    input_shape = input_shape,
                    activation='relu',
                    kernel_initializer='he_uniform'))

    model.add(Dense(256,
                    activation='relu',
                    kernel_initializer='he_uniform'))

    model.add(Dense(64,
                    activation='relu',
                    kernel_initializer='he_uniform'))

    model.add(Dense(output_shape,
                    activation='linear',
                    kernel_initializer='he_uniform'))

    model.compile(loss = 'mse',
                  optimizer= RMSprop(lr = 0.0001,
                                     rho = 0.95,
                                     epsilon = 0.01),
                  metrics=["accuracy"])

    model.summary()
    return model




if __name__ == '__main__':
    agent = DQNAgent()
    # agent.train()
    agent.perform()