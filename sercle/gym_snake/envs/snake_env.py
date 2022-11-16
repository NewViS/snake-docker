import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from flask import Flask
from flask_socketio import *

from gym_snake.envs.snake import Snake

import logging

logger = logging.getLogger(__name__)


class SnakeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, blocks=10, block_size=50):
        self.blocks = blocks
        self.width = block_size * blocks
        self.snake = None

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            dtype=np.float32,
            low=np.array([0, 0, 0, -1, -1]),
            high=np.array([1, 1, 1, 1, 1]),
        )

        self.seed()
        self.viewer = None
        self.rewards = None

    def set_rewards(self, rew_step, rew_apple, rew_death, rew_death2, rew_apple_func):
        self.rewards = [rew_step, rew_apple, rew_death, rew_death2, rew_apple_func]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        if action != 0:
            self.snake.direction = self.snake.DIRECTIONS[self.snake.direction[action]]

        info = {}

        self.snake.update()
        info['apple_ate'] = self.snake.apple_ate

        raw_state, reward, done = self.snake.get_raw_state()
        info['apples'] = self.snake.cnt_apples

        state = np.array(raw_state, dtype=np.float32)
        state /= self.blocks

        return state, reward, done, info

    def reset(self):
        if self.rewards:
            self.snake = Snake(self.blocks, self.width // self.blocks, self.np_random,
                               rew_step=self.rewards[0], rew_apple=self.rewards[1],
                               rew_death=self.rewards[2], rew_death2=self.rewards[3],
                               rew_apple_func=self.rewards[4],)
        else:
            self.snake = Snake(self.blocks, self.width // self.blocks, self.np_random)
        raw_state = self.snake.get_raw_state()

        state = np.array(raw_state[0], dtype=np.float32)
        state /= self.blocks

        return state

    def render(self, mode='human'):
        return

    def _create_block(self, w):
        return

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


app=Flask(__name__)
app.config['SECRET_KEY']='secret!'
socketio = SocketIO(app)
env = None

@socketio.on('create_env')
def create_env(self, arg):
    self.env = gym.make('Snake-v0', block_size=arg[0], blocks=arg[1])
    self.env.seed(arg[2])

@socketio.on('connect')
def connect (auth):
    emit ('my response', {'data': 'Connected'})
    print('I connected')

@socketio.on('disconnect')
def disconnect():
    print('Client disconnected')

@socketio.on('render_server')
def getW(self):
    emit('render_client', w = self.snake.blockw)