import argparse
from time import sleep
import gym
import gym_snake
from rl.model import DQN
import torch
import numpy as np
import keyboard

import random


class Agent:
    def __init__(self, state_size, action_size, pth_path, seed):
        self.model = DQN(state_size, action_size, seed)
        self.model.load_state_dict(torch.load(pth_path))

    def act(self, state):
        state = torch.from_numpy(state).float()
        self.model.eval()
        with torch.no_grad():
            action_values = self.model(state)

        return np.argmax(action_values.data.numpy())


def main(load_path, render, times, seed, block_size, blocks):
    keyboard.add_hotkey('w', activity('UP'))
    keyboard.add_hotkey('a', activity('LEFT'))
    keyboard.add_hotkey('s', activity('DOWN'))
    keyboard.add_hotkey('d', activity('RIGHT'))

    env = get_env(seed, block_size, blocks)
    agent = Agent(env.observation_space.shape[0], env.action_space.n, load_path, seed)
    watch_agent(agent, env, times, render)



def activity(key):
    print(key)

def get_env(seed, block_size, blocks):
    env = gym.make('Snake-v0', block_size=block_size, blocks=blocks)
    env.seed(seed)
    return env


def watch_agent(agent, env, times, render):
    scores = []
    apples = []

    for i in range(1, times + 1):
        state = env.reset()
        score = 0
        steps_after_last_apple = 0
        action = 1
        while True:
            if render:
                env.render()
                #sleep(0.5)
            #action = agent.act(state)
            
            #keyboard.hook(activity)
            #action = random.randint(1,4)

            if keyboard.is_pressed('w'):    action = 1
            if keyboard.is_pressed('s'):    action = 2
            if keyboard.is_pressed('a'):    action = 3
            if keyboard.is_pressed('d'):    action = 4
            sleep(0.5)

            state, reward, done, info = env.step(action)
            score += reward
            if done:
                break

            steps_after_last_apple += 1
            if steps_after_last_apple > 200:
                break
            if info['apple_ate']:
                steps_after_last_apple = 0

        scores.append(score)
        apples.append(info['apples'])
        print(f'\rEpisode {i}\t'
              f'Average apples: {np.mean(apples):.2f}\t'
              f'Average score: {np.mean(scores):.2f}', end='')
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Test trained agent')
    parser.add_argument('--load_path', default='rl/default_model.pth', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--times', default=3, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--blocks', default=10, type=int)
    parser.add_argument('--block_size', default=50, type=int)

    args = parser.parse_args()
    main(
        load_path=args.load_path,
        #render=args.render,
        render = True,
        times=args.times,
        seed=args.seed,
        block_size=args.block_size,
        blocks=args.blocks,
    )
