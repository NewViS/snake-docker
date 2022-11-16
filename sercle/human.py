import argparse
from time import sleep
import gym
import socketio
#import gym_snake.envs.snake_env
import numpy as np
import keyboard

sio = socketio.Client()

@sio.event
def connect():
    print("I'm connected!")

@sio.event
def connect_error(data):
    print("The connection failed!")

@sio.event
def disconnect():
    print("I'm disconnected!")

def main(load_path, render, times, seed, block_size, blocks):
    sio.connect('http://127.0.0.1:5000',  wait=True, wait_timeout= 5)

    #env = get_env(seed, block_size, blocks)
    sio.emit('create_env', [block_size,blocks,seed])
    
    watch_agent(times, render)

@sio.on('render_client')
def render(self, w):
        from gym.envs.classic_control import rendering
        #w = self.snake.blockw
        

        if self.viewer is None:
            self.viewer = rendering.Viewer(self.width, self.width)
            apple = self._create_block(w)
            self.apple_trans = rendering.Transform()
            apple.add_attr(self.apple_trans)
            apple.set_color(*self.snake.apple.color)
            self.viewer.add_geom(apple)

            head = self._create_block(w)
            self.head_trans = rendering.Transform()
            head.add_attr(self.head_trans)
            head.set_color(*self.snake.head.color)
            self.viewer.add_geom(head)

            self.body = []
            for i in range(len(self.snake.body)):
                body = self._create_block(w)
                body_trans = rendering.Transform()
                body.add_attr(body_trans)
                body.set_color(*self.snake.body[0].color)

                self.body.append(body_trans)
                self.viewer.add_geom(body)

        self.apple_trans.set_translation(self.snake.apple.x, self.snake.apple.y)
        self.head_trans.set_translation(self.snake.head.x, self.snake.head.y)

        if len(self.snake.body) > len(self.body):
            body = self._create_block(w)
            body_trans = rendering.Transform()
            body.add_attr(body_trans)
            body.set_color(*self.snake.body[0].color)

            self.body.append(body_trans)
            self.viewer.add_geom(body)
        elif len(self.snake.body) < len(self.body):
            self.body, trash = self.body[len(self.body) - len(self.snake.body):], \
                               self.body[:len(self.body) - len(self.snake.body)]
            for i in range(len(trash)):
                trash[i].set_translation(-w, -w)

        for i in range(len(self.body)):
            self.body[i].set_translation(self.snake.body[i].x, self.snake.body[i].y)

        self.viewer.render()

def _create_block(self, w):
        from gym.envs.classic_control import rendering
        return rendering.FilledPolygon([(0, 0), (0, w), (w, w), (w, 0)])

def activity(key):
    print(key)

def get_env(seed, block_size, blocks):
    env = gym.make('Snake-v0', block_size=block_size, blocks=blocks)
    env.seed(seed)
    return env


def watch_agent(times, render):
    scores = []
    apples = []

    for i in range(1, times + 1):
        sio.emit('env_reset')
        score = 0
        steps_after_last_apple = 0
        action = 1
        while True:
            if render:
                #env.render()
                sio.emit('render_server')
                #sleep(0.5)
            #action = agent.act(state)
            
            #keyboard.hook(activity)
            #action = random.randint(1,4)
            
            sleep(0.5)
            if keyboard.is_pressed('w'):    action = 1
            if keyboard.is_pressed('s'):    action = 2
            if keyboard.is_pressed('a'):    action = 3
            if keyboard.is_pressed('d'):    action = 4

            sio.emit('do_step', action)
            """state, reward, done, info = env.step(action)
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
              f'Average score: {np.mean(scores):.2f}', end='')"""
    
    sio.emit('env_close')


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


