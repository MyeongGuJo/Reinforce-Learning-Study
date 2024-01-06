import torch
from pathlib import Path
import datetime, os

import gym
from gym.wrappers import FrameStack
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros

from env.environment import (
    SkipFrame,
    GrayScaleObservation,
    ResizeObservation,
)
from agent.agent import Mario
from metric.metirc_logger import MetricLogger

# initialize super-mario-bros environment
if gym.__version__ < '0.26':
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", new_step_api=True)
else:
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", render_mode='rgb', apply_api_compatibility=True)

# 상태 공간을 2가지로 제한
#   0. 오른쪽으로 걷기
#   1. 오른쪽으로 점프
env = JoypadSpace(env, [["right"], ["right", "A"]])

env.reset()
next_state, reward, done, trunc, info = env.step(action=0)
print(f"{next_state.shape},\n {reward},\n {done},\n {info}")

# apply wrapper to envrironment
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=84)

if gym.__version__ < '0.26':
    env = FrameStack(env, num_stack=4, new_step_api=True)
else:
    env = FrameStack(env, num_stack=4)


# excute game
use_cuda = torch.cuda.is_available()
print(f"Using CUDA: {use_cuda}")
print()

save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)

mario = Mario(
    state_dim=(4, 84, 84),
    action_dim=env.action_space.n,
    save_dir=save_dir
)

logger = MetricLogger(save_dir)

episodes = 10
for e in range(episodes):

    state = env.reset()

    while True:

        # Agent
        action = mario.act(state)

        # Agent processes action
        next_state, reward, done, trunc, info = env.step(action)

        # memorize
        mario.cache(state, next_state, action, reward, done)

        # learn
        q, loss = mario.learn()

        # record
        logger.log_step(reward, loss, q)

        # update state
        state = next_state

        # check whether Agent finish the game or not
        if done or info["flag_get"]:
            break
    
    logger.log_episode()

    if e % 20 == 0:
        logger.record(
            episode=e,
            epsilon=mario.exploration_rate,
            step=mario.curr_step,
        )