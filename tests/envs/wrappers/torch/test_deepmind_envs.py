import pytest
import warnings

from collections.abc import Mapping
import gymnasium as gym

import torch

from skrl.envs.wrappers.torch import DeepMindWrapper, wrap_env

from ....utilities import is_running_on_github_actions


def test_env(capsys: pytest.CaptureFixture):
    num_envs = 1
    action = torch.ones((num_envs, 1))

    # load wrap the environment
    try:
        from dm_control import suite
    except ImportError as e:
        if is_running_on_github_actions():
            raise e
        else:
            pytest.skip(f"Unable to import DeepMind environment: {e}")

    original_env = suite.load(domain_name="pendulum", task_name="swingup")
    env = wrap_env(original_env, "auto")
    assert isinstance(env, DeepMindWrapper)
    env = wrap_env(original_env, "dm")
    assert isinstance(env, DeepMindWrapper)

    # check properties
    assert env.state_space is None
    assert isinstance(env.observation_space, gym.Space) and sorted(list(env.observation_space.keys())) == [
        "orientation",
        "velocity",
    ]
    assert isinstance(env.action_space, gym.Space) and env.action_space.shape == (1,)
    assert isinstance(env.num_envs, int) and env.num_envs == num_envs
    assert isinstance(env.num_agents, int) and env.num_agents == 1
    assert isinstance(env.device, torch.device)
    # check internal properties
    assert env._env is original_env
    assert env._unwrapped is original_env
    # check methods
    for _ in range(2):
        observation, info = env.reset()
        assert isinstance(observation, torch.Tensor) and observation.shape == torch.Size([num_envs, 3])
        assert isinstance(info, Mapping)
        for _ in range(3):
            observation, reward, terminated, truncated, info = env.step(action)
            if not is_running_on_github_actions():
                env.render()
            assert isinstance(observation, torch.Tensor) and observation.shape == torch.Size([num_envs, 3])
            assert isinstance(reward, torch.Tensor) and reward.shape == torch.Size([num_envs, 1])
            assert isinstance(terminated, torch.Tensor) and terminated.shape == torch.Size([num_envs, 1])
            assert isinstance(truncated, torch.Tensor) and truncated.shape == torch.Size([num_envs, 1])
            assert isinstance(info, Mapping)

    env.close()
