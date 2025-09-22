from typing import Any, Mapping, Optional, Tuple, Union, Dict, List
import copy
import itertools
import gymnasium
from packaging import version

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

from skrl import config, logger
from skrl.agents.torch import Agent
from skrl.memories.torch import Memory
from skrl.models.torch import Model
from skrl.resources.schedulers.torch import KLAdaptiveLR


# [start-config-dict-torch]
AGGRE_PPO_DEFAULT_CONFIG = {
    "rollouts": 16,
    "learning_epochs": 8,
    "mini_batches": 2,
    "discount_factor": 0.99,
    "lambda": 0.95,
    "learning_rate": 1e-3,
    "learning_rate_scheduler": None,
    "learning_rate_scheduler_kwargs": {},
    "state_preprocessor": None,
    "state_preprocessor_kwargs": {},
    "value_preprocessor": None,
    "value_preprocessor_kwargs": {},
    "random_timesteps": 0,
    "learning_starts": 0,
    "grad_norm_clip": 0.5,
    "ratio_clip": 0.2,
    "value_clip": 0.2,
    "clip_predicted_values": False,
    "entropy_loss_scale": 0.01,
    "value_loss_scale": 1.0,
    "goal_loss_scale": 0.1,
    "kl_threshold": 0,
    "goal_space_size": 16,  # 目标空间维度
    "goal_sampling_std": 0.3,  # 目标采样标准差
    "goal_learning_rate": 1e-4,  # 目标生成器学习率
    "goal_update_frequency": 4,  # 目标更新频率
    "rewards_shaper": None,
    "time_limit_bootstrap": False,
    "mixed_precision": False,
    "experiment": {
        "directory": "",
        "experiment_name": "",
        "write_interval": "auto",
        "checkpoint_interval": "auto",
        "store_separately": False,
        "wandb": False,
        "wandb_kwargs": {},
    },
}
# [end-config-dict-torch]


class GoalGenerator(Model):
    """AGGRE 目标生成器 - 通过对抗训练生成适合的目标"""

    def __init__(self, observation_space, action_space, device=None, goal_space_size=16):
        super().__init__(observation_space, action_space, device)

        # 获取状态维度
        if isinstance(observation_space, gymnasium.Space):
            state_dim = observation_space.shape[0]
        else:
            state_dim = observation_space

        self.goal_space_size = goal_space_size

        # 目标生成器网络
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, goal_space_size * 2),  # 输出均值和方差
        )

    def compute(self, inputs, role):
        states = inputs.get("states")
        if states is None:
            raise ValueError("States must be provided for GoalGenerator")

        # 生成目标的均值和方差
        output = self.net(states)
        mean = output[:, : self.goal_space_size]
        log_std = output[:, self.goal_space_size :]

        return mean, log_std, {}

    def sample_goals(self, states):
        """采样目标"""
        mean, log_std, _ = self.compute({"states": states}, role="goal_generator")
        std = log_std.exp()
        distribution = dist.Normal(mean, std)
        goals = distribution.rsample()  # 重参数化采样
        return goals, distribution.log_prob(goals)


class GoalDiscriminator(Model):
    """目标判别器 - 判断目标是否适合当前策略水平"""

    def __init__(self, observation_space, action_space, device=None, goal_space_size=16):
        super().__init__(observation_space, action_space, device)

        # 获取状态维度
        if isinstance(observation_space, gymnasium.Space):
            state_dim = observation_space.shape[0]
        else:
            state_dim = observation_space

        # 输入: 状态 + 目标
        total_dim = state_dim + goal_space_size

        self.net = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),  # 输出目标适合的概率
        )

    def compute(self, inputs, role):
        states = inputs.get("states")
        goals = inputs.get("goals")

        if states is None or goals is None:
            raise ValueError("Both states and goals must be provided for GoalDiscriminator")

        # 拼接状态和目标
        x = torch.cat([states, goals], dim=-1)
        suitability = self.net(x)

        return suitability, torch.tensor(0.0, device=states.device), {}


class GoalConditionedPolicy(Model):
    """目标条件策略 - 基于状态和目标生成动作"""

    def __init__(self, observation_space, action_space, device=None, goal_space_size=16):
        super().__init__(observation_space, action_space, device)

        # 获取状态维度
        if isinstance(observation_space, gymnasium.Space):
            state_dim = observation_space.shape[0]
        else:
            state_dim = observation_space

        # 总输入维度 = 状态 + 目标
        total_dim = state_dim + goal_space_size

        self.feature_net = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

        self.mean_layer = nn.Linear(64, action_space.shape[0])
        self.log_std_layer = nn.Parameter(torch.zeros(action_space.shape[0]))

    def compute(self, inputs, role):
        states = inputs.get("states")
        goals = inputs.get("goals")

        if states is None or goals is None:
            raise ValueError("Both states and goals must be provided for GoalConditionedPolicy")

        # 拼接状态和目标
        x = torch.cat([states, goals], dim=-1)
        features = self.feature_net(x)

        mean = self.mean_layer(features)
        log_std = self.log_std_layer.expand_as(mean)

        return mean, log_std, {}


class AGGRE_PPO(Agent):
    def __init__(
        self,
        models: Mapping[str, Model],
        memory: Optional[Union[Memory, Tuple[Memory]]] = None,
        observation_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        action_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        device: Optional[Union[str, torch.device]] = None,
        cfg: Optional[dict] = None,
    ) -> None:
        _cfg = copy.deepcopy(AGGRE_PPO_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})
        super().__init__(
            models=models,
            memory=memory,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            cfg=_cfg,
        )

        # models
        self.policy = self.models.get("policy", None)
        self.value = self.models.get("value", None)
        self.goal_generator = self.models.get("goal_generator", None)
        self.goal_discriminator = self.models.get("goal_discriminator", None)

        # 配置参数
        self._goal_space_size = self.cfg["goal_space_size"]
        self._goal_sampling_std = self.cfg["goal_sampling_std"]
        self._goal_learning_rate = self.cfg["goal_learning_rate"]
        self._goal_update_frequency = self.cfg["goal_update_frequency"]
        self._goal_loss_scale = self.cfg["goal_loss_scale"]

        # PPO 参数
        self._learning_epochs = self.cfg["learning_epochs"]
        self._mini_batches = self.cfg["mini_batches"]
        self._rollouts = self.cfg["rollouts"]
        self._rollout = 0

        # 初始化优化器
        self._setup_optimizers()

        # 内部状态
        self._current_goals = None
        self._current_log_prob = None
        self._current_values = None

    def _setup_optimizers(self):
        """设置优化器"""
        if self.policy is not None and self.value is not None:
            self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.cfg["learning_rate"])
            self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=self.cfg["learning_rate"])

        if self.goal_generator is not None:
            self.goal_optimizer = torch.optim.Adam(self.goal_generator.parameters(), lr=self._goal_learning_rate)

        if self.goal_discriminator is not None:
            self.discriminator_optimizer = torch.optim.Adam(
                self.goal_discriminator.parameters(), lr=self._goal_learning_rate
            )

    def init(self, trainer_cfg: Optional[Mapping[str, Any]] = None) -> None:
        """Initialize the agent"""
        super().init(trainer_cfg=trainer_cfg)
        self.set_mode("eval")

        # 创建内存张量
        if self.memory is not None:
            self.memory.create_tensor(name="states", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="actions", size=self.action_space, dtype=torch.float32)
            self.memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="terminated", size=1, dtype=torch.bool)
            self.memory.create_tensor(name="truncated", size=1, dtype=torch.bool)
            self.memory.create_tensor(name="log_prob", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="values", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="returns", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="advantages", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="goals", size=self._goal_space_size, dtype=torch.float32)

    def act(self, states: torch.Tensor, timestep: int, timesteps: int) -> torch.Tensor:
        """生成动作"""
        if timestep < self.cfg["random_timesteps"]:
            return self.policy.random_act({"states": states}, role="policy")

        # 生成或采样目标
        if timestep % self._goal_update_frequency == 0:
            with torch.no_grad():
                if self.goal_generator is not None:
                    goals, _ = self.goal_generator.sample_goals(states)
                    self._current_goals = goals
                else:
                    # 如果没有目标生成器，使用随机目标
                    self._current_goals = (
                        torch.randn(states.shape[0], self._goal_space_size, device=states.device)
                        * self._goal_sampling_std
                    )

        # 基于状态和目标生成动作
        actions, log_prob, _ = self.policy.act({"states": states, "goals": self._current_goals}, role="policy")

        self._current_log_prob = log_prob

        # 获取价值估计
        values, _, _ = self.value.act({"states": states}, role="value")
        self._current_values = values

        return actions, log_prob, values

    def record_transition(
        self, states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps
    ):
        """记录转移"""
        super().record_transition(
            states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps
        )

        if self.memory is not None:
            # 存储到内存
            self.memory.add_samples(
                states=states,
                actions=actions,
                rewards=rewards,
                next_states=next_states,
                terminated=terminated,
                truncated=truncated,
                log_prob=self._current_log_prob,
                values=self._current_values,
                goals=self._current_goals,
            )

    def _update(self, timestep: int, timesteps: int) -> None:
        """主要更新步骤"""
        # 1. 首先更新策略和价值网络 (标准 PPO)
        self._update_policy_value()

        # 2. 更新目标判别器
        self._update_discriminator()

        # 3. 更新目标生成器 (对抗训练)
        self._update_goal_generator()

    def _update_policy_value(self):
        """更新策略和价值网络 (标准 PPO 更新)"""
        # 这里实现标准的 PPO 更新逻辑
        # 包括计算 advantage、policy loss、value loss 等
        pass

    def _update_discriminator(self):
        """更新目标判别器"""
        if self.goal_discriminator is None:
            return

        # 从内存中采样数据
        states = self.memory.get_tensor_by_name("states")
        goals = self.memory.get_tensor_by_name("goals")
        rewards = self.memory.get_tensor_by_name("rewards")

        # 计算目标适合的标签 (基于奖励)
        # 奖励越高，目标越适合
        labels = torch.sigmoid(rewards * 10)  # 缩放奖励到 [0,1] 范围

        # 判别器前向传播
        suitability, _, _ = self.goal_discriminator.act({"states": states, "goals": goals}, role="goal_discriminator")

        # 二元交叉熵损失
        loss = F.binary_cross_entropy(suitability, labels)

        # 更新判别器
        self.discriminator_optimizer.zero_grad()
        loss.backward()
        self.discriminator_optimizer.step()

    def _update_goal_generator(self):
        """更新目标生成器 (对抗训练)"""
        if self.goal_generator is None or self.goal_discriminator is None:
            return

        # 从内存中采样状态
        states = self.memory.get_tensor_by_name("states")

        # 生成新目标
        new_goals, goal_log_prob = self.goal_generator.sample_goals(states)

        # 判别器评估新目标
        with torch.no_grad():
            suitability, _, _ = self.goal_discriminator.act(
                {"states": states, "goals": new_goals}, role="goal_discriminator"
            )

        # AGGRE 损失: 最大化判别器认为目标适合的概率
        # 同时添加熵正则化以鼓励探索
        goal_loss = -torch.log(suitability + 1e-8).mean() * self._goal_loss_scale
        entropy_loss = -goal_log_prob.mean() * 0.01  # 熵正则化

        total_loss = goal_loss + entropy_loss

        # 更新目标生成器
        self.goal_optimizer.zero_grad()
        total_loss.backward()
        self.goal_optimizer.step()

    def set_mode(self, mode: str) -> None:
        """设置模型模式"""
        for model in [self.policy, self.value, self.goal_generator, self.goal_discriminator]:
            if model is not None:
                getattr(model, mode)()
