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
PPO_FUN_DEFAULT_CONFIG = {
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
    "manager_loss_scale": 0.1,
    "kl_threshold": 0,
    "goal_embedding_size": 16,
    "manager_horizon": 8,
    "manager_learning_rate": 1e-4,
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


class PPO_FUN(Agent):
    def __init__(
        self,
        models: Mapping[str, Model],
        memory: Optional[Union[Memory, Tuple[Memory]]] = None,
        observation_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        action_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        device: Optional[Union[str, torch.device]] = None,
        cfg: Optional[dict] = None,
    ) -> None:
        _cfg = copy.deepcopy(PPO_FUN_DEFAULT_CONFIG)
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
        self.manager = self.models.get("manager", None)
        self.policy = self.models.get("policy", None)
        self.value = self.models.get("value", None)

        # checkpoint models
        self.checkpoint_modules["manager"] = self.manager
        self.checkpoint_modules["policy"] = self.policy
        self.checkpoint_modules["value"] = self.value

        # broadcast models' parameters in distributed runs
        if config.torch.is_distributed:
            logger.info(f"Broadcasting models' parameters")
            if self.manager is not None:
                self.manager.broadcast_parameters()
            if self.policy is not None:
                self.policy.broadcast_parameters()
            if self.value is not None:
                self.value.broadcast_parameters()

        # configuration
        self._goal_embedding_size = self.cfg["goal_embedding_size"]
        self._manager_horizon = self.cfg["manager_horizon"]
        self._manager_learning_rate = self.cfg["manager_learning_rate"]
        self._manager_loss_scale = self.cfg["manager_loss_scale"]

        self._learning_epochs = self.cfg["learning_epochs"]
        self._mini_batches = self.cfg["mini_batches"]
        self._rollouts = self.cfg["rollouts"]
        self._rollout = 0

        self._grad_norm_clip = self.cfg["grad_norm_clip"]
        self._ratio_clip = self.cfg["ratio_clip"]
        self._value_clip = self.cfg["value_clip"]
        self._clip_predicted_values = self.cfg["clip_predicted_values"]

        self._value_loss_scale = self.cfg["value_loss_scale"]
        self._entropy_loss_scale = self.cfg["entropy_loss_scale"]

        self._kl_threshold = self.cfg["kl_threshold"]

        self._learning_rate = self.cfg["learning_rate"]
        self._learning_rate_scheduler = self.cfg["learning_rate_scheduler"]

        self._state_preprocessor = self.cfg["state_preprocessor"]
        self._value_preprocessor = self.cfg["value_preprocessor"]

        self._discount_factor = self.cfg["discount_factor"]
        self._lambda = self.cfg["lambda"]

        self._random_timesteps = self.cfg["random_timesteps"]
        self._learning_starts = self.cfg["learning_starts"]

        self._rewards_shaper = self.cfg["rewards_shaper"]
        self._time_limit_bootstrap = self.cfg["time_limit_bootstrap"]

        self._mixed_precision = self.cfg["mixed_precision"]

        # set up automatic mixed precision
        self._device_type = torch.device(device).type
        if version.parse(torch.__version__) >= version.parse("2.4"):
            self.scaler = torch.amp.GradScaler(device=self._device_type, enabled=self._mixed_precision)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self._mixed_precision)

        # set up optimizers and learning rate schedulers
        if self.manager is not None:
            self.manager_optimizer = torch.optim.Adam(self.manager.parameters(), lr=self._manager_learning_rate)
            self.checkpoint_modules["manager_optimizer"] = self.manager_optimizer

        if self.policy is not None and self.value is not None:
            if self.policy is self.value:
                self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self._learning_rate)
            else:
                self.optimizer = torch.optim.Adam(
                    itertools.chain(self.policy.parameters(), self.value.parameters()), lr=self._learning_rate
                )

            if self._learning_rate_scheduler is not None:
                self.scheduler = self._learning_rate_scheduler(
                    self.optimizer, **self.cfg["learning_rate_scheduler_kwargs"]
                )

            self.checkpoint_modules["optimizer"] = self.optimizer

        # set up preprocessors
        if self._state_preprocessor:
            self._state_preprocessor = self._state_preprocessor(**self.cfg["state_preprocessor_kwargs"])
            self.checkpoint_modules["state_preprocessor"] = self._state_preprocessor
        else:
            self._state_preprocessor = self._empty_preprocessor

        if self._value_preprocessor:
            self._value_preprocessor = self._value_preprocessor(**self.cfg["value_preprocessor_kwargs"])
            self.checkpoint_modules["value_preprocessor"] = self._value_preprocessor
        else:
            self._value_preprocessor = self._empty_preprocessor

        self._current_log_prob = None
        self._current_values = None
        self._current_next_states = None
        self._current_goal_embedding = None
        self._manager_step_inited = False
        self._manager_states = []
        self._manager_goals = []

    def init(self, trainer_cfg: Optional[Mapping[str, Any]] = None) -> None:
        """Initialize the agent"""
        super().init(trainer_cfg=trainer_cfg)
        self.set_mode("eval")

        # create tensors in memory
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
            self.memory.create_tensor(name="goals", size=self._goal_embedding_size, dtype=torch.float32)

            # tensors sampled during training
            self._tensors_names = ["states", "actions", "log_prob", "values", "returns", "advantages", "goals"]

        # reset internal state
        self._current_log_prob = None
        self._current_values = None
        self._current_next_states = None
        self._current_goal_embedding = None
        self._manager_step_inited = False
        self._manager_states = []
        self._manager_goals = []

    def act(self, states: torch.Tensor, timestep: int, timesteps: int) -> torch.Tensor:
        """Process the environment's states to make a decision (actions) using the feudal hierarchy"""
        # sample random actions
        if timestep < self._random_timesteps:
            return self.policy.random_act({"states": self._state_preprocessor(states)}, role="policy")

        # manager generates new goal every manager_horizon steps
        if not self._manager_step_inited:
            goal_embedding, _, _ = self.manager.act({"states": self._state_preprocessor(states)})
            self._current_goal_embedding = torch.zeros_like(goal_embedding)
            self._manager_step_count = torch.zeros((len(goal_embedding),), device=self.device, dtype=torch.int32)
            self._manager_step_inited = True

        needs_new_goal = self._manager_step_count % self._manager_horizon == 0
        if torch.any(needs_new_goal):
            goal_embedding, _, _ = self.manager.act({"states": self._state_preprocessor(states)})
            indices = needs_new_goal
            self._current_goal_embedding[indices] = goal_embedding[indices]
            self._manager_states.append(states[indices].clone())
            self._manager_goals.append(goal_embedding[indices].clone())

        self._manager_step_count += 1

        # policy generates action based on state and current goal
        with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
            actions, log_prob, outputs = self.policy.act(
                {"states": self._state_preprocessor(states), "goals": self._current_goal_embedding}, role="policy"
            )
            self._current_log_prob = log_prob

        return actions, log_prob, outputs

    def record_transition(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        infos: Any,
        timestep: int,
        timesteps: int,
    ) -> None:
        """Record an environment transition in memory"""
        super().record_transition(
            states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps
        )

        if self.memory is not None:
            self._current_next_states = next_states

            if self._manager_step_inited:
                done = (terminated | truncated).squeeze()
                if torch.any(done):
                    self._manager_step_count[done] = 0

            # reward shaping
            if self._rewards_shaper is not None:
                rewards = self._rewards_shaper(rewards, timestep, timesteps)

            # compute values
            with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                values, _, _ = self.value.act({"states": self._state_preprocessor(states)}, role="value")
                values = self._value_preprocessor(values, inverse=True)

            # time-limit (truncation) bootstrapping
            if self._time_limit_bootstrap:
                rewards += self._discount_factor * values * truncated

            # storage transition in memory
            self.memory.add_samples(
                states=states,
                actions=actions,
                rewards=rewards,
                next_states=next_states,
                terminated=terminated,
                truncated=truncated,
                log_prob=self._current_log_prob,
                values=values,
                goals=self._current_goal_embedding,
            )
            for memory in self.secondary_memories:
                memory.add_samples(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    next_states=next_states,
                    terminated=terminated,
                    truncated=truncated,
                    log_prob=self._current_log_prob,
                    values=values,
                    goals=self._current_goal_embedding,
                )

    def pre_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called before the interaction with the environment"""
        pass

    def post_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called after the interaction with the environment"""
        self._rollout += 1
        if not self._rollout % self._rollouts and timestep >= self._learning_starts:
            self.set_mode("train")
            self._update(timestep, timesteps)
            self.set_mode("eval")

        super().post_interaction(timestep, timesteps)

    def _update(self, timestep: int, timesteps: int) -> None:
        """Algorithm's main update step"""

        def compute_gae(
            rewards: torch.Tensor,
            dones: torch.Tensor,
            values: torch.Tensor,
            next_values: torch.Tensor,
            discount_factor: float = 0.99,
            lambda_coefficient: float = 0.95,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """Compute the Generalized Advantage Estimator (GAE)"""
            advantage = 0
            advantages = torch.zeros_like(rewards)
            not_dones = dones.logical_not()
            memory_size = rewards.shape[0]

            # advantages computation
            for i in reversed(range(memory_size)):
                next_value = values[i + 1] if i < memory_size - 1 else next_values
                advantage = (
                    rewards[i]
                    + discount_factor * not_dones[i] * next_value
                    - values[i]
                    + discount_factor * lambda_coefficient * not_dones[i] * advantage
                )
                advantages[i] = advantage

            # returns computation
            returns = advantages + values
            # normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            return returns, advantages

        # compute returns and advantages
        with torch.no_grad(), torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
            self.value.train(False)
            last_values, _, _ = self.value.act(
                {"states": self._state_preprocessor(self._current_next_states.float())}, role="value"
            )
            self.value.train(True)
            last_values = self._value_preprocessor(last_values, inverse=True)

        values = self.memory.get_tensor_by_name("values")
        rewards = self.memory.get_tensor_by_name("rewards")
        terminated = self.memory.get_tensor_by_name("terminated")
        truncated = self.memory.get_tensor_by_name("truncated")

        returns, advantages = compute_gae(
            rewards=rewards,
            dones=terminated | truncated,
            values=values,
            next_values=last_values,
            discount_factor=self._discount_factor,
            lambda_coefficient=self._lambda,
        )

        self.memory.set_tensor_by_name("values", self._value_preprocessor(values, train=True))
        self.memory.set_tensor_by_name("returns", self._value_preprocessor(returns, train=True))
        self.memory.set_tensor_by_name("advantages", advantages)

        # sample mini-batches from memory
        sampled_batches = self.memory.sample_all(names=self._tensors_names, mini_batches=self._mini_batches)

        cumulative_policy_loss = 0
        cumulative_entropy_loss = 0
        cumulative_value_loss = 0
        cumulative_manager_loss = 0

        # Update manager network
        if len(self._manager_states) > 0:
            manager_loss = self._update_manager()
            cumulative_manager_loss += manager_loss.item()

        # learning epochs
        for epoch in range(self._learning_epochs):
            kl_divergences = []

            # mini-batches loop
            for (
                sampled_states,
                sampled_actions,
                sampled_log_prob,
                sampled_values,
                sampled_returns,
                sampled_advantages,
                sampled_goals,
            ) in sampled_batches:
                # Update policy and value networks
                with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                    sampled_states = self._state_preprocessor(sampled_states, train=not epoch)

                    # Policy forward pass - get new action distribution
                    _, next_log_prob, _ = self.policy.act(
                        {
                            "states": sampled_states,
                            "goals": sampled_goals,
                            "taken_actions": sampled_actions,
                        },
                        role="policy",
                    )

                    # compute approximate KL divergence
                    with torch.no_grad():
                        ratio = next_log_prob - sampled_log_prob
                        kl_divergence = ((torch.exp(ratio) - 1) - ratio).mean()
                        kl_divergences.append(kl_divergence)

                    # early stopping with KL divergence
                    if self._kl_threshold and kl_divergence > self._kl_threshold:
                        break

                    # compute entropy loss
                    if self._entropy_loss_scale:
                        entropy_loss = -self._entropy_loss_scale * self.policy.get_entropy(role="policy").mean()
                    else:
                        entropy_loss = 0

                    # compute policy loss
                    ratio = torch.exp(next_log_prob - sampled_log_prob)
                    surrogate = sampled_advantages * ratio
                    surrogate_clipped = sampled_advantages * torch.clamp(
                        ratio, 1.0 - self._ratio_clip, 1.0 + self._ratio_clip
                    )

                    policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

                    # compute value loss
                    predicted_values, _, _ = self.value.act({"states": sampled_states}, role="value")

                    if self._clip_predicted_values:
                        predicted_values = sampled_values + torch.clamp(
                            predicted_values - sampled_values, min=-self._value_clip, max=self._value_clip
                        )
                    value_loss = self._value_loss_scale * F.mse_loss(sampled_returns, predicted_values)

                # optimization step for policy
                self.optimizer.zero_grad()
                self.scaler.scale(policy_loss + entropy_loss).backward()

                if config.torch.is_distributed:
                    self.policy.reduce_parameters()
                    if self.policy is not self.value:
                        self.value.reduce_parameters()

                if self._grad_norm_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    if self.policy is self.value:
                        nn.utils.clip_grad_norm_(self.policy.parameters(), self._grad_norm_clip)
                    else:
                        nn.utils.clip_grad_norm_(
                            itertools.chain(self.policy.parameters(), self.value.parameters()), self._grad_norm_clip
                        )

                self.scaler.step(self.optimizer)
                self.scaler.update()

                # update cumulative losses
                cumulative_policy_loss += policy_loss.item()
                cumulative_value_loss += value_loss.item()
                if self._entropy_loss_scale:
                    cumulative_entropy_loss += entropy_loss.item()

            # update learning rate
            if self._learning_rate_scheduler:
                if isinstance(self.scheduler, KLAdaptiveLR):
                    kl = torch.tensor(kl_divergences, device=self.device).mean()
                    # reduce (collect from all workers/processes) KL in distributed runs
                    if config.torch.is_distributed:
                        torch.distributed.all_reduce(kl, op=torch.distributed.ReduceOp.SUM)
                        kl /= config.torch.world_size
                    self.scheduler.step(kl.item())
                else:
                    self.scheduler.step()

        # record data
        self.track_data("Loss / Policy loss", cumulative_policy_loss / (self._learning_epochs * self._mini_batches))
        self.track_data("Loss / Value loss", cumulative_value_loss / (self._learning_epochs * self._mini_batches))
        if self._entropy_loss_scale:
            self.track_data(
                "Loss / Entropy loss", cumulative_entropy_loss / (self._learning_epochs * self._mini_batches)
            )
        if len(self._manager_states) > 0:
            self.track_data(
                "Loss / Manager loss", cumulative_manager_loss / (self._learning_epochs * self._mini_batches)
            )

        self.track_data("Policy / Standard deviation", self.policy.distribution(role="policy").stddev.mean().item())

        if self._learning_rate_scheduler:
            self.track_data("Learning / Learning rate", self.scheduler.get_last_lr()[0])

        # reset manager buffers
        self._manager_states.clear()
        self._manager_goals.clear()

    def _update_manager(self) -> torch.Tensor:
        """Update the manager policy using intrinsic reward based on policy's progress"""
        if len(self._manager_states) == 0:
            return torch.tensor(0.0, device=self.device)

        # Convert buffers to tensors
        manager_states = torch.stack(self._manager_states)
        manager_goals = torch.stack(self._manager_goals)

        with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
            # Get state features using manager's encoder
            if hasattr(self.manager, "encode"):
                state_features = self.manager.encode(self._state_preprocessor(manager_states))
            else:
                # Fallback: use the manager's internal representation
                state_features, _, _ = self.manager.act({"states": self._state_preprocessor(manager_states)})

            # Compute intrinsic reward: cosine similarity between state features and goals
            intrinsic_reward = F.cosine_similarity(state_features, manager_goals, dim=-1)

            # Manager loss: maximize intrinsic reward (progress towards goals)
            manager_loss = -intrinsic_reward.mean() * self._manager_loss_scale

        # Update manager
        self.manager_optimizer.zero_grad()
        self.scaler.scale(manager_loss).backward()

        if config.torch.is_distributed:
            self.manager.reduce_parameters()

        if self._grad_norm_clip > 0:
            self.scaler.unscale_(self.manager_optimizer)
            nn.utils.clip_grad_norm_(self.manager.parameters(), self._grad_norm_clip)

        self.scaler.step(self.manager_optimizer)

        return manager_loss

    def set_mode(self, mode: str) -> None:
        """Set the model mode (train or eval)"""
        for model in [self.manager, self.policy, self.value]:
            if model is not None:
                getattr(model, mode)()

        if mode == "eval":
            self._manager_states.clear()
            self._manager_goals.clear()
