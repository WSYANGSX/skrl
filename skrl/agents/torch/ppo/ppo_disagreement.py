from typing import Any, Mapping, Optional, Tuple, Union, List
import copy
import itertools
import gymnasium
from packaging import version

import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl import config, logger
from skrl.agents.torch import Agent
from skrl.memories.torch import Memory
from skrl.models.torch import Model
from skrl.resources.schedulers.torch import KLAdaptiveRL

# [start-config-dict-torch-disagreement]
PPO_DISAGREEMENT_DEFAULT_CONFIG = {
    "rollouts": 16,  # number of rollouts before updating
    "learning_epochs": 8,  # number of learning epochs during each update
    "mini_batches": 2,  # number of mini batches during each learning epoch
    "discount_factor": 0.99,  # discount factor (gamma)
    "lambda": 0.95,  # TD(lambda) coefficient (lam) for computing returns and advantages
    "learning_rate": 1e-3,  # learning rate
    "learning_rate_scheduler": None,  # learning rate scheduler class (see torch.optim.lr_scheduler)
    "learning_rate_scheduler_kwargs": {},  # learning rate scheduler's kwargs (e.g. {"step_size": 1e-3})
    "state_preprocessor": None,  # state preprocessor class (see skrl.resources.preprocessors)
    "state_preprocessor_kwargs": {},  # state preprocessor's kwargs (e.g. {"size": env.observation_space})
    "value_preprocessor": None,  # value preprocessor class (see skrl.resources.preprocessors)
    "value_preprocessor_kwargs": {},  # value preprocessor's kwargs (e.g. {"size": 1})
    "random_timesteps": 0,  # random exploration steps
    "learning_starts": 0,  # learning starts after this many steps
    "grad_norm_clip": 0.5,  # clipping coefficient for the norm of the gradients
    "ratio_clip": 0.2,  # clipping coefficient for computing the clipped surrogate objective
    "value_clip": 0.2,  # clipping coefficient for computing the value loss (if clip_predicted_values is True)
    "clip_predicted_values": False,  # clip predicted values during value loss computation
    "entropy_loss_scale": 0.0,  # entropy loss scaling factor
    "value_loss_scale": 1.0,  # value loss scaling factor
    "kl_threshold": 0,  # KL divergence threshold for early stopping
    "rewards_shaper": None,  # rewards shaping function: Callable(reward, timestep, timesteps) -> reward
    "time_limit_bootstrap": False,  # bootstrap at timeout termination (episode truncation)
    "mixed_precision": False,  # enable automatic mixed precision for higher performance
    # Disagreement algorithm specific parameters
    "use_disagreement": True,  # enable disagreement intrinsic reward
    "intrinsic_reward_scale": 0.1,  # scaling factor for intrinsic reward
    "extrinsic_reward_scale": 1.0,  # scaling factor for extrinsic reward
    "disagreement_ensemble_size": 5,  # number of dynamics models in ensemble
    "dynamics_learning_rate": 1e-3,  # learning rate for dynamics models
    "dynamics_update_freq": 1,  # update dynamics models every N policy updates
    "experiment": {
        "directory": "",  # experiment's parent directory
        "experiment_name": "",  # experiment name
        "write_interval": "auto",  # TensorBoard writing interval (timesteps)
        "checkpoint_interval": "auto",  # interval for checkpoints (timesteps)
        "store_separately": False,  # whether to store checkpoints separately
        "wandb": False,  # whether to use Weights & Biases
        "wandb_kwargs": {},  # wandb kwargs (see https://docs.wandb.ai/ref/python/init)
    },
}
# [end-config-dict-torch-disagreement]


class DisagreementDynamicsModel(nn.Module):
    """Forward dynamics model for Disagreement algorithm"""

    def __init__(self, observation_space: int, action_space: int, hidden_size: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(observation_space + action_space, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, observation_space),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


class PPODisagreement(Agent):
    def __init__(
        self,
        models: Mapping[str, Model],
        memory: Optional[Union[Memory, Tuple[Memory]]] = None,
        observation_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        action_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        device: Optional[Union[str, torch.device]] = None,
        cfg: Optional[dict] = None,
    ) -> None:
        """Proximal Policy Optimization with Disagreement intrinsic reward

        https://arxiv.org/abs/1906.04161

        :param models: Models used by the agent
        :type models: dictionary of skrl.models.torch.Model
        :param memory: Memory to storage the transitions.
                       If it is a tuple, the first element will be used for training and
                       for the rest only the environment transitions will be added
        :type memory: skrl.memory.torch.Memory, list of skrl.memory.torch.Memory or None
        :param observation_space: Observation/state space or shape (default: ``None``)
        :type observation_space: int, tuple or list of int, gymnasium.Space or None, optional
        :param action_space: Action space or shape (default: ``None``)
        :type action_space: int, tuple or list of int, gymnasium.Space or None, optional
        :param device: Device on which a tensor/array is or will be allocated (default: ``None``).
                       If None, the device will be either ``"cuda"`` if available or ``"cpu"``
        :type device: str or torch.device, optional
        :param cfg: Configuration dictionary
        :type cfg: dict

        :raises KeyError: If the models dictionary is missing a required key
        """
        _cfg = copy.deepcopy(PPO_DISAGREEMENT_DEFAULT_CONFIG)
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

        # checkpoint models
        self.checkpoint_modules["policy"] = self.policy
        self.checkpoint_modules["value"] = self.value

        # Disagreement specific components
        self._use_disagreement = self.cfg["use_disagreement"]
        self._intrinsic_reward_scale = self.cfg["intrinsic_reward_scale"]
        self._extrinsic_reward_scale = self.cfg["extrinsic_reward_scale"]
        self._ensemble_size = self.cfg["disagreement_ensemble_size"]
        self._dynamics_update_freq = self.cfg["dynamics_update_freq"]

        if self._use_disagreement:
            # Create ensemble of dynamics models
            self.dynamics_models = nn.ModuleList(
                [
                    DisagreementDynamicsModel(
                        observation_space=self.observation_space[0]
                        if isinstance(self.observation_space, tuple)
                        else self.observation_space,
                        action_space=self.action_space[0]
                        if isinstance(self.action_space, tuple)
                        else self.action_space,
                    ).to(self.device)
                    for _ in range(self._ensemble_size)
                ]
            )

            self.dynamics_optimizers = [
                torch.optim.Adam(model.parameters(), lr=self.cfg["dynamics_learning_rate"])
                for model in self.dynamics_models
            ]

            self.checkpoint_modules["dynamics_models"] = self.dynamics_models

        # broadcast models' parameters in distributed runs
        if config.torch.is_distributed:
            logger.info(f"Broadcasting models' parameters")
            if self.policy is not None:
                self.policy.broadcast_parameters()
                if self.value is not None and self.policy is not self.value:
                    self.value.broadcast_parameters()
            if self._use_disagreement:
                for model in self.dynamics_models:
                    model.broadcast_parameters()

        # configuration
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

        # set up optimizer and learning rate scheduler
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

        # disagreement specific tracking
        self._dynamics_update_count = 0
        self._current_intrinsic_reward = None

    def init(self, trainer_cfg: Optional[Mapping[str, Any]] = None) -> None:
        """Initialize the agent"""
        super().init(trainer_cfg=trainer_cfg)
        self.set_mode("eval")

        # create tensors in memory
        if self.memory is not None:
            self.memory.create_tensor(name="states", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="actions", size=self.action_space, dtype=torch.float32)
            self.memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="next_states", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="terminated", size=1, dtype=torch.bool)
            self.memory.create_tensor(name="truncated", size=1, dtype=torch.bool)
            self.memory.create_tensor(name="log_prob", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="values", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="returns", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="advantages", size=1, dtype=torch.float32)

            if self._use_disagreement:
                self.memory.create_tensor(name="intrinsic_rewards", size=1, dtype=torch.float32)
                self.memory.create_tensor(name="extrinsic_rewards", size=1, dtype=torch.float32)

            # tensors sampled during training
            self._tensors_names = ["states", "actions", "log_prob", "values", "returns", "advantages"]

        # create temporary variables needed for storage and computation
        self._current_log_prob = None
        self._current_next_states = None

    def _compute_intrinsic_reward(
        self, states: torch.Tensor, actions: torch.Tensor, next_states: torch.Tensor
    ) -> torch.Tensor:
        """Compute intrinsic reward based on disagreement between dynamics models"""
        if not self._use_disagreement:
            return torch.zeros(states.shape[0], 1, device=self.device)

        with torch.no_grad(), torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
            # Get predictions from all dynamics models
            predictions = []
            for model in self.dynamics_models:
                pred_next_state = model(states, actions)
                predictions.append(pred_next_state.unsqueeze(1))  # [batch_size, 1, state_dim]

            # Stack predictions: [batch_size, ensemble_size, state_dim]
            predictions = torch.cat(predictions, dim=1)

            # Compute variance across ensemble (disagreement)
            variance = torch.var(predictions, dim=1)  # [batch_size, state_dim]

            # Intrinsic reward is the mean variance across state dimensions
            intrinsic_reward = torch.mean(variance, dim=-1, keepdim=True)  # [batch_size, 1]

            return intrinsic_reward * self._intrinsic_reward_scale

    def _update_dynamics_models(self, states: torch.Tensor, actions: torch.Tensor, next_states: torch.Tensor) -> None:
        """Update the ensemble of dynamics models"""
        if not self._use_disagreement:
            return

        self._dynamics_update_count += 1

        # Only update dynamics models every N policy updates
        if self._dynamics_update_count % self._dynamics_update_freq != 0:
            return

        for model, optimizer in zip(self.dynamics_models, self.dynamics_optimizers):
            model.train()

            with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                pred_next_states = model(states, actions)
                dynamics_loss = F.mse_loss(pred_next_states, next_states)

            optimizer.zero_grad()
            self.scaler.scale(dynamics_loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()

            model.eval()

    def act(self, states: torch.Tensor, timestep: int, timesteps: int) -> torch.Tensor:
        """Process the environment's states to make a decision (actions) using the main policy"""
        # sample random actions
        if timestep < self._random_timesteps:
            return self.policy.random_act({"states": self._state_preprocessor(states)}, role="policy")

        # sample stochastic actions
        with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
            actions, log_prob, outputs = self.policy.act({"states": self._state_preprocessor(states)}, role="policy")
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

            # compute intrinsic reward if using disagreement
            intrinsic_reward = torch.zeros_like(rewards)
            extrinsic_reward = rewards.clone()

            if self._use_disagreement:
                intrinsic_reward = self._compute_intrinsic_reward(states, actions, next_states)
                self._current_intrinsic_reward = intrinsic_reward.mean().item()

                # Combine extrinsic and intrinsic rewards
                rewards = self._extrinsic_reward_scale * extrinsic_reward + intrinsic_reward

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
            memory_data = {
                "states": states,
                "actions": actions,
                "rewards": rewards,
                "next_states": next_states,
                "terminated": terminated,
                "truncated": truncated,
                "log_prob": self._current_log_prob,
                "values": values,
            }

            if self._use_disagreement:
                memory_data["intrinsic_rewards"] = intrinsic_reward
                memory_data["extrinsic_rewards"] = extrinsic_reward

            self.memory.add_samples(**memory_data)
            for memory in self.secondary_memories:
                memory.add_samples(**memory_data)

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

        # write tracking data and checkpoints
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
            advantage = 0
            advantages = torch.zeros_like(rewards)
            not_dones = dones.logical_not()
            memory_size = rewards.shape[0]

            # advantages computation
            for i in reversed(range(memory_size)):
                next_value = values[i + 1] if i < memory_size - 1 else last_values
                advantage = (
                    rewards[i]
                    - values[i]
                    + discount_factor * not_dones[i] * (next_value + lambda_coefficient * advantage)
                )
                advantages[i] = advantage
            # returns computation
            returns = advantages + values
            # normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            return returns, advantages

        # Update dynamics models if using disagreement
        if self._use_disagreement and self.memory is not None:
            states = self.memory.get_tensor_by_name("states")
            actions = self.memory.get_tensor_by_name("actions")
            next_states = self.memory.get_tensor_by_name("next_states")
            self._update_dynamics_models(states, actions, next_states)

        # compute returns and advantages
        with torch.no_grad(), torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
            self.value.train(False)
            last_values, _, _ = self.value.act(
                {"states": self._state_preprocessor(self._current_next_states.float())}, role="value"
            )
            self.value.train(True)
            last_values = self._value_preprocessor(last_values, inverse=True)

        values = self.memory.get_tensor_by_name("values")
        returns, advantages = compute_gae(
            rewards=self.memory.get_tensor_by_name("rewards"),
            dones=self.memory.get_tensor_by_name("terminated") | self.memory.get_tensor_by_name("truncated"),
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
            ) in sampled_batches:
                with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                    sampled_states = self._state_preprocessor(sampled_states, train=not epoch)

                    _, next_log_prob, _ = self.policy.act(
                        {"states": sampled_states, "taken_actions": sampled_actions}, role="policy"
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
                    surrogate_clipped = sampled_advantages * torch.clip(
                        ratio, 1.0 - self._ratio_clip, 1.0 + self._ratio_clip
                    )

                    policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

                    # compute value loss
                    predicted_values, _, _ = self.value.act({"states": sampled_states}, role="value")

                    if self._clip_predicted_values:
                        predicted_values = sampled_values + torch.clip(
                            predicted_values - sampled_values, min=-self._value_clip, max=self._value_clip
                        )
                    value_loss = self._value_loss_scale * F.mse_loss(sampled_returns, predicted_values)

                # optimization step
                self.optimizer.zero_grad()
                self.scaler.scale(policy_loss + entropy_loss + value_loss).backward()

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

        self.track_data("Policy / Standard deviation", self.policy.distribution(role="policy").stddev.mean().item())

        if self._learning_rate_scheduler:
            self.track_data("Learning / Learning rate", self.scheduler.get_last_lr()[0])

        # Track disagreement-specific metrics
        if self._use_disagreement:
            if hasattr(self, "_current_intrinsic_reward"):
                self.track_data("Reward / Intrinsic", self._current_intrinsic_reward)

            # Track extrinsic reward from memory if available
            if self.memory.has_tensor("extrinsic_rewards"):
                extrinsic_rewards = self.memory.get_tensor_by_name("extrinsic_rewards")
                self.track_data("Reward / Extrinsic", extrinsic_rewards.mean().item())

            # Track total reward
            if self.memory.has_tensor("rewards"):
                total_rewards = self.memory.get_tensor_by_name("rewards")
                self.track_data("Reward / Total", total_rewards.mean().item())
