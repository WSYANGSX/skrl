from typing import Any, Mapping, Optional, Tuple, Union, Dict

import copy
import gymnasium
from packaging import version

import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl import config, logger
from skrl.agents.torch import Agent
from skrl.memories.torch import Memory
from skrl.models.torch import Model
from skrl.resources.schedulers.torch import KLAdaptiveLR

# fmt: off
# [start-config-dict-torch]
PPO_ICM_DEFAULT_CONFIG = {
    "rollouts": 16,                 # number of rollouts before updating
    "learning_epochs": 8,           # number of learning epochs during each update
    "mini_batches": 2,              # number of mini batches during each learning epoch

    "discount_factor": 0.99,        # discount factor (gamma)
    "lambda": 0.95,                 # TD(lambda) coefficient (lam) for computing returns and advantages

    "learning_rate": 1e-3,                  # learning rate
    "learning_rate_scheduler": None,        # learning rate scheduler class (see torch.optim.lr_scheduler)
    "learning_rate_scheduler_kwargs": {},   # learning rate scheduler's kwargs (e.g. {"step_size": 1e-3})

    "state_preprocessor": None,             # state preprocessor class (see skrl.resources.preprocessors)
    "state_preprocessor_kwargs": {},        # state preprocessor's kwargs (e.g. {"size": env.observation_space})
    "value_preprocessor": None,             # value preprocessor class (see skrl.resources.preprocessors)
    "value_preprocessor_kwargs": {},        # value preprocessor's kwargs (e.g. {"size": 1})

    "random_timesteps": 0,          # random exploration steps
    "learning_starts": 0,           # learning starts after this many steps

    "grad_norm_clip": 0.5,              # clipping coefficient for the norm of the gradients
    "ratio_clip": 0.2,                  # clipping coefficient for computing the clipped surrogate objective
    "value_clip": 0.2,                  # clipping coefficient for computing the value loss (if clip_predicted_values is True)
    "clip_predicted_values": False,     # clip predicted values during value loss computation

    "entropy_loss_scale": 0.0,      # entropy loss scaling factor
    "value_loss_scale": 1.0,        # value loss scaling factor

    "kl_threshold": 0,              # KL divergence threshold for early stopping

    "rewards_shaper": None,         # rewards shaping function: Callable(reward, timestep, timesteps) -> reward
    "time_limit_bootstrap": False,  # bootstrap at timeout termination (episode truncation)

    "mixed_precision": False,       # enable automatic mixed precision for higher performance

    # ICM specific parameters
    "icm_enabled": False,           # whether to enable ICM
    "icm_beta": 0.2,                # weight for intrinsic reward (β)
    "icm_eta": 0.01,                # scaling factor for intrinsic reward (η)
    "icm_forward_loss_scale": 0.2,  # scaling factor for forward model loss
    "icm_inverse_loss_scale": 0.8,  # scaling factor for inverse model loss
    "icm_feature_dim": 256,         # dimension of feature vector

    "experiment": {
        "directory": "",            # experiment's parent directory
        "experiment_name": "",      # experiment name
        "write_interval": "auto",   # TensorBoard writing interval (timesteps)

        "checkpoint_interval": "auto",      # interval for checkpoints (timesteps)
        "store_separately": False,          # whether to store checkpoints separately

        "wandb": False,             # whether to use Weights & Biases
        "wandb_kwargs": {}          # wandb kwargs (see https://docs.wandb.ai/ref/python/init)
    }
}
# [end-config-dict-torch]
# fmt: on


class ICM(nn.Module):
    """Intrinsic Curiosity Module (ICM) for generating intrinsic rewards"""

    def __init__(self,
                 observation_space: gymnasium.Space,
                 action_space: gymnasium.Space,
                 feature_dim: int = 256,
                 device: Union[str, torch.device] = "cpu"):
        super(ICM, self).__init__()
        self.device = device
        self.feature_dim = feature_dim

        # Feature encoding network (φ)
        self.feature_encoder = nn.Sequential(
            nn.Linear(observation_space.shape[0], 256), nn.ReLU(),
            nn.Linear(256, feature_dim), nn.ReLU()).to(device)

        # Inverse model: predicts action from current and next state features
        self.inverse_model = nn.Sequential(
            nn.Linear(feature_dim * 2, 256), nn.ReLU(),
            nn.Linear(256, action_space.shape[0])).to(device)

        # Forward model: predicts next state feature from current state feature and action
        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + action_space.shape[0], 256), nn.ReLU(),
            nn.Linear(256, feature_dim)).to(device)

    def forward(self, states: torch.Tensor, next_states: torch.Tensor,
                actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute ICM components and intrinsic reward"""
        # Encode states to features
        phi_s = self.feature_encoder(states)
        phi_s_next = self.feature_encoder(next_states)

        # Inverse model: predict action from state features
        predicted_actions = self.inverse_model(
            torch.cat([phi_s, phi_s_next], dim=1))

        # Forward model: predict next state feature
        predicted_phi_s_next = self.forward_model(
            torch.cat([phi_s, actions], dim=1))

        # Intrinsic reward: prediction error of forward model
        intrinsic_reward = 0.5 * F.mse_loss(
            predicted_phi_s_next, phi_s_next.detach(), reduction='none').mean(
                dim=1, keepdim=True)

        return {
            "predicted_actions": predicted_actions,
            "predicted_phi_s_next": predicted_phi_s_next,
            "intrinsic_reward": intrinsic_reward,
            "phi_s": phi_s,
            "phi_s_next": phi_s_next
        }

    def compute_loss(self,
                     states: torch.Tensor,
                     next_states: torch.Tensor,
                     actions: torch.Tensor,
                     forward_scale: float = 0.2,
                     inverse_scale: float = 0.8) -> Dict[str, torch.Tensor]:
        """Compute ICM losses"""
        outputs = self.forward(states, next_states, actions)

        # Inverse loss: action prediction
        inverse_loss = F.mse_loss(outputs["predicted_actions"], actions)

        # Forward loss: next state prediction
        forward_loss = F.mse_loss(outputs["predicted_phi_s_next"],
                                  outputs["phi_s_next"].detach())

        total_loss = inverse_scale * inverse_loss + forward_scale * forward_loss

        return {
            "total_loss": total_loss,
            "inverse_loss": inverse_loss,
            "forward_loss": forward_loss,
            "intrinsic_reward": outputs["intrinsic_reward"]
        }


class PPO_ICM(Agent):

    def __init__(
        self,
        models: Mapping[str, Model],
        memory: Optional[Union[Memory, Tuple[Memory]]] = None,
        observation_space: Optional[Union[int, Tuple[int],
                                          gymnasium.Space]] = None,
        action_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        device: Optional[Union[str, torch.device]] = None,
        cfg: Optional[dict] = None,
    ) -> None:
        _cfg = copy.deepcopy(PPO_ICM_DEFAULT_CONFIG)
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

        # ICM module
        self.icm_enabled = self.cfg["icm_enabled"]
        if self.icm_enabled:
            self.icm = ICM(observation_space, action_space,
                           self.cfg["icm_feature_dim"], self.device)
            self.checkpoint_modules["icm"] = self.icm

        # checkpoint models
        self.checkpoint_modules["policy"] = self.policy
        self.checkpoint_modules["value"] = self.value

        # broadcast models' parameters in distributed runs
        if config.torch.is_distributed:
            logger.info(f"Broadcasting models' parameters")
            if self.policy is not None:
                self.policy.broadcast_parameters()
                if self.value is not None and self.policy is not self.value:
                    self.value.broadcast_parameters()

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

        # ICM parameters
        self._icm_beta = self.cfg["icm_beta"]
        self._icm_eta = self.cfg["icm_eta"]
        self._icm_forward_loss_scale = self.cfg["icm_forward_loss_scale"]
        self._icm_inverse_loss_scale = self.cfg["icm_inverse_loss_scale"]

        # set up automatic mixed precision
        self._device_type = torch.device(device).type
        if version.parse(torch.__version__) >= version.parse("2.4"):
            self.scaler = torch.amp.GradScaler(device=self._device_type,
                                               enabled=self._mixed_precision)
        else:
            self.scaler = torch.cuda.amp.GradScaler(
                enabled=self._mixed_precision)

        # set up optimizer and learning rate scheduler
        if self.policy is not None and self.value is not None:
            params = []
            if self.policy is self.value:
                params.extend(self.policy.parameters())
            else:
                params.extend(self.policy.parameters())
                params.extend(self.value.parameters())

            # Add ICM parameters if enabled
            if self.icm_enabled:
                params.extend(self.icm.parameters())

            self.optimizer = torch.optim.Adam(params, lr=self._learning_rate)

            if self._learning_rate_scheduler is not None:
                self.scheduler = self._learning_rate_scheduler(
                    self.optimizer,
                    **self.cfg["learning_rate_scheduler_kwargs"])

            self.checkpoint_modules["optimizer"] = self.optimizer

        # set up preprocessors
        if self._state_preprocessor:
            self._state_preprocessor = self._state_preprocessor(
                **self.cfg["state_preprocessor_kwargs"])
            self.checkpoint_modules[
                "state_preprocessor"] = self._state_preprocessor
        else:
            self._state_preprocessor = self._empty_preprocessor

        if self._value_preprocessor:
            self._value_preprocessor = self._value_preprocessor(
                **self.cfg["value_preprocessor_kwargs"])
            self.checkpoint_modules[
                "value_preprocessor"] = self._value_preprocessor
        else:
            self._value_preprocessor = self._empty_preprocessor

    def init(self, trainer_cfg: Optional[Mapping[str, Any]] = None) -> None:
        """Initialize the agent"""
        super().init(trainer_cfg=trainer_cfg)
        self.set_mode("eval")

        # create tensors in memory
        if self.memory is not None:
            self.memory.create_tensor(name="states",
                                      size=self.observation_space,
                                      dtype=torch.float32)
            self.memory.create_tensor(name="next_states",
                                      size=self.observation_space,
                                      dtype=torch.float32)
            self.memory.create_tensor(name="actions",
                                      size=self.action_space,
                                      dtype=torch.float32)
            self.memory.create_tensor(name="rewards",
                                      size=1,
                                      dtype=torch.float32)
            self.memory.create_tensor(name="intrinsic_rewards",
                                      size=1,
                                      dtype=torch.float32)
            self.memory.create_tensor(name="terminated",
                                      size=1,
                                      dtype=torch.bool)
            self.memory.create_tensor(name="truncated",
                                      size=1,
                                      dtype=torch.bool)
            self.memory.create_tensor(name="log_prob",
                                      size=1,
                                      dtype=torch.float32)
            self.memory.create_tensor(name="values",
                                      size=1,
                                      dtype=torch.float32)
            self.memory.create_tensor(name="returns",
                                      size=1,
                                      dtype=torch.float32)
            self.memory.create_tensor(name="advantages",
                                      size=1,
                                      dtype=torch.float32)

            # tensors sampled during training
            self._tensors_names = [
                "states", "actions", "log_prob", "values", "returns",
                "advantages"
            ]
            if self.icm_enabled:
                self._tensors_names.extend(
                    ["next_states", "intrinsic_rewards"])

        # create temporary variables needed for storage and computation
        self._current_log_prob = None
        self._current_next_states = None

    def act(self, states: torch.Tensor, timestep: int,
            timesteps: int) -> torch.Tensor:
        """Process the environment's states to make a decision (actions) using the main policy"""
        # sample random actions
        if timestep < self._random_timesteps:
            return self.policy.random_act(
                {"states": self._state_preprocessor(states)}, role="policy")

        # sample stochastic actions
        with torch.autocast(device_type=self._device_type,
                            enabled=self._mixed_precision):
            actions, log_prob, outputs = self.policy.act(
                {"states": self._state_preprocessor(states)}, role="policy")
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
        super().record_transition(states, actions, rewards, next_states,
                                  terminated, truncated, infos, timestep,
                                  timesteps)

        if self.memory is not None:
            self._current_next_states = next_states

            # reward shaping
            if self._rewards_shaper is not None:
                rewards = self._rewards_shaper(rewards, timestep, timesteps)

            # compute values
            with torch.autocast(device_type=self._device_type,
                                enabled=self._mixed_precision):
                values, _, _ = self.value.act(
                    {"states": self._state_preprocessor(states)}, role="value")
                values = self._value_preprocessor(values, inverse=True)

            # compute intrinsic reward if ICM is enabled
            intrinsic_rewards = torch.zeros_like(rewards)
            if self.icm_enabled:
                with torch.no_grad():
                    icm_outputs = self.icm(
                        self._state_preprocessor(states),
                        self._state_preprocessor(next_states), actions)
                    intrinsic_rewards = self._icm_eta * icm_outputs[
                        "intrinsic_reward"]

            # time-limit (truncation) bootstrapping
            if self._time_limit_bootstrap:
                rewards += self._discount_factor * values * truncated

            # combine extrinsic and intrinsic rewards
            total_rewards = rewards + self._icm_beta * intrinsic_rewards

            # storage transition in memory
            self.memory.add_samples(
                states=states,
                next_states=next_states,
                actions=actions,
                rewards=total_rewards,
                intrinsic_rewards=intrinsic_rewards,
                terminated=terminated,
                truncated=truncated,
                log_prob=self._current_log_prob,
                values=values,
            )
            for memory in self.secondary_memories:
                memory.add_samples(
                    states=states,
                    next_states=next_states,
                    actions=actions,
                    rewards=total_rewards,
                    intrinsic_rewards=intrinsic_rewards,
                    terminated=terminated,
                    truncated=truncated,
                    log_prob=self._current_log_prob,
                    values=values,
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
        ) -> torch.Tensor:
            """Compute the Generalized Advantage Estimator (GAE)"""
            advantage = 0
            advantages = torch.zeros_like(rewards)
            not_dones = dones.logical_not()
            memory_size = rewards.shape[0]

            # advantages computation
            for i in reversed(range(memory_size)):
                next_values = values[i +
                                     1] if i < memory_size - 1 else last_values
                advantage = (rewards[i] - values[i] +
                             discount_factor * not_dones[i] *
                             (next_values + lambda_coefficient * advantage))
                advantages[i] = advantage
            # returns computation
            returns = advantages + values
            # normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() +
                                                             1e-8)

            return returns, advantages

        # compute returns and advantages
        with torch.no_grad(), torch.autocast(device_type=self._device_type,
                                             enabled=self._mixed_precision):
            self.value.train(False)
            last_values, _, _ = self.value.act(
                {
                    "states":
                    self._state_preprocessor(self._current_next_states.float())
                },
                role="value")
            self.value.train(True)
            last_values = self._value_preprocessor(last_values, inverse=True)

        values = self.memory.get_tensor_by_name("values")
        rewards = self.memory.get_tensor_by_name("rewards")

        returns, advantages = compute_gae(
            rewards=rewards,
            dones=self.memory.get_tensor_by_name("terminated")
            | self.memory.get_tensor_by_name("truncated"),
            values=values,
            next_values=last_values,
            discount_factor=self._discount_factor,
            lambda_coefficient=self._lambda,
        )

        self.memory.set_tensor_by_name(
            "values", self._value_preprocessor(values, train=True))
        self.memory.set_tensor_by_name(
            "returns", self._value_preprocessor(returns, train=True))
        self.memory.set_tensor_by_name("advantages", advantages)

        # sample mini-batches from memory
        sampled_batches = self.memory.sample_all(
            names=self._tensors_names, mini_batches=self._mini_batches)

        cumulative_policy_loss = 0
        cumulative_entropy_loss = 0
        cumulative_value_loss = 0
        cumulative_icm_loss = 0

        # learning epochs
        for epoch in range(self._learning_epochs):
            kl_divergences = []

            # mini-batches loop
            for batch in sampled_batches:
                if self.icm_enabled:
                    (sampled_states, sampled_actions, sampled_log_prob,
                     sampled_values, sampled_returns, sampled_advantages,
                     sampled_next_states, sampled_intrinsic_rewards) = batch
                else:
                    (sampled_states, sampled_actions, sampled_log_prob,
                     sampled_values, sampled_returns,
                     sampled_advantages) = batch

                with torch.autocast(device_type=self._device_type,
                                    enabled=self._mixed_precision):

                    sampled_states = self._state_preprocessor(sampled_states,
                                                              train=not epoch)

                    _, next_log_prob, _ = self.policy.act(
                        {
                            "states": sampled_states,
                            "taken_actions": sampled_actions
                        },
                        role="policy")

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
                        entropy_loss = -self._entropy_loss_scale * self.policy.get_entropy(
                            role="policy").mean()
                    else:
                        entropy_loss = 0

                    # compute policy loss
                    ratio = torch.exp(next_log_prob - sampled_log_prob)
                    surrogate = sampled_advantages * ratio
                    surrogate_clipped = sampled_advantages * torch.clip(
                        ratio, 1.0 - self._ratio_clip, 1.0 + self._ratio_clip)

                    policy_loss = -torch.min(surrogate,
                                             surrogate_clipped).mean()

                    # compute value loss
                    predicted_values, _, _ = self.value.act(
                        {"states": sampled_states}, role="value")

                    if self._clip_predicted_values:
                        predicted_values = sampled_values + torch.clip(
                            predicted_values - sampled_values,
                            min=-self._value_clip,
                            max=self._value_clip)
                    value_loss = self._value_loss_scale * F.mse_loss(
                        sampled_returns, predicted_values)

                    # compute ICM loss if enabled
                    icm_loss = 0
                    if self.icm_enabled:
                        sampled_next_states_preprocessed = self._state_preprocessor(
                            sampled_next_states, train=not epoch)
                        icm_losses = self.icm.compute_loss(
                            sampled_states, sampled_next_states_preprocessed,
                            sampled_actions, self._icm_forward_loss_scale,
                            self._icm_inverse_loss_scale)
                        icm_loss = icm_losses["total_loss"]

                # optimization step
                self.optimizer.zero_grad()
                total_loss = policy_loss + entropy_loss + value_loss
                if self.icm_enabled:
                    total_loss += icm_loss

                self.scaler.scale(total_loss).backward()

                if config.torch.is_distributed:
                    self.policy.reduce_parameters()
                    if self.policy is not self.value:
                        self.value.reduce_parameters()
                    if self.icm_enabled:
                        # Implement reduce_parameters for ICM if needed
                        pass

                if self._grad_norm_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    params = []
                    if self.policy is self.value:
                        params.extend(self.policy.parameters())
                    else:
                        params.extend(self.policy.parameters())
                        params.extend(self.value.parameters())
                    if self.icm_enabled:
                        params.extend(self.icm.parameters())

                    nn.utils.clip_grad_norm_(params, self._grad_norm_clip)

                self.scaler.step(self.optimizer)
                self.scaler.update()

                # update cumulative losses
                cumulative_policy_loss += policy_loss.item()
                cumulative_value_loss += value_loss.item()
                if self._entropy_loss_scale:
                    cumulative_entropy_loss += entropy_loss.item()
                if self.icm_enabled:
                    cumulative_icm_loss += icm_loss.item()

            # update learning rate
            if self._learning_rate_scheduler:
                if isinstance(self.scheduler, KLAdaptiveLR):
                    kl = torch.tensor(kl_divergences,
                                      device=self.device).mean()
                    # reduce (collect from all workers/processes) KL in distributed runs
                    if config.torch.is_distributed:
                        torch.distributed.all_reduce(
                            kl, op=torch.distributed.ReduceOp.SUM)
                        kl /= config.torch.world_size
                    self.scheduler.step(kl.item())
                else:
                    self.scheduler.step()

        # record data
        self.track_data(
            "Loss / Policy loss", cumulative_policy_loss /
            (self._learning_epochs * self._mini_batches))
        self.track_data(
            "Loss / Value loss", cumulative_value_loss /
            (self._learning_epochs * self._mini_batches))
        if self._entropy_loss_scale:
            self.track_data(
                "Loss / Entropy loss", cumulative_entropy_loss /
                (self._learning_epochs * self._mini_batches))
        if self.icm_enabled:
            self.track_data(
                "Loss / ICM loss", cumulative_icm_loss /
                (self._learning_epochs * self._mini_batches))
            self.track_data("Reward / Intrinsic",
                            sampled_intrinsic_rewards.mean().item())
            self.track_data("Reward / Extrinsic",
                            (rewards -
                             sampled_intrinsic_rewards).mean().item())

        self.track_data(
            "Policy / Standard deviation",
            self.policy.distribution(role="policy").stddev.mean().item())

        if self._learning_rate_scheduler:
            self.track_data("Learning / Learning rate",
                            self.scheduler.get_last_lr()[0])
