from dataclasses import dataclass, field
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal


def _to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
	return torch.as_tensor(x, dtype=torch.float32, device=device)


class MLP(nn.Module):
	def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 128):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(in_dim, hidden_dim),
			nn.Tanh(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.Tanh(),
			nn.Linear(hidden_dim, out_dim),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.net(x)


class AttackerActor(nn.Module):
	def __init__(self, obs_dim: int):
		super().__init__()
		self.backbone = MLP(obs_dim, 1)
		self.log_std = nn.Parameter(torch.tensor([-0.5], dtype=torch.float32))

	def dist(self, obs: torch.Tensor) -> Normal:
		mean = self.backbone(obs)
		std = torch.exp(self.log_std).clamp(min=1e-3, max=2.0)
		return Normal(mean, std)

	def sample(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		dist = self.dist(obs)
		if deterministic:
			raw_action = dist.mean
		else:
			raw_action = dist.rsample()

		# Convert network output to angle space [0, 360].
		angle = torch.sigmoid(raw_action) * 360.0
		log_prob = dist.log_prob(raw_action).sum(dim=-1)
		entropy = dist.entropy().sum(dim=-1)
		return angle, log_prob, entropy

	def evaluate_actions(self, obs: torch.Tensor, angle: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		# Inverse of angle mapping for PPO likelihood computation.
		bounded = torch.clamp(angle / 360.0, 1e-6, 1.0 - 1e-6)
		raw_action = torch.log(bounded / (1.0 - bounded))

		dist = self.dist(obs)
		log_prob = dist.log_prob(raw_action).sum(dim=-1)
		entropy = dist.entropy().sum(dim=-1)
		return log_prob, entropy


class DefenderActor(nn.Module):
	ACTION_MAP = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32)

	def __init__(self, obs_dim: int):
		super().__init__()
		self.logits_net = MLP(obs_dim, 3)

	def dist(self, obs: torch.Tensor) -> Categorical:
		logits = self.logits_net(obs)
		return Categorical(logits=logits)

	def sample(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		dist = self.dist(obs)
		if deterministic:
			action_idx = torch.argmax(dist.probs, dim=-1)
		else:
			action_idx = dist.sample()
		log_prob = dist.log_prob(action_idx)
		entropy = dist.entropy()
		return action_idx, log_prob, entropy

	def action_to_env(self, action_idx: torch.Tensor) -> torch.Tensor:
		return self.ACTION_MAP.to(action_idx.device)[action_idx]

	def evaluate_actions(self, obs: torch.Tensor, action_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		dist = self.dist(obs)
		log_prob = dist.log_prob(action_idx)
		entropy = dist.entropy()
		return log_prob, entropy


class Critic(nn.Module):
	def __init__(self, state_dim: int):
		super().__init__()
		self.value_net = MLP(state_dim, 1)

	def forward(self, state: torch.Tensor) -> torch.Tensor:
		return self.value_net(state).squeeze(-1)


@dataclass
class RolloutBuffer:
	attacker_obs: List[np.ndarray] = field(default_factory=list)
	defender_obs: List[np.ndarray] = field(default_factory=list)
	state_obs: List[np.ndarray] = field(default_factory=list)
	attacker_action: List[float] = field(default_factory=list)
	defender_action_idx: List[int] = field(default_factory=list)
	attacker_logp: List[float] = field(default_factory=list)
	defender_logp: List[float] = field(default_factory=list)
	attacker_value: List[float] = field(default_factory=list)
	defender_value: List[float] = field(default_factory=list)
	attacker_reward: List[float] = field(default_factory=list)
	defender_reward: List[float] = field(default_factory=list)
	done: List[float] = field(default_factory=list)


class MAPPOPolicy:
	def __init__(
		self,
		attacker_obs_dim: int = 4,
		defender_obs_dim: int = 3,
		state_dim: int = 7,
		frame_stack: int = 5,
		gamma: float = 0.99,
		gae_lambda: float = 0.95,
		clip_eps: float = 0.2,
		vf_coef: float = 0.5,
		ent_coef: float = 0.01,
		actor_lr: float = 3e-4,
		critic_lr: float = 1e-3,
		device: str = "cpu",
	):
		self.device = torch.device(device)
		self.attacker_obs_dim = attacker_obs_dim
		self.defender_obs_dim = defender_obs_dim
		self.state_dim = state_dim
		self.frame_stack = frame_stack

		stacked_attacker_obs_dim = attacker_obs_dim * frame_stack
		stacked_defender_obs_dim = defender_obs_dim * frame_stack
		stacked_state_dim = state_dim * frame_stack

		self.attacker_history = deque(maxlen=frame_stack)
		self.defender_history = deque(maxlen=frame_stack)
		self.state_history = deque(maxlen=frame_stack)

		self.attacker_actor = AttackerActor(stacked_attacker_obs_dim).to(self.device)
		self.defender_actor = DefenderActor(stacked_defender_obs_dim).to(self.device)
		self.attacker_critic = Critic(stacked_state_dim).to(self.device)
		self.defender_critic = Critic(stacked_state_dim).to(self.device)

		self.attacker_actor_opt = optim.Adam(self.attacker_actor.parameters(), lr=actor_lr)
		self.defender_actor_opt = optim.Adam(self.defender_actor.parameters(), lr=actor_lr)
		self.attacker_critic_opt = optim.Adam(self.attacker_critic.parameters(), lr=critic_lr)
		self.defender_critic_opt = optim.Adam(self.defender_critic.parameters(), lr=critic_lr)

		self.gamma = gamma
		self.gae_lambda = gae_lambda
		self.clip_eps = clip_eps
		self.vf_coef = vf_coef
		self.ent_coef = ent_coef

	@staticmethod
	def _build_state(attacker_obs: np.ndarray, defender_obs: np.ndarray) -> np.ndarray:
		return np.concatenate([attacker_obs, defender_obs], axis=0).astype(np.float32)

	def reset_history(self) -> None:
		self.attacker_history.clear()
		self.defender_history.clear()
		self.state_history.clear()

	def _observe(self, obs: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
		attacker_obs = np.asarray(obs["attacker"], dtype=np.float32)
		defender_obs = np.asarray(obs["defender"], dtype=np.float32)
		state_obs = self._build_state(attacker_obs, defender_obs)

		if len(self.attacker_history) == 0:
			for _ in range(self.frame_stack):
				self.attacker_history.append(attacker_obs.copy())
				self.defender_history.append(defender_obs.copy())
				self.state_history.append(state_obs.copy())
		else:
			self.attacker_history.append(attacker_obs.copy())
			self.defender_history.append(defender_obs.copy())
			self.state_history.append(state_obs.copy())

		stacked_attacker_obs = np.concatenate(list(self.attacker_history), axis=0).astype(np.float32)
		stacked_defender_obs = np.concatenate(list(self.defender_history), axis=0).astype(np.float32)
		stacked_state_obs = np.concatenate(list(self.state_history), axis=0).astype(np.float32)
		return stacked_attacker_obs, stacked_defender_obs, stacked_state_obs

	def act(self, obs: Dict[str, np.ndarray], deterministic: bool = False) -> Tuple[float, int]:
		stacked_attacker_obs, stacked_defender_obs, _ = self._observe(obs)
		attacker_obs = _to_tensor(stacked_attacker_obs, self.device).unsqueeze(0)
		defender_obs = _to_tensor(stacked_defender_obs, self.device).unsqueeze(0)

		with torch.no_grad():
			attacker_angle, _, _ = self.attacker_actor.sample(attacker_obs, deterministic=deterministic)
			defender_idx, _, _ = self.defender_actor.sample(defender_obs, deterministic=deterministic)
			defender_action = self.defender_actor.action_to_env(defender_idx)

		return float(attacker_angle.item()), int(defender_action.item())

	def collect_episode(
		self,
		env,
		max_steps: int = 150,
		deterministic: bool = False,
		renderer=None,
	) -> Tuple[RolloutBuffer, Dict[str, float]]:
		buffer = RolloutBuffer()
		obs = env.reset()
		self.reset_history()

		total_attacker_reward = 0.0
		total_defender_reward = 0.0
		steps = 0

		for t in range(max_steps):
			attacker_obs, defender_obs, state = self._observe(obs)

			attacker_obs_t = _to_tensor(attacker_obs, self.device).unsqueeze(0)
			defender_obs_t = _to_tensor(defender_obs, self.device).unsqueeze(0)
			state_t = _to_tensor(state, self.device).unsqueeze(0)

			with torch.no_grad():
				attacker_angle, attacker_logp, _ = self.attacker_actor.sample(attacker_obs_t, deterministic=deterministic)
				defender_idx, defender_logp, _ = self.defender_actor.sample(defender_obs_t, deterministic=deterministic)
				defender_action = self.defender_actor.action_to_env(defender_idx)

				attacker_value = self.attacker_critic(state_t)
				defender_value = self.defender_critic(state_t)

			next_obs, (attacker_reward, defender_reward), env_done = env.step(
				float(attacker_angle.item()),
				int(defender_action.item()),
			)

			if renderer is not None:
				renderer.render()
				if not renderer.running:
					done = True
					buffer.attacker_obs.append(attacker_obs)
					buffer.defender_obs.append(defender_obs)
					buffer.state_obs.append(state)
					buffer.attacker_action.append(float(attacker_angle.item()))
					buffer.defender_action_idx.append(int(defender_idx.item()))
					buffer.attacker_logp.append(float(attacker_logp.item()))
					buffer.defender_logp.append(float(defender_logp.item()))
					buffer.attacker_value.append(float(attacker_value.item()))
					buffer.defender_value.append(float(defender_value.item()))
					buffer.attacker_reward.append(float(attacker_reward))
					buffer.defender_reward.append(float(defender_reward))
					buffer.done.append(float(done))

					total_attacker_reward += float(attacker_reward)
					total_defender_reward += float(defender_reward)
					steps += 1
					break

			done = env_done

			buffer.attacker_obs.append(attacker_obs)
			buffer.defender_obs.append(defender_obs)
			buffer.state_obs.append(state)
			buffer.attacker_action.append(float(attacker_angle.item()))
			buffer.defender_action_idx.append(int(defender_idx.item()))
			buffer.attacker_logp.append(float(attacker_logp.item()))
			buffer.defender_logp.append(float(defender_logp.item()))
			buffer.attacker_value.append(float(attacker_value.item()))
			buffer.defender_value.append(float(defender_value.item()))
			buffer.attacker_reward.append(float(attacker_reward))
			buffer.defender_reward.append(float(defender_reward))
			buffer.done.append(float(done))

			total_attacker_reward += float(attacker_reward)
			total_defender_reward += float(defender_reward)
			steps += 1

			obs = next_obs
			if done:
				break

		return buffer, {
			"attacker_return": total_attacker_reward,
			"defender_return": total_defender_reward,
			"episode_len": steps,
		}

	def _compute_gae(self, rewards: np.ndarray, values: np.ndarray, dones: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
		advantages = np.zeros_like(rewards, dtype=np.float32)
		gae = 0.0
		next_value = 0.0

		for t in reversed(range(len(rewards))):
			mask = 1.0 - dones[t]
			delta = rewards[t] + self.gamma * next_value * mask - values[t]
			gae = delta + self.gamma * self.gae_lambda * mask * gae
			advantages[t] = gae
			next_value = values[t]

		returns = advantages + values
		return advantages, returns

	def update(self, buffer: RolloutBuffer, update_epochs: int = 10, minibatch_size: int = 64) -> Dict[str, float]:
		attacker_obs = _to_tensor(np.array(buffer.attacker_obs, dtype=np.float32), self.device)
		defender_obs = _to_tensor(np.array(buffer.defender_obs, dtype=np.float32), self.device)
		state_obs = _to_tensor(np.array(buffer.state_obs, dtype=np.float32), self.device)

		attacker_action = _to_tensor(np.array(buffer.attacker_action, dtype=np.float32), self.device).unsqueeze(-1)
		defender_action_idx = torch.as_tensor(np.array(buffer.defender_action_idx, dtype=np.int64), device=self.device)

		old_attacker_logp = _to_tensor(np.array(buffer.attacker_logp, dtype=np.float32), self.device)
		old_defender_logp = _to_tensor(np.array(buffer.defender_logp, dtype=np.float32), self.device)

		attacker_value = np.array(buffer.attacker_value, dtype=np.float32)
		defender_value = np.array(buffer.defender_value, dtype=np.float32)
		dones = np.array(buffer.done, dtype=np.float32)

		attacker_adv, attacker_ret = self._compute_gae(np.array(buffer.attacker_reward, dtype=np.float32), attacker_value, dones)
		defender_adv, defender_ret = self._compute_gae(np.array(buffer.defender_reward, dtype=np.float32), defender_value, dones)

		attacker_adv_t = _to_tensor(attacker_adv, self.device)
		defender_adv_t = _to_tensor(defender_adv, self.device)
		attacker_ret_t = _to_tensor(attacker_ret, self.device)
		defender_ret_t = _to_tensor(defender_ret, self.device)

		attacker_adv_t = (attacker_adv_t - attacker_adv_t.mean()) / (attacker_adv_t.std() + 1e-8)
		defender_adv_t = (defender_adv_t - defender_adv_t.mean()) / (defender_adv_t.std() + 1e-8)

		n = state_obs.size(0)
		minibatch_size = min(minibatch_size, n)

		stats = {
			"attacker_policy_loss": 0.0,
			"defender_policy_loss": 0.0,
			"attacker_value_loss": 0.0,
			"defender_value_loss": 0.0,
			"attacker_entropy": 0.0,
			"defender_entropy": 0.0,
		}
		count = 0

		for _ in range(update_epochs):
			indices = torch.randperm(n, device=self.device)
			for start in range(0, n, minibatch_size):
				idx = indices[start : start + minibatch_size]

				a_obs_mb = attacker_obs[idx]
				d_obs_mb = defender_obs[idx]
				s_obs_mb = state_obs[idx]

				a_action_mb = attacker_action[idx]
				d_action_mb = defender_action_idx[idx]

				old_a_logp_mb = old_attacker_logp[idx]
				old_d_logp_mb = old_defender_logp[idx]

				a_adv_mb = attacker_adv_t[idx]
				d_adv_mb = defender_adv_t[idx]

				a_ret_mb = attacker_ret_t[idx]
				d_ret_mb = defender_ret_t[idx]

				# Attacker PPO objective
				new_a_logp, a_entropy = self.attacker_actor.evaluate_actions(a_obs_mb, a_action_mb)
				a_ratio = torch.exp(new_a_logp - old_a_logp_mb)
				a_surr1 = a_ratio * a_adv_mb
				a_surr2 = torch.clamp(a_ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * a_adv_mb
				a_policy_loss = -torch.min(a_surr1, a_surr2).mean()

				a_value_pred = self.attacker_critic(s_obs_mb)
				a_value_loss = F.mse_loss(a_value_pred, a_ret_mb)

				a_loss = a_policy_loss + self.vf_coef * a_value_loss - self.ent_coef * a_entropy.mean()

				self.attacker_actor_opt.zero_grad()
				self.attacker_critic_opt.zero_grad()
				a_loss.backward()
				nn.utils.clip_grad_norm_(self.attacker_actor.parameters(), 0.5)
				nn.utils.clip_grad_norm_(self.attacker_critic.parameters(), 0.5)
				self.attacker_actor_opt.step()
				self.attacker_critic_opt.step()

				# Defender PPO objective
				new_d_logp, d_entropy = self.defender_actor.evaluate_actions(d_obs_mb, d_action_mb)
				d_ratio = torch.exp(new_d_logp - old_d_logp_mb)
				d_surr1 = d_ratio * d_adv_mb
				d_surr2 = torch.clamp(d_ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * d_adv_mb
				d_policy_loss = -torch.min(d_surr1, d_surr2).mean()

				d_value_pred = self.defender_critic(s_obs_mb)
				d_value_loss = F.mse_loss(d_value_pred, d_ret_mb)

				d_loss = d_policy_loss + self.vf_coef * d_value_loss - self.ent_coef * d_entropy.mean()

				self.defender_actor_opt.zero_grad()
				self.defender_critic_opt.zero_grad()
				d_loss.backward()
				nn.utils.clip_grad_norm_(self.defender_actor.parameters(), 0.5)
				nn.utils.clip_grad_norm_(self.defender_critic.parameters(), 0.5)
				self.defender_actor_opt.step()
				self.defender_critic_opt.step()

				stats["attacker_policy_loss"] += float(a_policy_loss.item())
				stats["defender_policy_loss"] += float(d_policy_loss.item())
				stats["attacker_value_loss"] += float(a_value_loss.item())
				stats["defender_value_loss"] += float(d_value_loss.item())
				stats["attacker_entropy"] += float(a_entropy.mean().item())
				stats["defender_entropy"] += float(d_entropy.mean().item())
				count += 1

		if count > 0:
			for k in stats:
				stats[k] /= count

		return stats

	def train(self, env, total_episodes: int = 1000, max_steps: int = 150, update_epochs: int = 10, minibatch_size: int = 64) -> List[Dict[str, float]]:
		history: List[Dict[str, float]] = []

		for ep in range(1, total_episodes + 1):
			buffer, rollout_stats = self.collect_episode(env, max_steps=max_steps, deterministic=False)
			learn_stats = self.update(buffer, update_epochs=update_epochs, minibatch_size=minibatch_size)

			merged = {**rollout_stats, **learn_stats, "episode": ep}
			history.append(merged)

			if ep % 25 == 0 or ep == 1:
				print(
					f"Episode {ep:4d} | "
					f"Attacker return: {rollout_stats['attacker_return']:+.3f} | "
					f"Defender return: {rollout_stats['defender_return']:+.3f} | "
					f"Len: {int(rollout_stats['episode_len'])}"
				)

		return history

	def save(self, path: str, extra_state: Optional[Dict[str, Any]] = None) -> None:
		payload = {
			"frame_stack": self.frame_stack,
			"attacker_actor": self.attacker_actor.state_dict(),
			"defender_actor": self.defender_actor.state_dict(),
			"attacker_critic": self.attacker_critic.state_dict(),
			"defender_critic": self.defender_critic.state_dict(),
			"extra_state": extra_state or {},
		}
		torch.save(payload, path)

	def load(self, path: str) -> Dict[str, Any]:
		payload = torch.load(path, map_location=self.device)
		saved_frame_stack = payload.get("frame_stack", 1)
		if saved_frame_stack != self.frame_stack:
			raise ValueError(
				f"Checkpoint frame_stack={saved_frame_stack} does not match current policy frame_stack={self.frame_stack}."
			)
		self.attacker_actor.load_state_dict(payload["attacker_actor"])
		self.defender_actor.load_state_dict(payload["defender_actor"])
		self.attacker_critic.load_state_dict(payload["attacker_critic"])
		self.defender_critic.load_state_dict(payload["defender_critic"])
		return payload.get("extra_state", {})