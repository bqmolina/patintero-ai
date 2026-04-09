from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
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
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
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
		raw_action = dist.mean if deterministic else dist.rsample()
		angle = torch.sigmoid(raw_action) * 360.0
		log_prob = dist.log_prob(raw_action).sum(dim=-1)
		entropy = dist.entropy().sum(dim=-1)
		return angle, log_prob, entropy

	def evaluate_actions(self, obs: torch.Tensor, angle: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
		return Categorical(logits=self.logits_net(obs))

	def sample(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		dist = self.dist(obs)
		action_idx = torch.argmax(dist.probs, dim=-1) if deterministic else dist.sample()
		return action_idx, dist.log_prob(action_idx), dist.entropy()

	def action_to_env(self, action_idx: torch.Tensor) -> torch.Tensor:
		return self.ACTION_MAP.to(action_idx.device)[action_idx]

	def evaluate_actions(self, obs: torch.Tensor, action_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		dist = self.dist(obs)
		return dist.log_prob(action_idx), dist.entropy()


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
	attacker_state_obs: List[np.ndarray] = field(default_factory=list)
	defender_state_obs: List[np.ndarray] = field(default_factory=list)
	attacker_action: List[np.ndarray] = field(default_factory=list)
	defender_action_idx: List[np.ndarray] = field(default_factory=list)
	attacker_logp: List[np.ndarray] = field(default_factory=list)
	defender_logp: List[np.ndarray] = field(default_factory=list)
	attacker_value: List[np.ndarray] = field(default_factory=list)
	defender_value: List[np.ndarray] = field(default_factory=list)
	attacker_reward: List[np.ndarray] = field(default_factory=list)
	defender_reward: List[np.ndarray] = field(default_factory=list)
	done: List[float] = field(default_factory=list)


class MAPPOPolicy:
	def __init__(
		self,
		num_attackers: int = 5,
		num_defenders: int = 5,
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
		self.num_attackers = int(num_attackers)
		self.num_defenders = int(num_defenders)
		self.frame_stack = int(frame_stack)

		self.board_dim = 5
		self.score_context_dim = 3
		self.attacker_row_dim = 4
		self.defender_row_dim = 4
		self.global_state_dim = self.num_attackers * self.attacker_row_dim + self.num_defenders * self.defender_row_dim + self.board_dim + self.score_context_dim
		self.attacker_obs_dim = (
			self.attacker_row_dim
			+ self.num_defenders * self.defender_row_dim
			+ (self.num_attackers - 1) * 2
			+ self.board_dim
			+ self.score_context_dim
		)
		self.defender_obs_dim = (
			self.defender_row_dim
			+ self.num_attackers * 2
			+ (self.num_defenders - 1) * 2
			+ (self.num_defenders - 1) * 2
			+ self.board_dim
			+ self.score_context_dim
		)
		self.attacker_state_dim = self.attacker_obs_dim + self.global_state_dim
		self.defender_state_dim = self.defender_obs_dim + self.global_state_dim

		stacked_attacker_obs_dim = self.attacker_obs_dim * self.frame_stack
		stacked_defender_obs_dim = self.defender_obs_dim * self.frame_stack
		stacked_attacker_state_dim = self.attacker_state_dim * self.frame_stack
		stacked_defender_state_dim = self.defender_state_dim * self.frame_stack

		self.attacker_local_history = deque(maxlen=self.frame_stack)
		self.defender_local_history = deque(maxlen=self.frame_stack)
		self.attacker_state_history = deque(maxlen=self.frame_stack)
		self.defender_state_history = deque(maxlen=self.frame_stack)

		self.attacker_actor = AttackerActor(stacked_attacker_obs_dim).to(self.device)
		self.defender_actor = DefenderActor(stacked_defender_obs_dim).to(self.device)
		self.attacker_critic = Critic(stacked_attacker_state_dim).to(self.device)
		self.defender_critic = Critic(stacked_defender_state_dim).to(self.device)

		self.attacker_actor_opt = optim.Adam(self.attacker_actor.parameters(), lr=actor_lr)
		self.defender_actor_opt = optim.Adam(self.defender_actor.parameters(), lr=actor_lr)
		self.attacker_critic_opt = optim.Adam(self.attacker_critic.parameters(), lr=critic_lr)
		self.defender_critic_opt = optim.Adam(self.defender_critic.parameters(), lr=critic_lr)

		self.gamma = gamma
		self.gae_lambda = gae_lambda
		self.clip_eps = clip_eps
		self.vf_coef = vf_coef
		self.ent_coef = ent_coef

	def _build_local_observations(self, obs: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
		attackers = np.asarray(obs["attackers"], dtype=np.float32)
		defenders = np.asarray(obs["defenders"], dtype=np.float32)
		board = np.asarray(obs["board"], dtype=np.float32).reshape(-1)
		global_state = np.asarray(obs["state"], dtype=np.float32).reshape(-1)
		score_context = global_state[-self.score_context_dim :]
		defender_positions = defenders[:, :2]
		defender_meta = defenders[:, 2:4]

		attacker_local_obs = []
		attacker_state_obs = []
		for attacker_idx in range(self.num_attackers):
			self_obs = attackers[attacker_idx]
			self_pos = self_obs[:2]
			rel_defenders = (defender_positions - self_pos).reshape(-1)
			defender_meta_flat = defender_meta.reshape(-1)
			rel_attackers = np.delete(attackers[:, :2], attacker_idx, axis=0) - self_pos
			local_obs = np.concatenate(
				[self_obs, rel_defenders, defender_meta_flat, rel_attackers.reshape(-1), board, score_context],
				axis=0,
			)
			attacker_local_obs.append(local_obs)
			attacker_state_obs.append(np.concatenate([global_state, local_obs], axis=0))

		defender_local_obs = []
		defender_state_obs = []
		for defender_idx in range(self.num_defenders):
			self_obs = defenders[defender_idx]
			self_pos = self_obs[:2]
			rel_attackers = attackers[:, :2] - self_pos
			rel_defenders = np.delete(defender_positions, defender_idx, axis=0) - self_pos
			rel_defender_meta = np.delete(defender_meta, defender_idx, axis=0).reshape(-1)
			local_obs = np.concatenate(
				[self_obs, rel_attackers.reshape(-1), rel_defenders.reshape(-1), rel_defender_meta, board, score_context],
				axis=0,
			)
			defender_local_obs.append(local_obs)
			defender_state_obs.append(np.concatenate([global_state, local_obs], axis=0))

		return (
			np.asarray(attacker_local_obs, dtype=np.float32),
			np.asarray(attacker_state_obs, dtype=np.float32),
			np.asarray(defender_local_obs, dtype=np.float32),
			np.asarray(defender_state_obs, dtype=np.float32),
		)

	def reset_history(self) -> None:
		self.attacker_local_history.clear()
		self.defender_local_history.clear()
		self.attacker_state_history.clear()
		self.defender_state_history.clear()

	def _observe(
		self,
		obs: Dict[str, np.ndarray],
	) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
		attacker_local_obs, attacker_state_obs, defender_local_obs, defender_state_obs = self._build_local_observations(obs)

		if len(self.attacker_local_history) == 0:
			for _ in range(self.frame_stack):
				self.attacker_local_history.append(attacker_local_obs.copy())
				self.defender_local_history.append(defender_local_obs.copy())
				self.attacker_state_history.append(attacker_state_obs.copy())
				self.defender_state_history.append(defender_state_obs.copy())
		else:
			self.attacker_local_history.append(attacker_local_obs.copy())
			self.defender_local_history.append(defender_local_obs.copy())
			self.attacker_state_history.append(attacker_state_obs.copy())
			self.defender_state_history.append(defender_state_obs.copy())

		stacked_attacker_local_obs = np.concatenate(list(self.attacker_local_history), axis=-1).astype(np.float32)
		stacked_defender_local_obs = np.concatenate(list(self.defender_local_history), axis=-1).astype(np.float32)
		stacked_attacker_state_obs = np.concatenate(list(self.attacker_state_history), axis=-1).astype(np.float32)
		stacked_defender_state_obs = np.concatenate(list(self.defender_state_history), axis=-1).astype(np.float32)
		return stacked_attacker_local_obs, stacked_attacker_state_obs, stacked_defender_local_obs, stacked_defender_state_obs

	def act(self, obs: Dict[str, np.ndarray], deterministic: bool = False) -> Tuple[np.ndarray, np.ndarray]:
		stacked_attacker_local_obs, _, stacked_defender_local_obs, _ = self._observe(obs)
		attacker_obs = _to_tensor(stacked_attacker_local_obs, self.device)
		defender_obs = _to_tensor(stacked_defender_local_obs, self.device)

		with torch.no_grad():
			attacker_angle, _, _ = self.attacker_actor.sample(attacker_obs, deterministic=deterministic)
			defender_idx, _, _ = self.defender_actor.sample(defender_obs, deterministic=deterministic)
			defender_action = self.defender_actor.action_to_env(defender_idx)

		return attacker_angle.squeeze(-1).cpu().numpy().astype(np.float32), defender_action.cpu().numpy().astype(np.int64)

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

		for _ in range(max_steps):
			attacker_local_obs, attacker_state_obs, defender_local_obs, defender_state_obs = self._observe(obs)

			attacker_local_obs_t = _to_tensor(attacker_local_obs, self.device)
			attacker_state_obs_t = _to_tensor(attacker_state_obs, self.device)
			defender_local_obs_t = _to_tensor(defender_local_obs, self.device)
			defender_state_obs_t = _to_tensor(defender_state_obs, self.device)

			with torch.no_grad():
				attacker_angle, attacker_logp, _ = self.attacker_actor.sample(attacker_local_obs_t, deterministic=deterministic)
				defender_idx, defender_logp, _ = self.defender_actor.sample(defender_local_obs_t, deterministic=deterministic)
				defender_action = self.defender_actor.action_to_env(defender_idx)

				attacker_value = self.attacker_critic(attacker_state_obs_t)
				defender_value = self.defender_critic(defender_state_obs_t)

			next_obs, (attacker_reward, defender_reward), env_done = env.step(
				attacker_angle.squeeze(-1).cpu().numpy(),
				defender_action.cpu().numpy(),
			)

			render_closed = False
			if renderer is not None:
				renderer.render()
				render_closed = not renderer.running

			done = bool(env_done or render_closed)

			buffer.attacker_obs.append(attacker_local_obs)
			buffer.defender_obs.append(defender_local_obs)
			buffer.attacker_state_obs.append(attacker_state_obs)
			buffer.defender_state_obs.append(defender_state_obs)
			buffer.attacker_action.append(attacker_angle.squeeze(-1).cpu().numpy().astype(np.float32))
			buffer.defender_action_idx.append(defender_idx.cpu().numpy().astype(np.int64))
			buffer.attacker_logp.append(attacker_logp.cpu().numpy().astype(np.float32))
			buffer.defender_logp.append(defender_logp.cpu().numpy().astype(np.float32))
			buffer.attacker_value.append(attacker_value.cpu().numpy().astype(np.float32))
			buffer.defender_value.append(defender_value.cpu().numpy().astype(np.float32))
			buffer.attacker_reward.append(np.asarray(attacker_reward, dtype=np.float32))
			buffer.defender_reward.append(np.asarray(defender_reward, dtype=np.float32))
			buffer.done.append(float(done))

			total_attacker_reward += float(np.sum(attacker_reward))
			total_defender_reward += float(np.sum(defender_reward))
			steps += 1

			if done:
				break

			obs = next_obs

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

	def _compute_multiagent_gae(self, rewards: np.ndarray, values: np.ndarray, dones: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
		advantages = np.zeros_like(rewards, dtype=np.float32)
		returns = np.zeros_like(rewards, dtype=np.float32)
		for agent_idx in range(rewards.shape[1]):
			agent_advantages, agent_returns = self._compute_gae(rewards[:, agent_idx], values[:, agent_idx], dones)
			advantages[:, agent_idx] = agent_advantages
			returns[:, agent_idx] = agent_returns
		return advantages, returns

	def update(self, buffer: RolloutBuffer, update_epochs: int = 10, minibatch_size: int = 64) -> Dict[str, float]:
		attacker_obs = _to_tensor(np.concatenate(buffer.attacker_obs, axis=0), self.device)
		defender_obs = _to_tensor(np.concatenate(buffer.defender_obs, axis=0), self.device)
		attacker_state_obs = _to_tensor(np.concatenate(buffer.attacker_state_obs, axis=0), self.device)
		defender_state_obs = _to_tensor(np.concatenate(buffer.defender_state_obs, axis=0), self.device)

		attacker_action = _to_tensor(np.concatenate(buffer.attacker_action, axis=0), self.device).unsqueeze(-1)
		defender_action_idx = torch.as_tensor(np.concatenate(buffer.defender_action_idx, axis=0), device=self.device, dtype=torch.long)

		old_attacker_logp = _to_tensor(np.concatenate(buffer.attacker_logp, axis=0), self.device)
		old_defender_logp = _to_tensor(np.concatenate(buffer.defender_logp, axis=0), self.device)

		attacker_value = np.stack(buffer.attacker_value, axis=0)
		defender_value = np.stack(buffer.defender_value, axis=0)
		dones = np.asarray(buffer.done, dtype=np.float32)

		attacker_adv, attacker_ret = self._compute_multiagent_gae(np.stack(buffer.attacker_reward, axis=0), attacker_value, dones)
		defender_adv, defender_ret = self._compute_multiagent_gae(np.stack(buffer.defender_reward, axis=0), defender_value, dones)

		attacker_adv_t = _to_tensor(attacker_adv.reshape(-1), self.device)
		defender_adv_t = _to_tensor(defender_adv.reshape(-1), self.device)
		attacker_ret_t = _to_tensor(attacker_ret.reshape(-1), self.device)
		defender_ret_t = _to_tensor(defender_ret.reshape(-1), self.device)

		attacker_adv_t = (attacker_adv_t - attacker_adv_t.mean()) / (attacker_adv_t.std() + 1e-8)
		defender_adv_t = (defender_adv_t - defender_adv_t.mean()) / (defender_adv_t.std() + 1e-8)

		n = attacker_obs.size(0)
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
				a_state_mb = attacker_state_obs[idx]
				d_state_mb = defender_state_obs[idx]

				a_action_mb = attacker_action[idx]
				d_action_mb = defender_action_idx[idx]

				old_a_logp_mb = old_attacker_logp[idx]
				old_d_logp_mb = old_defender_logp[idx]

				a_adv_mb = attacker_adv_t[idx]
				d_adv_mb = defender_adv_t[idx]

				a_ret_mb = attacker_ret_t[idx]
				d_ret_mb = defender_ret_t[idx]

				new_a_logp, a_entropy = self.attacker_actor.evaluate_actions(a_obs_mb, a_action_mb)
				a_ratio = torch.exp(new_a_logp - old_a_logp_mb)
				a_surr1 = a_ratio * a_adv_mb
				a_surr2 = torch.clamp(a_ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * a_adv_mb
				a_policy_loss = -torch.min(a_surr1, a_surr2).mean()

				a_value_pred = self.attacker_critic(a_state_mb)
				a_value_loss = F.mse_loss(a_value_pred, a_ret_mb)
				a_loss = a_policy_loss + self.vf_coef * a_value_loss - self.ent_coef * a_entropy.mean()

				self.attacker_actor_opt.zero_grad()
				self.attacker_critic_opt.zero_grad()
				a_loss.backward()
				nn.utils.clip_grad_norm_(self.attacker_actor.parameters(), 0.5)
				nn.utils.clip_grad_norm_(self.attacker_critic.parameters(), 0.5)
				self.attacker_actor_opt.step()
				self.attacker_critic_opt.step()

				new_d_logp, d_entropy = self.defender_actor.evaluate_actions(d_obs_mb, d_action_mb)
				d_ratio = torch.exp(new_d_logp - old_d_logp_mb)
				d_surr1 = d_ratio * d_adv_mb
				d_surr2 = torch.clamp(d_ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * d_adv_mb
				d_policy_loss = -torch.min(d_surr1, d_surr2).mean()

				d_value_pred = self.defender_critic(d_state_mb)
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
			for key in stats:
				stats[key] /= count

		return stats

	def train(self, env, total_episodes: int = 1000, max_steps: int = 150, update_epochs: int = 10, minibatch_size: int = 64) -> List[Dict[str, float]]:
		history: List[Dict[str, float]] = []

		for ep in range(1, total_episodes + 1):
			buffer, rollout_stats = self.collect_episode(env, max_steps=max_steps, deterministic=False)
			learn_stats = self.update(buffer, update_epochs=update_epochs, minibatch_size=minibatch_size)
			merged = {**rollout_stats, **learn_stats, "episode": ep}
			history.append(merged)

			if ep % 25 == 0 or ep == 1:
				timestamp = datetime.now().isoformat(timespec="seconds")
				print(
					f"[{timestamp}] Episode {ep:4d} | "
					f"Attacker return: {rollout_stats['attacker_return']:+.3f} | "
					f"Defender return: {rollout_stats['defender_return']:+.3f} | "
					f"Len: {int(rollout_stats['episode_len'])}"
				)

		return history

	def save(self, path: str, extra_state: Optional[Dict[str, Any]] = None) -> None:
		payload = {
			"num_attackers": self.num_attackers,
			"num_defenders": self.num_defenders,
			"frame_stack": self.frame_stack,
			"attacker_obs_dim": self.attacker_obs_dim,
			"defender_obs_dim": self.defender_obs_dim,
			"attacker_state_dim": self.attacker_state_dim,
			"defender_state_dim": self.defender_state_dim,
			"attacker_actor": self.attacker_actor.state_dict(),
			"defender_actor": self.defender_actor.state_dict(),
			"attacker_critic": self.attacker_critic.state_dict(),
			"defender_critic": self.defender_critic.state_dict(),
			"extra_state": extra_state or {},
		}
		torch.save(payload, path)

	def load(self, path: str) -> Dict[str, Any]:
		payload = torch.load(path, map_location=self.device)
		if payload.get("num_attackers", self.num_attackers) != self.num_attackers:
			raise ValueError(
				f"Checkpoint num_attackers={payload.get('num_attackers')} does not match current policy num_attackers={self.num_attackers}."
			)
		if payload.get("num_defenders", self.num_defenders) != self.num_defenders:
			raise ValueError(
				f"Checkpoint num_defenders={payload.get('num_defenders')} does not match current policy num_defenders={self.num_defenders}."
			)
		if payload.get("frame_stack", self.frame_stack) != self.frame_stack:
			raise ValueError(
				f"Checkpoint frame_stack={payload.get('frame_stack')} does not match current policy frame_stack={self.frame_stack}."
			)
		self.attacker_actor.load_state_dict(payload["attacker_actor"])
		self.defender_actor.load_state_dict(payload["defender_actor"])
		self.attacker_critic.load_state_dict(payload["attacker_critic"])
		self.defender_critic.load_state_dict(payload["defender_critic"])
		return payload.get("extra_state", {})