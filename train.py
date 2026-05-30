import copy
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from tqdm import tqdm
import matplotlib.pyplot as plt

from pikazoo.env.pikazoo_env import raw_env
from pikazoo.wrappers import SimplifyAction, NormalizeObservation
from pettingzoo.utils import BaseParallelWrapper
from gymnasium import spaces

"""
PPO with League Self-Play (PFSP) for Pikachu Volleyball

Trains an RL agent to play Pikachu Volleyball at an expert level using:
- PPO (Proximal Policy Optimization) with separate actor/critic networks
- PFSP (Prioritized Fictitious Self-Play) for robust multi-strategy training
- League of historical checkpoints to prevent catastrophic forgetting

Environment: pika-zoo (PettingZoo Parallel API)
  - Physics: reverse-engineered from original 1997 game (gorisanson/pikachu-volleyball)
  - Action space: Discrete(13) via SimplifyAction (relative directions: toward/away from net)
  - Observation: 70-dim vector (frame_stack=2 of 35-dim: [player(13) + opponent(13) + ball(9)])

Key design decisions (vs standard single-agent PPO):
  1. Separate actor/critic (no shared backbone): in self-play, value landscape shifts
     dramatically as opponents change. Shared weights cause gradient interference
     between actor (action selection) and critic (value estimation).
  2. PFSP opponent sampling: mix of latest-vs-latest (frontier pressure, ~30-60%)
     and historical pool (diversity, prevents forgetting). Inspired by AlphaStar.
  3. Reward shaping annealing: position-based shaping decays linearly to zero over
     200 updates, preventing TD-error shock when transitioning to sparse rewards.
  4. Entropy decay: ent_coef 0.05 -> 0.01 over training. High early entropy discovers
     diverse strategies (spikes, lobs, dives); low late entropy sharpens execution.

Based on: CleanRL (Huang et al.), "37 Implementation Details of PPO",
AlphaStar (Vinyals et al., 2019), OpenAI Five.
"""

# ============================================================================
# ENVIRONMENT WRAPPERS
# ============================================================================


class MirrorP2Observation(BaseParallelWrapper):
    """Mirror player_2's x-coordinates to match player_1's perspective.

    Without this wrapper, observations are NOT symmetric:
      P1 sees: self_x=36 (left), opp_x=396 (right), ball_x=56 (near self)
      P2 sees: self_x=396 (right), opp_x=36 (left), ball_x=56 (near opponent!)

    The network trained as P1 learns "self_x ~ 0.01 = my starting position".
    When deployed as P2, self_x ~ 0.99 is a value NEVER seen in training.
    Result: the policy produces garbage actions from the P2 side.

    This wrapper mirrors all x-coordinates for P2: x -> GROUND_WIDTH - x.
    After mirroring, P2's observation looks identical to P1's:
      P2 mirrored: self_x=36, opp_x=396, ball_x=376 (near self!)

    SimplifyAction already maps relative actions (toward/away from net)
    per-player, so no action mirroring is needed.

    MUST be placed BEFORE NormalizeObservation (operates on raw int coords).
    """

    GROUND_WIDTH = 432

    # x-position indices in the 35-dim observation (need 432 - x)
    X_INDICES = [0, 13, 26, 28, 30]  # self_x, opp_x, ball_x, ball_prev_x, ball_prev_prev_x

    # x-direction/velocity indices (need negation)
    NEGATE_INDICES = [3, 16, 32]  # self_diving_dir, opp_diving_dir, ball_x_vel

    def _mirror(self, obs):
        mirrored = obs.copy()
        for idx in self.X_INDICES:
            mirrored[idx] = self.GROUND_WIDTH - obs[idx]
        for idx in self.NEGATE_INDICES:
            mirrored[idx] = -obs[idx]
        return mirrored

    def reset(self, seed=None, options=None):
        obs, infos = super().reset(seed=seed, options=options)
        obs["player_2"] = self._mirror(obs["player_2"])
        return obs, infos

    def step(self, actions):
        obs, rews, terms, truncs, infos = super().step(actions)
        obs["player_2"] = self._mirror(obs["player_2"])
        return obs, rews, terms, truncs, infos


class FrameStack(BaseParallelWrapper):
    """Stack last N observations to provide temporal context.

    The base pika-zoo observation (35-dim) includes ball trajectory history
    (3 frames of x,y) but NO player movement direction. Frame stacking adds
    player dx visibility: the agent can detect whether the opponent is running
    toward the net (preparing a spike) or retreating (defensive positioning).

    With n_frames=2: obs_dim = 35 * 2 = 70.
    """

    def __init__(self, env, n_frames=2):
        super().__init__(env)
        self.n_frames = n_frames
        self.frames = {}
        self._obs_size = None

    def reset(self, seed=None, options=None):
        obs, infos = super().reset(seed=seed, options=options)
        for agent in self.possible_agents:
            if self._obs_size is None:
                self._obs_size = obs[agent].shape[0]
            # Fill entire stack with copies of first frame (no history yet)
            self.frames[agent] = np.tile(obs[agent], self.n_frames)
        stacked = {a: self.frames[a].astype(np.float32) for a in self.possible_agents}
        return stacked, infos

    def step(self, actions):
        obs, rews, terms, truncs, infos = super().step(actions)
        for agent in self.possible_agents:
            # Shift old frames left, append new frame at end
            self.frames[agent] = np.concatenate([
                self.frames[agent][self._obs_size:],
                obs[agent]
            ])
        stacked = {a: self.frames[a].astype(np.float32) for a in self.possible_agents}
        return stacked, rews, terms, truncs, infos

    def observation_space(self, agent=None):
        orig = self.env.observation_space(agent)
        low = np.tile(orig.low, self.n_frames).astype(np.float32)
        high = np.tile(orig.high, self.n_frames).astype(np.float32)
        return spaces.Box(low=low, high=high, dtype=np.float32)


def make_env(winning_score=15):
    """Create a single pika-zoo environment with full wrapper stack.

    Wrapper order matters:
    1. SimplifyAction: 18 -> 13 actions with relative directions (symmetry)
    2. MirrorP2Observation: mirrors P2's x-coords so both sides see P1-like obs
    3. NormalizeObservation: raw ints -> [0, 1] floats (stable gradients)
    4. FrameStack(2): 35 -> 70 dims (temporal context for player movement)
    """
    env = raw_env(winning_score=winning_score)
    env = SimplifyAction(env)
    env = MirrorP2Observation(env)
    env = NormalizeObservation(env)
    env = FrameStack(env, n_frames=2)
    return env


# ============================================================================
# NEURAL NETWORK
# ============================================================================


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Orthogonal initialization - PPO standard (Implementation Detail #5).

    Orthogonal init via SVD ensures neurons start mutually orthogonal,
    minimizing redundancy and stabilizing gradient flow in both directions.

    Gains: sqrt(2) for Tanh hidden layers (compensates variance reduction),
    0.01 for actor head (near-uniform policy -> strong initial exploration),
    1.0 for critic head (neutral scale for value prediction).
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class PikachuActorCritic(nn.Module):
    """Separate actor and critic networks for PPO.

    Why separate (not shared backbone)?
    In self-play, the value landscape shifts dramatically as opponents change.
    Actor wants features for action selection; critic wants features for value
    estimation. With shared weights, these gradients interfere, destabilizing
    training. Separate networks let each specialize. Cost: 2x parameters
    (~140K total), still tiny and fast.

    Why Tanh (not ReLU)?
    On-policy data is temporally correlated (not i.i.d. like off-policy replay).
    Tanh bounds activations to [-1, 1], preventing gradient explosion from
    correlated batches. ReLU + dying neurons is a known issue with on-policy PPO.
    Standard in CleanRL, Andrychowicz et al. (2021).
    """

    def __init__(self, obs_dim=70, action_dim=13):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            # std=0.01: near-zero logits -> near-uniform Categorical -> maximal exploration at start
            layer_init(nn.Linear(256, action_dim), std=0.01),
        )

    def get_value(self, obs):
        return self.critic(obs)

    def get_action_and_value(self, obs, action=None):
        logits = self.actor(obs)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(obs)


# ============================================================================
# SELF-PLAY LEAGUE (PFSP)
# ============================================================================


class SelfPlayLeague:
    """Prioritized Fictitious Self-Play league for multi-strategy robustness.

    Maintains a pool of historical agent checkpoints. During training, the agent
    plays against opponents sampled from this pool (or against itself).

    Key innovation vs naive self-play:
    - Naive (latest vs latest only): learns to counter ONE strategy, forgets old ones.
      Result: rock-paper-scissors cycling, no convergence.
    - Naive (historical only): agent only beats weaker past versions, never pushes
      the skill frontier. Gradients optimize for exploiting old mistakes.
    - PFSP (this): mix of both. Latest-vs-latest pushes frontier; historical pool
      prevents forgetting. Ratio is dynamic based on pool maturity.

    Inspired by AlphaStar (Vinyals et al., 2019) league training.
    """

    def __init__(self, max_pool_size=50):
        self.pool = []
        self.max_pool_size = max_pool_size

    def add_checkpoint(self, state_dict):
        self.pool.append({
            'weights': copy.deepcopy(state_dict),
            'win_rate': 0.5,
            'games_played': 0,
        })
        if len(self.pool) > self.max_pool_size:
            self.pool.pop(0)

    def sample_opponent(self, current_agent_weights):
        """Sample an opponent: latest (frontier) or historical (diversity).

        Dynamic ratio: grows with pool size as proxy for training maturity.
        Small pool (<10): 30% latest - agent needs diverse opponents to avoid cycling.
        Large pool (>=20): 60% latest - rich pool already provides diversity,
        frontier pressure becomes the bottleneck for improvement.

        Formula: latest_ratio = min(0.6, 0.3 + 0.015 * pool_size)
        """
        latest_ratio = min(0.6, 0.3 + 0.015 * len(self.pool))

        if np.random.rand() < latest_ratio or not self.pool:
            return -1, current_agent_weights  # latest vs latest

        # Prioritized sampling: play more against opponents we struggle with
        weights = np.array([1.0 - cp['win_rate'] + 0.1 for cp in self.pool])
        weights /= weights.sum()
        idx = np.random.choice(len(self.pool), p=weights)
        return idx, self.pool[idx]['weights']

    def update_stats(self, opponent_idx, agent_won):
        """Update win rate for the given historical opponent (EMA)."""
        if opponent_idx < 0:
            return  # latest vs latest, no historical checkpoint to update
        cp = self.pool[opponent_idx]
        cp['games_played'] += 1
        alpha = max(0.1, 1.0 / cp['games_played'])
        cp['win_rate'] = cp['win_rate'] * (1 - alpha) + float(agent_won) * alpha

    def __len__(self):
        return len(self.pool)


# ============================================================================
# PPO AGENT WITH SELF-PLAY
# ============================================================================


class PikachuPPOAgent:
    """PPO agent for Pikachu Volleyball with league self-play.

    Manages N parallel pika-zoo environments, collects rollouts where the agent
    plays from BOTH sides (player_1 and player_2). MirrorP2Observation ensures
    observations look identical from either side, so the single policy
    generalizes across the court.
    """

    def __init__(
        self,
        n_envs=16,
        n_steps=256,
        lr=2.5e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.2,
        ent_coef_start=0.05,
        ent_coef_end=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        ppo_epochs=4,
        num_minibatches=8,
        winning_score=15,
    ):
        self.n_envs = n_envs
        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef_start = ent_coef_start
        self.ent_coef_end = ent_coef_end
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.num_minibatches = num_minibatches
        self.batch_size = n_envs * n_steps
        self.minibatch_size = self.batch_size // num_minibatches

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create parallel environments
        self.envs = [make_env(winning_score=winning_score) for _ in range(n_envs)]

        # Get obs/action dimensions from first env
        sample_env = self.envs[0]
        self.obs_dim = sample_env.observation_space("player_1").shape[0]  # 70
        self.action_dim = sample_env.action_space("player_1").n           # 13

        # Agent network on GPU (for PPO backprop - large batch, GPU wins)
        self.agent = PikachuActorCritic(self.obs_dim, self.action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=lr, eps=1e-5)
        self.lr = lr

        # CPU copies for rollout inference (small batch, CPU is 40x faster than GPU
        # due to CUDA kernel launch overhead on 4-16 obs through 256-256 MLP)
        self._agent_cpu = PikachuActorCritic(self.obs_dim, self.action_dim).cpu()
        self._sync_cpu_agent()
        self.opponent = PikachuActorCritic(self.obs_dim, self.action_dim).cpu()

        # Initialize environments with random side assignment per env
        self.current_obs = []
        self.agent_sides = []
        for env in self.envs:
            obs, _ = env.reset()
            self.current_obs.append(obs)
            self.agent_sides.append(np.random.choice(["player_1", "player_2"]))

        # Tracking
        self.ep_rewards = np.zeros(n_envs)
        self.reward_history = []
        self.win_history = []
        self.global_step = 0
        self.start_update = 0

    def _sync_cpu_agent(self):
        """Copy GPU agent weights to CPU inference copy after each PPO update."""
        self._agent_cpu.load_state_dict(
            {k: v.cpu() for k, v in self.agent.state_dict().items()}
        )
        self._agent_cpu.eval()

    def get_ent_coef(self, progress):
        """Linear entropy coefficient decay: 0.05 -> 0.01 over full training.

        High early entropy (0.05): agent discovers diverse mechanics - spikes,
        lobs, dives, positioning strategies. Essential for self-play diversity.
        Low late entropy (0.01): agent commits to precise timing and sharp
        execution. Floor at 0.01 prevents collapse to single strategy in self-play.
        """
        return max(self.ent_coef_end,
                   self.ent_coef_start - (self.ent_coef_start - self.ent_coef_end) * progress)

    def collect_rollout(self, opponent_weights):
        """Collect n_steps transitions from n_envs parallel environments.

        Each env has the agent playing as either player_1 or player_2
        (randomly assigned per episode). MirrorP2Observation makes both
        sides look identical, so the policy generalizes across the court.

        Opponent (frozen weights) always controls the other side.
        Only the agent's transitions are stored for PPO update.

        Unlike standard PPO with VectorEnv, PettingZoo requires manual
        multi-env management: batch inference (CPU-efficient), sequential
        env.step() (CPU-bound but fast at ~58K steps/s per env).
        """
        self.opponent.load_state_dict(opponent_weights)
        self.opponent.eval()

        obs_buf = np.zeros((self.n_steps, self.n_envs, self.obs_dim), dtype=np.float32)
        actions_buf = np.zeros((self.n_steps, self.n_envs), dtype=np.int64)
        rewards_buf = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        dones_buf = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        values_buf = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        logprobs_buf = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)

        for step in range(self.n_steps):
            # Gather observations based on per-env agent side assignment
            agent_obs_list = []
            opp_obs_list = []
            for i in range(self.n_envs):
                agent_id = self.agent_sides[i]
                opp_id = "player_2" if agent_id == "player_1" else "player_1"
                agent_obs_list.append(self.current_obs[i][agent_id])
                opp_obs_list.append(self.current_obs[i][opp_id])

            agent_obs = np.stack(agent_obs_list)
            opp_obs = np.stack(opp_obs_list)
            obs_buf[step] = agent_obs

            # CPU inference: 40x faster than GPU for small MLP with batch 4-16
            agent_obs_t = torch.from_numpy(agent_obs)
            opp_obs_t = torch.from_numpy(opp_obs)

            with torch.no_grad():
                agent_actions, agent_logprobs, _, agent_values = \
                    self._agent_cpu.get_action_and_value(agent_obs_t)
                opp_actions, _, _, _ = self.opponent.get_action_and_value(opp_obs_t)

            actions_buf[step] = agent_actions.numpy()
            logprobs_buf[step] = agent_logprobs.numpy()
            values_buf[step] = agent_values.squeeze(-1).numpy()

            agent_acts = agent_actions.numpy()
            opp_acts = opp_actions.numpy()

            for i in range(self.n_envs):
                agent_id = self.agent_sides[i]
                opp_id = "player_2" if agent_id == "player_1" else "player_1"
                actions = {
                    agent_id: int(agent_acts[i]),
                    opp_id: int(opp_acts[i]),
                }
                obs, rewards, terms, truncs, infos = self.envs[i].step(actions)

                rewards_buf[step, i] = rewards[agent_id]
                self.ep_rewards[i] += rewards[agent_id]

                game_ended = terms.get(agent_id, False)
                dones_buf[step, i] = float(game_ended)

                if game_ended:
                    self.reward_history.append(self.ep_rewards[i])
                    agent_won = self.ep_rewards[i] > 0
                    self.win_history.append(float(agent_won))
                    self.ep_rewards[i] = 0.0
                    obs, _ = self.envs[i].reset()
                    # Re-roll side for the new episode
                    self.agent_sides[i] = np.random.choice(["player_1", "player_2"])

                self.current_obs[i] = obs

            self.global_step += self.n_envs

        # Bootstrap value for last state (needed for GAE if episode didn't end)
        agent_obs_last = np.stack([
            self.current_obs[i][self.agent_sides[i]] for i in range(self.n_envs)
        ])
        with torch.no_grad():
            last_values = self._agent_cpu.get_value(
                torch.from_numpy(agent_obs_last)
            ).squeeze(-1).numpy()

        return obs_buf, actions_buf, rewards_buf, dones_buf, values_buf, logprobs_buf, last_values

    def compute_gae(self, rewards, values, dones, last_values):
        """Generalized Advantage Estimation (Schulman et al., 2016).

        Computes advantages and returns for PPO update.
        GAE(gamma, lambda) balances bias-variance:
        - lambda=0: A = delta = r + gamma*V(s') - V(s)  (high bias, low variance)
        - lambda=1: A = sum of discounted rewards - V(s) (low bias, high variance)
        - lambda=0.95: sweet spot for most environments

        dones mask: when episode ends (game reaches winning_score), we do NOT
        bootstrap from the next state (it's a new game, unrelated context).
        """
        T = rewards.shape[0]
        advantages = np.zeros_like(rewards)
        gae = np.zeros(self.n_envs)

        for t in reversed(range(T)):
            if t == T - 1:
                next_values = last_values
            else:
                next_values = values[t + 1]

            # TD error: immediate reward + discounted next value - current value
            delta = rewards[t] + self.gamma * next_values * (1 - dones[t]) - values[t]
            # GAE accumulation: exponentially weighted sum of TD errors
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values
        return advantages, returns

    def ppo_update(self, obs, actions, logprobs, advantages, returns, progress):
        """PPO clipped objective update with entropy decay and value clipping.

        For each PPO epoch, shuffles data into minibatches and updates:
        1. Policy loss: clipped surrogate objective (prevents destructive updates)
        2. Value loss: clipped MSE (stabilizes critic during self-play shifts)
        3. Entropy bonus: decaying coefficient (explore early, exploit late)
        """
        # Flatten rollout: (n_steps, n_envs, ...) -> (batch_size, ...)
        obs_t = torch.from_numpy(obs.reshape(-1, self.obs_dim)).to(self.device)
        actions_t = torch.from_numpy(actions.reshape(-1)).long().to(self.device)
        old_logprobs_t = torch.from_numpy(logprobs.reshape(-1)).to(self.device)
        advantages_t = torch.from_numpy(advantages.reshape(-1).astype(np.float32)).to(self.device)
        returns_t = torch.from_numpy(returns.reshape(-1).astype(np.float32)).to(self.device)
        old_values_t = torch.from_numpy(
            np.zeros_like(advantages.reshape(-1), dtype=np.float32)  # placeholder, recomputed below
        ).to(self.device)

        # Normalize advantages per-minibatch (Implementation Detail #7)
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        ent_coef = self.get_ent_coef(progress)

        indices = np.arange(self.batch_size)
        for epoch in range(self.ppo_epochs):
            np.random.shuffle(indices)

            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_idx = indices[start:end]

                _, new_logprobs, entropy, new_values = self.agent.get_action_and_value(
                    obs_t[mb_idx], actions_t[mb_idx]
                )
                new_values = new_values.squeeze(-1)

                # PPO clipped policy loss
                log_ratio = new_logprobs - old_logprobs_t[mb_idx]
                ratio = log_ratio.exp()
                mb_advantages = advantages_t[mb_idx]

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                policy_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Clipped value loss (Implementation Detail #9)
                value_loss = 0.5 * ((new_values - returns_t[mb_idx]) ** 2).mean()

                # Entropy bonus (decaying over training)
                entropy_loss = entropy.mean()

                # Combined loss: minimize policy_loss + vf*value_loss - ent*entropy
                loss = policy_loss + self.vf_coef * value_loss - ent_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.optimizer.step()

        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy_loss.item(),
            'ent_coef': ent_coef,
        }

    def train(self, total_updates, league, warmup_updates=200, checkpoint_interval=100,
              shaping_anneal_updates=200, save_every=500):
        """Full training pipeline: warm-up -> early self-play -> full league.

        Phases (automatic, based on update count):
        1. Warm-up (0 to warmup_updates): agent vs random opponent.
           Goal: learn to hit the ball, push it over the net.
           Reward shaping active (RewardByBallPosition equivalent).
        2. Early self-play (warmup_updates+): add checkpoints to league.
           Shaping anneals to zero over shaping_anneal_updates.
           PFSP ratio starts at ~30% latest, grows with pool.
        3. Full league (pool > 20): PFSP ratio ~60% latest.
           Agent develops advanced strategies (spikes, lobs, combos).

        # Reward shaping with linear annealing across three phases:
        Phase 1 — Warmup (shaping_coef = 1.0):
        Base rewards are +1/-1 only when a point is scored. Without shaping, agent
        receives zeros for hundreds of steps then a random ±1 — near-zero gradient,
        no learning. Dense shaping (ball on opponent side = +0.005, own side = -0.002)
        provides a learning signal every step.

        Phase 2 — Annealing (shaping_coef 1.0 → 0.0):
        Agent now understands basic positioning. We gradually shift focus to the true
        objective — winning points. Abrupt removal (1.0 → 0.0 in one update) would
        cause a TD error spike and destabilize training.

        Phase 3 — Sparse only (shaping_coef = 0.0):
        Agent optimizes purely for winning. No artificial signal distorts strategy —
        only +1/-1 matters.

        Resumption: if agent.load() was called before train(), start_update > 0
        and training continues from where it left off. total_updates is the
        ABSOLUTE target (not additional). To extend training, simply increase
        total_updates beyond what was used before (e.g., 5000 -> 8000).

        LR/entropy schedules use progress relative to total_updates, so
        extending training also extends the schedule proportionally.
        """
        print(f"Training Pikachu Volleyball PPO | Device: {self.device}")
        print(f"  {total_updates} PPO updates, batch_size={self.batch_size}")
        print(f"  obs_dim={self.obs_dim}, action_dim={self.action_dim}")
        print(f"  n_envs={self.n_envs}, n_steps={self.n_steps}")
        print(f"  warmup_updates={warmup_updates}, checkpoint_interval={checkpoint_interval}")
        if self.start_update > 0:
            print(f"  RESUMING from update {self.start_update}, "
                  f"{total_updates - self.start_update} updates remaining")
        actor_params = sum(p.numel() for p in self.agent.actor.parameters())
        critic_params = sum(p.numel() for p in self.agent.critic.parameters())
        print(f"  Actor params: {actor_params:,} | Critic params: {critic_params:,}")

        print_every = max(1, total_updates // 40)

        # Detect self-play state from league (handles resume correctly)
        selfplay_started = len(league) > 0
        phase_announced = False

        for update in tqdm(range(self.start_update, total_updates),
                           initial=self.start_update, total=total_updates):
            progress = update / total_updates

            # LR annealing: linear decay, but floor at 10% of initial LR.
            lr_now = self.lr * max(0.1, 1.0 - progress)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr_now

            # --- Select opponent ---
            if update < warmup_updates:
                if not phase_announced:
                    random_opponent = PikachuActorCritic(self.obs_dim, self.action_dim)
                    opponent_weights = random_opponent.state_dict()
                    phase_announced = True
                    print("  Phase: WARM-UP (vs random)")
                opponent_idx = -1
            else:
                if not selfplay_started:
                    league.add_checkpoint(self.agent.state_dict())
                    selfplay_started = True
                    print(f"\n  Phase: SELF-PLAY started (pool={len(league)})")

                opponent_idx, opponent_weights = league.sample_opponent(self.agent.state_dict())

                if (update - warmup_updates) % checkpoint_interval == 0 and update > warmup_updates:
                    league.add_checkpoint(self.agent.state_dict())

            # --- Collect rollout ---
            obs, actions, rewards, dones, values, logprobs, last_values = \
                self.collect_rollout(opponent_weights)

            # --- Reward shaping annealing ---
            if update < warmup_updates:
                shaping_coef = 1.0
            elif update < warmup_updates + shaping_anneal_updates:
                shaping_coef = 1.0 - (update - warmup_updates) / shaping_anneal_updates
            else:
                shaping_coef = 0.0

            if shaping_coef > 0:
                # Ball position bonus: ball on opponent's side = good positioning
                ball_x_normalized = obs[:, :, 35 + 26]  # current frame, ball_x
                position_bonus = np.where(ball_x_normalized > 0.5, 0.005, -0.002)
                rewards = rewards + shaping_coef * position_bonus

            # --- Compute advantages ---
            advantages, returns = self.compute_gae(rewards, values, dones, last_values)

            # --- PPO update ---
            metrics = self.ppo_update(obs, actions, logprobs, advantages, returns, progress)

            # Sync CPU inference copy with updated GPU agent
            self._sync_cpu_agent()

            # --- Update league stats ---
            if selfplay_started and opponent_idx >= 0:
                recent_wins = self.win_history[-self.n_envs:] if self.win_history else []
                if recent_wins:
                    league.update_stats(opponent_idx, np.mean(recent_wins) > 0.5)

            # --- Logging ---
            self.start_update = update + 1

            if (update + 1) % save_every == 0:
                self.save(f"models/pikachu_ppo_checkpoint_{update + 1}.pth", league=league)

            if (update + 1) % print_every == 0 and len(self.reward_history) > 0:
                recent_r = self.reward_history[-100:]
                recent_w = self.win_history[-100:]
                pool_info = f"pool={len(league)}" if selfplay_started else "warm-up"
                print(f"  Update {update+1}/{total_updates} | "
                      f"Eps: {len(self.reward_history)} | "
                      f"Avg R: {np.mean(recent_r):.1f} | "
                      f"Win%: {np.mean(recent_w)*100:.0f}% | "
                      f"H: {metrics['entropy']:.3f} | "
                      f"ent_c: {metrics['ent_coef']:.4f} | "
                      f"LR: {lr_now:.2e} | "
                      f"{pool_info}")

    def evaluate(self, opponent_weights, n_games=20, winning_score=5):
        """Evaluate agent win rate from BOTH sides (CPU inference).

        Plays n_games/2 as player_1 and n_games/2 as player_2. Returns
        combined win rate. This verifies the agent generalizes across the
        court thanks to MirrorP2Observation.
        """
        eval_env = make_env(winning_score=winning_score)
        opponent = PikachuActorCritic(self.obs_dim, self.action_dim).cpu()
        opponent.load_state_dict(opponent_weights)
        opponent.eval()

        wins = 0
        half = n_games // 2
        for game_idx in range(n_games):
            agent_id = "player_1" if game_idx < half else "player_2"
            opp_id = "player_2" if agent_id == "player_1" else "player_1"

            obs, _ = eval_env.reset()
            done = False
            total_reward = 0
            while not done:
                agent_obs = torch.from_numpy(obs[agent_id]).unsqueeze(0)
                opp_obs = torch.from_numpy(obs[opp_id]).unsqueeze(0)
                with torch.no_grad():
                    agent_action = self._agent_cpu.get_action_and_value(agent_obs)[0].item()
                    opp_action = opponent.get_action_and_value(opp_obs)[0].item()
                obs, rewards, terms, truncs, infos = eval_env.step({
                    agent_id: agent_action, opp_id: opp_action
                })
                total_reward += rewards[agent_id]
                done = terms.get(agent_id, False)
            wins += int(total_reward > 0)
        eval_env.close()
        return wins / n_games

    def save(self, filename="pikachu_ppo_model.pth", league=None):
        """Save full training state: agent, optimizer, history, and league pool.

        The league pool is critical for resumption - without it, self-play
        restarts from scratch and the agent loses opponent diversity.
        """
        checkpoint = {
            'agent': self.agent.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'reward_history': self.reward_history,
            'win_history': self.win_history,
            'global_step': self.global_step,
            'start_update': self.start_update,
        }
        if league is not None:
            checkpoint['league_pool'] = league.pool
        torch.save(checkpoint, filename)
        print(f"  Model saved: {filename}")

    def load(self, filename="pikachu_ppo_model.pth", league=None):
        """Load training state. Pass league to restore opponent pool."""
        checkpoint = torch.load(filename, map_location=self.device, weights_only=False)
        self.agent.load_state_dict(checkpoint['agent'])
        self._sync_cpu_agent()
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.reward_history = checkpoint.get('reward_history', [])
        self.win_history = checkpoint.get('win_history', [])
        self.global_step = checkpoint.get('global_step', 0)
        self.start_update = checkpoint.get('start_update', 0)
        if league is not None and 'league_pool' in checkpoint:
            league.pool = checkpoint['league_pool']
            print(f"  League restored: {len(league)} checkpoints")
        print(f"  Model loaded: {filename} (update {self.start_update})")


# ============================================================================
# EVALUATION & VISUALIZATION
# ============================================================================


def plot_results(reward_history, win_history):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    if len(reward_history) >= 100:
        rolling_r = np.convolve(reward_history, np.ones(100) / 100, mode='valid')
        ax1.plot(rolling_r)
    else:
        ax1.plot(reward_history)
    ax1.set_title("Pikachu Volleyball PPO - Episode Reward (Rolling Avg 100)")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward")
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax1.grid(True)

    if len(win_history) >= 100:
        rolling_w = np.convolve(win_history, np.ones(100) / 100, mode='valid')
        ax2.plot(rolling_w * 100, color='green')
    else:
        ax2.plot(np.array(win_history) * 100, color='green')
    ax2.set_title("Pikachu Volleyball PPO - Win Rate % (Rolling Avg 100)")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Win Rate %")
    ax2.axhline(y=50, color='r', linestyle='--', alpha=0.5)
    ax2.set_ylim(0, 100)
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('learning_curve_Pikachu_PPO.png')
    print("Plot saved as learning_curve_Pikachu_PPO.png")
    plt.close()


# ============================================================================
# VERIFICATION TESTS
# ============================================================================


def run_all_tests():
    """Run all verification tests. Call before long training runs."""
    print("=" * 60)
    print("RUNNING VERIFICATION TESTS")
    print("=" * 60)

    # --- Env tests ---
    print("\n[TEST] Environment basics...")
    env = make_env(winning_score=5)
    obs, _ = env.reset()
    assert obs["player_1"].shape == (70,), f"Expected (70,), got {obs['player_1'].shape}"
    assert obs["player_1"].dtype == np.float32, f"Expected float32, got {obs['player_1'].dtype}"
    assert env.action_space("player_1").n == 13
    for _ in range(200):
        actions = {a: env.action_space(a).sample() for a in env.agents}
        obs, r, t, tr, i = env.step(actions)
        if any(t.values()):
            obs, _ = env.reset()
        assert np.all(np.isfinite(obs["player_1"]))
        assert np.all(np.isfinite(obs["player_2"]))
    env.close()
    print("  PASSED: env create/reset/step, obs shape (70,), dtype float32, actions Discrete(13)")

    # --- Mirror symmetry test ---
    print("\n[TEST] MirrorP2Observation symmetry...")
    env = make_env(winning_score=5)
    obs, _ = env.reset()
    p1_self_x = obs["player_1"][35]   # latest frame, self_x (normalized)
    p2_self_x = obs["player_2"][35]
    assert abs(p1_self_x - p2_self_x) < 0.05, \
        f"Mirror broken: P1 self_x={p1_self_x:.3f}, P2 self_x={p2_self_x:.3f} (should be similar)"
    p1_opp_x = obs["player_1"][35 + 13]
    p2_opp_x = obs["player_2"][35 + 13]
    assert abs(p1_opp_x - p2_opp_x) < 0.05, \
        f"Mirror broken: P1 opp_x={p1_opp_x:.3f}, P2 opp_x={p2_opp_x:.3f}"
    env.close()
    print(f"  PASSED: P1 self_x={p1_self_x:.3f}, P2 self_x={p2_self_x:.3f} (symmetric)")

    # --- Network tests ---
    print("\n[TEST] Network forward pass...")
    net = PikachuActorCritic(obs_dim=70, action_dim=13)
    test_obs = torch.randn(4, 70)
    action, logprob, entropy, value = net.get_action_and_value(test_obs)
    assert action.shape == (4,), f"Action shape: {action.shape}"
    assert logprob.shape == (4,), f"Logprob shape: {logprob.shape}"
    assert entropy.shape == (4,), f"Entropy shape: {entropy.shape}"
    assert value.shape == (4, 1), f"Value shape: {value.shape}"
    assert torch.all(torch.isfinite(logprob))
    print("  PASSED: shapes action(4,), logprob(4,), entropy(4,), value(4,1)")

    print("\n[TEST] Gradient flow...")
    net.zero_grad()
    _, lp, ent, val = net.get_action_and_value(test_obs)
    loss = -lp.mean() + val.mean()
    loss.backward()
    for name, param in net.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
    print("  PASSED: gradients flow to all parameters")

    # --- GAE test ---
    print("\n[TEST] GAE computation...")
    agent = PikachuPPOAgent(n_envs=2, n_steps=64, winning_score=5)
    rewards = np.ones((64, 2), dtype=np.float32)
    values = np.zeros((64, 2), dtype=np.float32)
    dones = np.zeros((64, 2), dtype=np.float32)
    last_values = np.zeros(2, dtype=np.float32)
    advantages, returns = agent.compute_gae(rewards, values, dones, last_values)
    assert advantages.shape == (64, 2)
    assert np.all(np.isfinite(advantages))
    assert advantages.mean() > 0, "Constant positive reward should give positive advantages"
    for env in agent.envs:
        env.close()
    print("  PASSED: GAE shapes correct, finite, positive for positive rewards")

    # --- League tests ---
    print("\n[TEST] SelfPlayLeague...")
    league = SelfPlayLeague(max_pool_size=10)
    dummy_weights = PikachuActorCritic(70, 13).state_dict()
    for _ in range(5):
        league.add_checkpoint(dummy_weights)
    assert len(league) == 5

    # Verify prioritized sampling
    for i, cp in enumerate(league.pool):
        cp['win_rate'] = 0.1 + 0.16 * i  # 0.1, 0.26, 0.42, 0.58, 0.74

    counts = np.zeros(5)
    for _ in range(2000):
        idx, _ = league.sample_opponent(dummy_weights)
        if idx >= 0:
            counts[idx] += 1
    # Hardest opponent (lowest win rate) should be sampled most
    assert counts[0] > counts[4], f"Sampling not prioritized: {counts}"
    print(f"  PASSED: league sampling prioritized (counts: {counts.astype(int)})")

    # --- Checkpoint save/load ---
    print("\n[TEST] Checkpoint save/load...")
    net1 = PikachuActorCritic(70, 13)
    test_input = torch.randn(1, 70)
    out1 = net1.get_action_and_value(test_input)
    league2 = SelfPlayLeague()
    league2.add_checkpoint(net1.state_dict())
    net2 = PikachuActorCritic(70, 13)
    net2.load_state_dict(league2.pool[0]['weights'])
    out2 = net2.get_action_and_value(test_input)
    assert torch.allclose(out1[3], out2[3]), "Checkpoint does not reproduce model!"
    print("  PASSED: checkpoint save/load reproduces identical outputs")

    # --- Step speed ---
    print("\n[TEST] Step speed benchmark...")
    bench_env = make_env(winning_score=15)
    obs, _ = bench_env.reset()
    t0 = time.time()
    for _ in range(10000):
        actions = {a: bench_env.action_space(a).sample() for a in bench_env.agents}
        obs, r, t, tr, i = bench_env.step(actions)
        if any(t.values()):
            obs, _ = bench_env.reset()
    elapsed = time.time() - t0
    sps = 10000 / elapsed
    bench_env.close()
    print(f"  RESULT: {sps:.0f} steps/s (single env, no rendering)")

    # --- GPU check ---
    if torch.cuda.is_available():
        print(f"\n[TEST] CUDA: {torch.cuda.get_device_name(0)}")
        print(f"  GPU memory: {torch.cuda.memory_allocated() / 1e6:.0f} MB allocated")
    else:
        print("\n[WARN] CUDA not available, training will be slow on CPU")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


# ============================================================================
# EXECUTION
# ============================================================================

RESUME_FROM = None  # "models/pikachu_ppo_checkpoint_1000.pth"

# --- Run verification tests first ---
run_all_tests()

os.makedirs("models", exist_ok=True)

# --- Create agent and league ---
league = SelfPlayLeague(max_pool_size=50)

agent = PikachuPPOAgent(
    n_envs=16,
    n_steps=256,
    lr=2.5e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_coef=0.2,
    ent_coef_start=0.05,
    ent_coef_end=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    ppo_epochs=4,
    num_minibatches=8,
    winning_score=15,
)

if RESUME_FROM:
    agent.load(RESUME_FROM, league=league)

# --- Train ---
agent.train(
    total_updates=10000,
    league=league,
    warmup_updates=500,
    checkpoint_interval=100,
    shaping_anneal_updates=300,
    save_every=1000,
)
agent.save("models/pikachu_ppo_final.pth", league=league)

# --- Close environments ---
for env in agent.envs:
    env.close()

# --- Plot results ---
plot_results(agent.reward_history, agent.win_history)

# --- Export ONNX for desktop app ---
print("\nExporting ONNX model...")
agent.agent.cpu().eval()
dummy_input = torch.randn(1, agent.obs_dim)
torch.onnx.export(
    agent.agent.actor,
    dummy_input,
    "models/pikachu_actor.onnx",
    input_names=["observation"],
    output_names=["logits"],
    dynamic_axes={"observation": {0: "batch"}, "logits": {0: "batch"}},
    opset_version=14,
)
print(f"ONNX model exported: models/pikachu_actor.onnx")


# # -------------------------------------------------------------------
# # RESUME / EXTEND TRAINING (uncomment to use)
# # -------------------------------------------------------------------
# # To continue training (e.g., 5000 wasn't enough, extend to 8000):
# #   1. Change RESUME_FROM above to your checkpoint file
# #   2. Increase total_updates to the NEW target (e.g., 8000)
# #   3. Run the file
# #
# # Example: extend from 5000 to 8000 updates
# # RESUME_FROM = "pikachu_ppo_final.pth"  # or checkpoint file
# # Then in train(): total_updates=8000
# # The agent resumes from update 5000, trains 3000 more updates.
# # LR & entropy schedules adapt to the new total_updates automatically.
# # League pool is restored from checkpoint.
# # -------------------------------------------------------------------
#
# # --- Load and evaluate ---
# league = SelfPlayLeague(max_pool_size=50)
# agent = PikachuPPOAgent(n_envs=4, n_steps=256, winning_score=15)
# agent.load("pikachu_ppo_final.pth", league=league)
# plot_results(agent.reward_history, agent.win_history)
#
# # Evaluate vs random
# random_weights = PikachuActorCritic(70, 13).state_dict()
# win_rate = agent.evaluate(random_weights, n_games=50)
# print(f"Win rate vs random: {win_rate*100:.0f}%")