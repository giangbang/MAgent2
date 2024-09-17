import copy
import ctypes
import numpy as np
from typing import List
from gymnasium.spaces import Box, Discrete
from gymnasium.utils import seeding
from pettingzoo.utils import wrappers
from pettingzoo.utils.env import ParallelEnv

from magent2 import Renderer
import magent2


def make_env(raw_env):
    def env_fn(**kwargs):
        env = raw_env(**kwargs)
        # env = wrappers.AssertOutOfBoundsWrapper(env)
        # env = wrappers.OrderEnforcingWrapper(env)
        return env

    return env_fn


class magent_parallel_env(ParallelEnv):
    def __init__(
        self,
        env: magent2.GridWorld,
        active_handles: List[ctypes.c_int32],
        names: List[str],
        map_size,
        max_cycles: int,
        reward_range,
        minimap_mode: bool,
        extra_features: bool,
        render_mode=None,
        return_handle_id: int = 1,
    ):
        assert len(names) == len(active_handles)

        self.map_size = map_size
        self.max_cycles = max_cycles
        self.minimap_mode = minimap_mode
        self.extra_features = extra_features
        self.env = env

        self.agents_handle_id = return_handle_id
        self.enemy_handle_id = len(active_handles) - 1 - self.agents_handle_id

        assert isinstance(self.agents_handle_id, int), self.agents_handle_id
        assert isinstance(self.enemy_handle_id, int), self.enemy_handle_id

        # save agent names, such as `blue`, `red`, `deer`, `tiger`, etc
        self.names = names
        self.handles = active_handles
        self._all_handles = self.env.get_handles()
        assert len(self._all_handles) < 3, "not support"
        env.reset()
        self.generate_map()
        self.team_sizes = [
            env.get_num(handle) for handle in self.handles
        ]  # gets updated as agents die
        self.agents = [
            f"{names[j]}_{i}"
            for j in range(len(self.team_sizes))
            for i in range(self.team_sizes[j])
        ]
        self.possible_agents = self.agents[:]
        num_actions = [env.get_action_space(handle)[0] for handle in self.handles]
        action_spaces_list = [
            Discrete(num_actions[j]) for j in range(len(self.team_sizes))
        ]
        # may change depending on environment config? Not sure.
        team_obs_shapes = self._calc_obs_shapes()
        state_shape = self._calc_state_shape()
        observation_space_list = [
            Box(low=0.0, high=2.0, shape=team_obs_shapes[j], dtype=np.float32)
            for j in range(len(self.team_sizes))
        ]

        self.state_space = Box(low=0.0, high=2.0, shape=state_shape, dtype=np.float32)
        reward_low, reward_high = reward_range

        if extra_features:
            for space in observation_space_list:
                idx = space.shape[2] - 3 if minimap_mode else space.shape[2] - 1
                space.low[:, :, idx] = reward_low
                space.high[:, :, idx] = reward_high
            idx_state = (
                self.state_space.shape[2] - 1
                if minimap_mode
                else self.state_space.shape[2] - 1
            )
            self.state_space.low[:, :, idx_state] = reward_low
            self.state_space.high[:, :, idx_state] = reward_high

        self.action_spaces = {
            name: space for name, space in zip(names, action_spaces_list)
        }
        self.observation_spaces = {
            name: space for name, space in zip(names, observation_space_list)
        }

        self._name2handle = {n: h for n, h in zip(names, self._all_handles)}

        self.max_team_size = copy.deepcopy(self.team_sizes)

        # if name == "blue", then you control the blue team
        self.agent_name = self.names[return_handle_id]
        self.n_agents = self.max_team_size[self.agents_handle_id]
        print("Name of the team you controll", self.agent_name)

        self.enemy_name = self.names[self.enemy_handle_id]
        self.n_enemy = self.max_team_size[self.enemy_handle_id]
        print("Name of the enemy team", self.enemy_name)

        assert len(self.team_sizes) == 2, "support two teams, not battle field"

        self._zero_obs = {
            agent: np.zeros_like(self.observation_spaces[agent.split("_")[0]].low)
            for agent in self.agents
        }
        self.base_state = np.zeros(self.state_space.shape, dtype="float32")
        walls = self.env._get_walls_info()  # type: ignore
        wall_x, wall_y = zip(*walls)
        self.base_state[wall_x, wall_y, 0] = 1
        self.render_mode = render_mode
        self._renderer = None
        self.frames = 0

        self.agents_handle = self._all_handles[self.agents_handle_id]
        self.enemy_handle = self._all_handles[self.enemy_handle_id]

        self.agent_action_space = self.action_spaces[self.agent_name]
        self.enemy_action_space = self.action_spaces[self.enemy_name]

        self.agent_observation_space = self.observation_spaces[self.agent_name]
        self.enemy_observation_space = self.observation_spaces[self.enemy_name]

        assert self.names[return_handle_id] in ["blue", "tiger"]
        self.set_random_enemy()  # enemy agents act randomly

        self._env_id = "None"  # declare by subclasses

    def seed(self, seed=None):
        if seed is None:
            _, seed = seeding.np_random()
        self.env.set_seed(seed)

    def set_random_enemy(self, random=True):
        self.random_enemy = random

    def _calc_obs_shapes(self):
        view_spaces = [self.env.get_view_space(handle) for handle in self.handles]
        feature_spaces = [self.env.get_feature_space(handle) for handle in self.handles]
        assert all(len(tup) == 3 for tup in view_spaces)
        assert all(len(tup) == 1 for tup in feature_spaces)
        feat_size = [[fs[0]] for fs in feature_spaces]
        for feature_space in feat_size:
            if not self.extra_features:
                feature_space[0] = 2 if self.minimap_mode else 0
        obs_spaces = [
            (view_space[:2] + (view_space[2] + feature_space[0],))
            for view_space, feature_space in zip(view_spaces, feat_size)
        ]
        return obs_spaces

    def _calc_state_shape(self):
        feature_spaces = [
            self.env.get_feature_space(handle) for handle in self._all_handles
        ]
        self._minimap_features = 2 if self.minimap_mode else 0
        # map channel and agent pair channel. Remove global agent position when minimap mode and extra features
        state_depth = (
            (max(feature_spaces)[0] - self._minimap_features) * self.extra_features
            + 1
            + len(self._all_handles) * 2
        )

        return (self.map_size, self.map_size, state_depth)

    def render(self):
        if self.render_mode is None:
            # gymnasium.logger.WARN(
            #     "You are calling render method without specifying any render mode."
            # )
            return

        if self._renderer is None:
            self._renderer = Renderer(self.env, self.map_size, self.render_mode)
        assert (
            self.render_mode == self._renderer.mode
        ), "mode must be consistent across render calls"
        return self._renderer.render(self.render_mode)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def magent_reset(self, seed=None, return_info=False, options=None):
        """reset function from magent2"""
        if seed is not None:
            self.seed(seed=seed)
        self.agents = self.possible_agents[:]
        self.env.reset()
        self.frames = 0
        self.team_sizes = [self.env.get_num(handle) for handle in self.handles]
        self.generate_map()
        return self._compute_observations(), {}

    def gym_reset(self, seed=None):
        """gym API reset"""
        if seed is not None:
            self.seed(seed=seed)
        self.env.reset()
        self.frames = 0
        self.team_sizes = [self.env.get_num(handle) for handle in self.handles]
        self.generate_map()

        obses = self.get_obses()
        return obses, {}

    def _compute_observations(self):
        """
        Return dict form of observations of each agent
        """
        observes = [None] * self.max_num_agents
        for handle in self.handles:
            ids = self.env.get_agent_id(handle)
            view, features = self.env.get_observation(handle)

            if self.minimap_mode and not self.extra_features:
                features = features[:, -2:]
            if self.minimap_mode or self.extra_features:
                feat_reshape = np.expand_dims(np.expand_dims(features, 1), 1)
                feat_img = np.tile(feat_reshape, (1, view.shape[1], view.shape[2], 1))
                fin_obs = np.concatenate([view, feat_img], axis=-1)
            else:
                fin_obs = np.copy(view)
            for id, obs in zip(ids, fin_obs):
                observes[id] = obs

        ret_agents = set(self.agents)
        return {
            agent: obs if obs is not None else self._zero_obs[agent]
            for agent, obs in zip(self.possible_agents, observes)
            if agent in ret_agents
        }

    def get_obses(self):
        return_obses = np.zeros(
            (self.n_agents, *self.agent_observation_space.shape), dtype=np.float32
        )
        ids = self.env.get_agent_id(self.agents_handle)

        if self.agents_handle_id > 0:
            ids -= np.sum(self.max_team_size[: self.agents_handle_id]).astype(np.int32)

        view, features = self.env.get_observation(self.agents_handle)
        if self.minimap_mode and not self.extra_features:
            features = features[:, -2:]
        if self.minimap_mode or self.extra_features:
            feat_reshape = np.expand_dims(np.expand_dims(features, 1), 1)
            feat_img = np.tile(feat_reshape, (1, view.shape[1], view.shape[2], 1))
            fin_obs = np.concatenate([view, feat_img], axis=-1)
        else:
            fin_obs = np.copy(view)
        return_obses[ids] = fin_obs
        return return_obses

    def get_rewards(self):
        return_rewards = np.zeros(self.n_agents, dtype=np.float32)
        ids = self.env.get_agent_id(self.agents_handle)

        if self.agents_handle_id > 0:
            ids -= np.sum(self.max_team_size[: self.agents_handle_id]).astype(np.int32)

        return_rewards[ids] = self.env.get_reward(self.agents_handle)
        return return_rewards[..., None]

    def get_dones(self, step_done: bool):
        return_dones = np.ones(self.n_agents, dtype=bool)
        if not step_done:
            ids = self.env.get_agent_id(self.agents_handle)
            if self.agents_handle_id > 0:
                ids -= np.sum(self.max_team_size[: self.agents_handle_id]).astype(
                    np.int32
                )

            return_dones[ids] = ~self.env.get_alive(self.agents_handle)
        return return_dones[..., None]

    def _compute_rewards(self):
        """
        Return dict form of rewards of each agent
        """
        rewards = np.zeros(self.max_num_agents)
        for handle in self.handles:
            ids = self.env.get_agent_id(handle)
            rewards[ids] = self.env.get_reward(handle)
        ret_agents = set(self.agents)
        return {
            agent: float(rew)
            for agent, rew in zip(self.possible_agents, rewards)
            if agent in ret_agents
        }

    def _compute_terminates(self, step_done):
        """
        Return dict form of termination states of each agent
        """
        dones = np.ones(self.max_num_agents, dtype=bool)
        if not step_done:
            for i, handle in enumerate(self.handles):
                ids = self.env.get_agent_id(handle)
                dones[ids] = ~self.env.get_alive(handle)
                self.team_sizes[i] = len(ids) - np.array(dones[ids]).sum()
        ret_agents = set(self.agents)
        return {
            agent: bool(done)
            for agent, done in zip(self.possible_agents, dones)
            if agent in ret_agents
        }

    def state(self):
        """Returns an observation of the global environment."""
        state = np.copy(self.base_state)

        for handle in self._all_handles:
            view, features = self.env.get_observation(handle)

            pos = self.env.get_pos(handle)
            if len(pos) == 0:
                continue
            pos_x, pos_y = zip(*pos)
            state[pos_x, pos_y, 1 + handle.value * 2] = 1
            state[pos_x, pos_y, 2 + handle.value * 2] = view[
                :, view.shape[1] // 2, view.shape[2] // 2, 2
            ]

            if self.extra_features:
                add_zeros = np.zeros(
                    (
                        features.shape[0],
                        state.shape[2]
                        - (
                            1
                            + len(self.team_sizes) * 2
                            + features.shape[1]
                            - self._minimap_features
                        ),
                    )
                )

                rewards = features[:, -1 - self._minimap_features]
                actions = features[:, : -1 - self._minimap_features]
                actions = np.concatenate((actions, add_zeros), axis=1)
                rewards = rewards.reshape(len(rewards), 1)
                state_features = np.hstack((actions, rewards))

                state[pos_x, pos_y, 1 + len(self.team_sizes) * 2 :] = state_features
        return state

    def get_enemy_random_actions(self):
        n_enemy_alive = np.sum(self.env.get_alive(self.enemy_handle))

        action_dim_enemy = int(self.enemy_action_space.n)
        # assert isinstance(
        #     action_dim_enemy, int
        # ), f"{self.enemy_action_space.n} {type(self.enemy_action_space.n)}"

        return_action = np.random.randint(
            0, action_dim_enemy, size=(n_enemy_alive,), dtype=np.int32
        )
        return return_action

    def gym_step(self, actions: np.ndarray):
        """Gym API step"""
        assert len(actions) == self.n_agents

        # set the action of the controlled agents
        ids = self.env.get_agent_id(self.agents_handle)
        # print(ids)
        if self.agents_handle_id > 0:
            ids -= np.sum(self.max_team_size[: self.agents_handle_id]).astype(np.int32)

        self.env.set_action(self.agents_handle, actions[ids])

        # set action of the enemy team (not controlled by training agents)
        if self.random_enemy:
            enemy_actions = self.get_enemy_random_actions()
        else:
            enemy_actions = self.get_enemy_pretrained_actions()
        self.env.set_action(self.enemy_handle, enemy_actions)

        self.frames += 1
        step_done = self.env.step()

        next_obses = self.get_obses()
        rewards = self.get_rewards()
        dones = self.get_dones(step_done)

        # IMPORTANT: check this; call clear_dead before or after get return data
        self.env.clear_dead()

        info = {"truncated": self.frames >= self.max_cycles}
        info["bad_transition"] = info["truncated"]  # harl
        if info["truncated"]:
            info["TimeLimit.truncated"] = True  # stable-baselines3
        return next_obses, rewards, dones, info

    def magent_step(self, all_actions: dict):
        """
        Step function from magent2
        Perform one step update to the environment,
        returns result in dict form of each agent
        """
        action_list = [-1] * len(self.agents)
        for i, agent in enumerate(self.agents):
            if agent in all_actions:
                action_list[i] = all_actions[agent]

        all_actions = np.asarray(action_list, dtype=np.int32)
        start_point = 0
        for i in range(len(self.handles)):
            size = self.team_sizes[i]
            self.env.set_action(
                self.handles[i], all_actions[start_point : (start_point + size)]
            )
            start_point += size

        self.frames += 1

        step_done = self.env.step()

        truncations = {agent: self.frames >= self.max_cycles for agent in self.agents}
        terminations = self._compute_terminates(step_done)
        observations = self._compute_observations()
        rewards = self._compute_rewards()
        infos = {agent: {} for agent in self.agents}
        self.env.clear_dead()
        self.agents = [
            agent
            for agent in self.agents
            if not (terminations[agent] or truncations[agent])
        ]

        if self.render_mode == "human":
            self.render()
        return observations, rewards, terminations, truncations, infos

    def load_pretrained_enemy(self):
        print("Loading pretrained enemy...")
        import torch
        import os

        model_path = os.path.dirname(os.path.realpath(__file__))
        model_name = self.enemy_name + ".pt"
        model_dir = os.path.join(
            model_path, "pretrained_model", self._env_id, model_name
        )
        from .torch_model import QNetwork

        self.enemy_model = QNetwork(
            self.enemy_observation_space.shape,
            self.enemy_action_space.n,
        )

        self.device = "cpu"
        self.enemy_model.load_state_dict(
            torch.load(model_dir, weights_only=True, map_location=self.device)
        )
        # self.enemy_model.to(self.device)
        print("Done loading.")

    def get_enemy_obses(self):
        view, features = self.env.get_observation(self.enemy_handle)
        if self.minimap_mode and not self.extra_features:
            features = features[:, -2:]
        if self.minimap_mode or self.extra_features:
            feat_reshape = np.expand_dims(np.expand_dims(features, 1), 1)
            feat_img = np.tile(feat_reshape, (1, view.shape[1], view.shape[2], 1))
            fin_obs = np.concatenate([view, feat_img], axis=-1)
        else:
            fin_obs = np.copy(view)

        return np.array(fin_obs)

    def get_enemy_pretrained_actions(self):
        import torch

        if getattr(self, "enemy_model", None) is None:
            self.load_pretrained_enemy()
        enemy_obses = self.get_enemy_obses()
        enemy_obses = (
            torch.Tensor(enemy_obses).float().permute([0, 3, 1, 2]).to(self.device)
        )

        with torch.no_grad():
            q_values = self.enemy_model(enemy_obses)
        actions = torch.argmax(q_values, dim=1).cpu().numpy()

        random_sample = np.random.rand(actions.shape[0])
        random_action_mask = random_sample < 0.05

        actions[random_action_mask] = np.random.randint(
            0,
            int(self.enemy_action_space.n),
            size=np.sum(random_action_mask.astype(int)),
        )

        return actions.astype(np.int32)

    def generate_map(self):
        raise NotImplementedError("Should be implemented by subclasses.")
