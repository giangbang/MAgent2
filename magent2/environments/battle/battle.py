# noqa
"""
## Battle

```{figure} battle.gif
:width: 140px
:name: battle
```

| Import             | `from magent2.environments import battle_v4` |
|--------------------|-------------------------------------------|
| Actions            | Discrete                                  |
| Parallel API       | Yes                                       |
| Manual Control     | No                                        |
| Agents             | `agents= [red_[0-80], blue_[0-80]]`       |
| Agents             | 162                                       |
| Action Shape       | (21)                                      |
| Action Values      | Discrete(21)                              |
| Observation Shape  | (13,13,5)                                 |
| Observation Values | [0,2]                                     |
| State Shape        | (45, 45, 5)                               |
| State Values       | (0, 2)                                    |


A large-scale team battle. Agents are rewarded for their individual performance, and not for the performance of their neighbors, so coordination is difficult.  Agents slowly regain HP over time, so it is best to kill an opposing agent quickly. Specifically, agents have 10 HP, are damaged 2 HP by
each attack, and recover 0.1 HP every turn.

Like all MAgent2 environments, agents can either move or attack each turn. An attack against another agent on their own team will not be registered.

### Arguments

``` python
battle_v4.env(map_size=45, minimap_mode=False, step_reward=-0.005,
dead_penalty=-0.1, attack_penalty=-0.1, attack_opponent_reward=0.2,
max_cycles=1000, extra_features=False)
```

`map_size`: Sets dimensions of the (square) map. Increasing the size increases the number of agents. Minimum size is 12.

`minimap_mode`: Turns on global minimap observations. These observations include your and your opponents piece densities binned over the 2d grid of the observation space. Also includes your `agent_position`, the absolute position on the map (rescaled from 0 to 1).


`step_reward`:  reward after every step

`dead_penalty`:  reward when killed

`attack_penalty`:  reward when attacking anything

`attack_opponent_reward`:  reward added for attacking an opponent

`max_cycles`:  number of frames (a step for each agent) until game terminates

`extra_features`: Adds additional features to observation (see table). Default False

#### Action space

Key: `move_N` means N separate actions, one to move to each of the N nearest squares on the grid.

Action options: `[do_nothing, move_12, attack_8]`

#### Reward

Reward is given as:

* 5 reward for killing an opponent
* -0.005 reward every step (step_reward option)
* -0.1 reward for attacking (attack_penalty option)
* 0.2 reward for attacking an opponent (attack_opponent_reward option)
* -0.1 reward for dying (dead_penalty option)

If multiple options apply, rewards are added.

#### Observation space

The observation space is a 13x13 map with the below channels (in order):

feature | number of channels
--- | ---
obstacle/off the map| 1
my_team_presence| 1
my_team_hp| 1
my_team_minimap(minimap_mode=True)| 1
other_team_presence| 1
other_team_hp| 1
other_team_minimap(minimap_mode=True)| 1
binary_agent_id(extra_features=True)| 10
one_hot_action(extra_features=True)| 21
last_reward(extra_features=True)| 1
agent_position(minimap_mode=True)| 2

### State space

The observation space is a 45x45 map. It contains the following channels, which are (in order):

feature | number of channels
--- | ---
obstacle map| 1
team_0_presence| 1
team_0_hp| 1
team_1_presence| 1
team_1_hp| 1
binary_agent_id(extra_features=True)| 10
one_hot_action(extra_features=True)|  21
last_reward(extra_features=True)| 1


### Version History

* v0: Initial MAgent2 release (0.3.0)

"""

import math

import numpy as np
from gymnasium.utils import EzPickle
from pettingzoo.utils.conversions import parallel_to_aec_wrapper

import magent2
from magent2.environments.magent_env import magent_parallel_env, make_env


default_map_size = 45
max_cycles_default = 300
KILL_REWARD = 5
minimap_mode_default = False
default_reward_args = dict(
    step_reward=-0.005,
    dead_penalty=-2.5,
    attack_penalty=-0.1,
    attack_opponent_reward=0.2,
)


def parallel_env(
    map_size=default_map_size,
    max_cycles=max_cycles_default,
    minimap_mode=minimap_mode_default,
    extra_features=False,
    render_mode=None,
    seed=None,
    **reward_args,
):
    env_reward_args = dict(**default_reward_args)
    env_reward_args.update(reward_args)
    return _parallel_env(
        map_size,
        minimap_mode,
        env_reward_args,
        max_cycles,
        extra_features,
        render_mode,
        seed,
    )


def raw_env(
    map_size=default_map_size,
    max_cycles=max_cycles_default,
    minimap_mode=minimap_mode_default,
    extra_features=False,
    seed=None,
    **reward_args,
):
    # return parallel_to_aec_wrapper(
    #     parallel_env(
    #         map_size, max_cycles, minimap_mode, extra_features, seed=seed, **reward_args
    #     )
    # )
    return parallel_env(
        map_size, max_cycles, minimap_mode, extra_features, seed=seed, **reward_args
    )


env = make_env(raw_env)


def get_config(
    map_size,
    minimap_mode,
    seed,
    step_reward,
    dead_penalty,
    attack_penalty,
    attack_opponent_reward,
):
    gw = magent2.gridworld
    cfg = gw.Config()

    cfg.set({"map_width": map_size, "map_height": map_size})
    cfg.set({"minimap_mode": minimap_mode})
    cfg.set({"embedding_size": 10})
    if seed is not None:
        cfg.set({"seed": seed})

    options = {
        "width": 1,
        "length": 1,
        "hp": 10,
        "speed": 2,
        "view_range": gw.CircleRange(6),
        "attack_range": gw.CircleRange(1.5),
        "damage": 2,
        "kill_reward": KILL_REWARD,
        "step_recover": 0.1,
        "step_reward": step_reward,
        "dead_penalty": dead_penalty,
        "attack_penalty": attack_penalty,
    }
    small = cfg.register_agent_type("small", options)

    g0 = cfg.add_group(small)
    g1 = cfg.add_group(small)

    a = gw.AgentSymbol(g0, index="any")
    b = gw.AgentSymbol(g1, index="any")

    # reward shaping to encourage attack
    cfg.add_reward_rule(
        gw.Event(a, "attack", b), receiver=a, value=attack_opponent_reward
    )
    cfg.add_reward_rule(
        gw.Event(b, "attack", a), receiver=b, value=attack_opponent_reward
    )

    return cfg


class _parallel_env(magent_parallel_env, EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "battle_v4",
        "render_fps": 5,
    }

    def __init__(
        self,
        map_size,
        minimap_mode,
        reward_args,
        max_cycles,
        extra_features,
        render_mode=None,
        seed=None,
    ):
        EzPickle.__init__(
            self,
            map_size,
            minimap_mode,
            reward_args,
            max_cycles,
            extra_features,
            render_mode,
            seed,
        )
        assert map_size >= 12, "size of map must be at least 12"
        env = magent2.GridWorld(
            get_config(map_size, minimap_mode, seed, **reward_args), map_size=map_size
        )
        self.leftID = 0
        self.rightID = 1
        reward_vals = np.array([KILL_REWARD] + list(reward_args.values()))
        reward_range = [
            np.minimum(reward_vals, 0).sum(),
            np.maximum(reward_vals, 0).sum(),
        ]
        names = ["red", "blue"]
        super().__init__(
            env,
            env.get_handles(),
            names,
            map_size,
            max_cycles,
            reward_range,
            minimap_mode,
            extra_features,
            render_mode,
        )

        self._env_id = "battle_v4"

    def generate_map(self):
        env, map_size, handles = self.env, self.map_size, self.handles
        """ generate a map, which consists of two squares of agents"""
        width = height = map_size
        init_num = map_size * map_size * 0.04
        gap = 3

        # left
        n = init_num
        side = int(math.sqrt(n)) * 2
        pos = []
        for x in range(width // 2 - gap - side, width // 2 - gap - side + side, 2):
            for y in range((height - side) // 2, (height - side) // 2 + side, 2):
                if 0 < x < width - 1 and 0 < y < height - 1:
                    pos.append([x, y, 0])
        team1_size = len(pos)
        env.add_agents(handles[self.leftID], method="custom", pos=pos)

        # right
        n = init_num
        side = int(math.sqrt(n)) * 2
        pos = []
        for x in range(width // 2 + gap, width // 2 + gap + side, 2):
            for y in range((height - side) // 2, (height - side) // 2 + side, 2):
                if 0 < x < width - 1 and 0 < y < height - 1:
                    pos.append([x, y, 0])

        pos = pos[:team1_size]
        env.add_agents(handles[self.rightID], method="custom", pos=pos)
