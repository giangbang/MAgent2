"""This file is mostly for debugging purpose"""

# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
import copy
import os
import random
import time
from dataclasses import dataclass

# import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
import magent2
from buffers import ReplayBuffer
import magent2.environment
from play import gameplay_video
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None  # type: ignore
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "battle_v4"
    """the id of the environment"""
    map_size: int = 45
    """map size of magent, lower mapsize has lower number of agents"""
    total_timesteps: int = 100000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments, always fixed with magent"""
    buffer_size: int = 50000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 10
    """the timesteps it takes to update the target network"""
    batch_size: int = 128
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.1
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.5
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 1000
    """timestep to start learning"""
    train_frequency: int = 1
    """the frequency of training"""
    random_opponent: bool = False
    """training against other dqn agents or random agents (randomness only apply for `red` team, or `deer`)"""
    sum_reward: bool = False
    """Training with reward sum of all agent in blueteam, for debuging purpose"""
    video_frequency: int = 10_000
    """Frequency for logging video training"""
    share_weight_all: bool = False
    """share weights between all handles (both friend and enemy), this is truly a selfplay setting"""


def make_env(env_id, seed, render=False, **kwargs):
    def thunk():
        import importlib

        importlib.import_module(f"magent2.environments.{env_id}")
        if render:
            env = eval(f"magent2.environments.{env_id}").env(
                **kwargs, render_mode="rgb_array"
            )
        else:
            env = eval(f"magent2.environments.{env_id}").env(**kwargs)

        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, observation_shape, action_shape):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(observation_shape[-1], observation_shape[-1], 3),
            nn.ReLU(),
            nn.Conv2d(observation_shape[-1], observation_shape[-1], 3),
            nn.ReLU(),
        )
        dummy_input = torch.randn(observation_shape).permute(2, 0, 1)
        dummy_output = self.cnn(dummy_input)
        flatten_dim = dummy_output.view(-1).shape[0]
        self.network = nn.Sequential(
            nn.Linear(flatten_dim, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, action_shape),
        )

    def forward(self, x):
        assert len(x.shape) >= 3, "only support magent input observation"
        x = self.cnn(x)
        if len(x.shape) == 3:
            batchsize = 1
        else:
            batchsize = x.shape[0]
        x = x.reshape(batchsize, -1)
        return self.network(x)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    args = tyro.cli(Args)
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    print(args)
    if args.share_weight_all:
        print("Share weight between both teams, true selfplay settings!")
        assert args.env_id not in [
            "tiger_deer_v3",
            "adversarial_pursuit_v4",
            "combined_arms_v6",
        ], "not support heterogeneous env"
        # assert not args.random_opponent, "if use random opponent, do not share weight"
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    import magent2.environments.magent_env

    envs: magent2.environments.magent_env.magent_parallel_env = make_env(
        args.env_id, args.seed, map_size=args.map_size
    )()
    vis_env = make_env(args.env_id, args.seed, render=True)()
    env_name = "_".join(args.env_id.split("_")[:-1])

    q_networks = {
        envs.names[0]: QNetwork(
            envs.observation_spaces[envs.names[0]].shape,
            envs.action_spaces[envs.names[0]].n,
        ).to(device)
    }
    
    print(q_networks[envs.names[0]])

    if args.share_weight_all:
        q_networks = {name: q_networks[envs.names[0]] for name in envs.names}
    else:
        q_networks = {
            name: QNetwork(
                envs.observation_spaces[envs.names[0]].shape,
                envs.action_spaces[envs.names[0]].n,
            ).to(device)
            for name in envs.names
        }

    optimizers = {
        envs.names[0]: optim.Adam(
            q_networks[envs.names[0]].parameters(), lr=args.learning_rate
        )
    }
    if args.share_weight_all:
        optimizers = {name: optimizers[envs.names[0]] for name in envs.names}
    else:
        optimizers = {
            name: optim.Adam(q_network.parameters(), lr=args.learning_rate)
            for name, q_network in q_networks.items()
        }

    target_networks = {
        envs.names[0]: QNetwork(
            envs.observation_spaces[envs.names[0]].shape,
            envs.action_spaces[envs.names[0]].n,
        ).to(device)
    }

    if args.share_weight_all:
        target_networks = {name: target_networks[envs.names[0]] for name in envs.names}
    else:
        target_networks = {
            name: QNetwork(
                envs.observation_spaces[envs.names[0]].shape,
                envs.action_spaces[envs.names[0]].n,
            ).to(device)
            for name in envs.names
        }

    for group, target_network in target_networks.items():
        target_network.load_state_dict(q_networks[group].state_dict())

    if args.share_weight_all:
        args.buffer_size = args.buffer_size * envs.n_agents

    rbs = {
        envs.names[0]: ReplayBuffer(
            args.buffer_size,
            envs.observation_spaces[envs.names[0]].shape,
            1,  # discrete action
            device,
            handle_timeout_termination=False,
        )
    }

    if args.share_weight_all:
        rbs = {name: rbs[envs.names[0]] for name in envs.names}
    else:
        rbs = {
            name: ReplayBuffer(
                args.buffer_size,
                envs.observation_spaces[envs.names[0]].shape,
                1,  # discrete action
                device,
                handle_timeout_termination=False,
            )
            for name in envs.names
        }  # each group corresponds to a seperate replay buffer, they share experience within each group
    current_eps_len = 0

    opponent_groups = []
    if args.random_opponent:  # red is enemy, act random
        opponent_groups = ["prey", "red", "redmelee", "redranged", "deer"]

    env = envs.env
    start_time = time.time()
    current_rewards_of_blueteam = 0

    # TRY NOT TO MODIFY: start the game
    envs.magent_reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        epsilon = linear_schedule(
            args.start_e,
            args.end_e,
            args.exploration_fraction * args.total_timesteps,
            global_step,
        )
        _actions = {}
        obs = envs._compute_observations()
        current_eps_len += 1
        for agent in envs.agents:
            # ALGO LOGIC: put action logic here
            group = agent.split("_")[0]
            if random.random() < epsilon or (
                args.random_opponent and group in opponent_groups
            ):
                # print(group in opponent_groups)
                actions = np.random.randint(0, envs.action_spaces[group].n, size=(1,))
            else:
                obs_tensor = (
                    torch.Tensor(obs[agent]).float().permute(2, 0, 1).to(device)
                )
                with torch.no_grad():
                    q_values = q_networks[group](obs_tensor)
                assert len(q_values.shape) == 2
                actions = torch.argmax(q_values, dim=1).cpu().numpy()
                assert np.prod(actions.shape) == 1

            _actions[agent] = actions[0]

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.magent_step(_actions)

        if args.sum_reward:
            sum_rw = 0
            for agent in obs.keys():
                group = agent.split("_")[0]
                if group not in opponent_groups:
                    sum_rw += rewards[agent]
            sum_rw = np.array((sum_rw,))

        # TRY NOT TO MODIFY: save data to reply buffer;
        for agent in obs.keys():
            group = agent.split("_")[0]
            if args.random_opponent and group in opponent_groups:
                continue
            r = np.array((rewards[agent],))
            if args.sum_reward:
                r = sum_rw
            termin = np.array((terminations[agent],))
            a = np.array((_actions[agent],))
            rbs[group].add(obs[agent], next_obs[agent], a, r, termin, [{}])

        # LOGGING REWARD
        if args.random_opponent:
            for agent, r in rewards.items():
                agent_group = agent.split("_")[0]
                if agent_group not in opponent_groups:
                    current_rewards_of_blueteam += r

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        # obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                for handle, rb in rbs.items():
                    if args.random_opponent and handle in opponent_groups:
                        continue
                    data = rb.sample(args.batch_size)
                    with torch.no_grad():
                        target_max, _ = target_networks[handle](
                            data.next_observations.permute(0, 3, 1, 2)
                        ).max(dim=1)
                        td_target = data.rewards.flatten() + args.gamma * target_max * (
                            1 - data.dones.flatten()
                        )
                    old_val = (
                        q_networks[handle](data.observations.permute(0, 3, 1, 2))
                        .gather(1, data.actions)
                        .squeeze()
                    )
                    loss = F.mse_loss(td_target, old_val)

                    # optimize the model
                    optimizer = optimizers[handle]
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                if global_step % 1000 == 0:
                    writer.add_scalar("losses/td_loss", loss, global_step)
                    writer.add_scalar(
                        "losses/q_values", old_val.mean().item(), global_step
                    )
                    print(
                        "SPS:",
                        int(global_step / (time.time() - start_time)),
                        f", {global_step}/{args.total_timesteps}",
                    )
                    writer.add_scalar(
                        "charts/SPS",
                        int(global_step / (time.time() - start_time)),
                        global_step,
                    )

            # update target network
            if global_step % args.target_network_frequency == 0:
                for handle in target_networks.keys():
                    target_network, q_network = (
                        target_networks[handle],
                        q_networks[handle],
                    )
                    for target_network_param, q_network_param in zip(
                        target_network.parameters(), q_network.parameters()
                    ):
                        target_network_param.data.copy_(
                            args.tau * q_network_param.data
                            + (1.0 - args.tau) * target_network_param.data
                        )

        # VIDEO LOGGING
        if (global_step + 1) % args.video_frequency == 0:
            print("Logging video...")
            model_path = f"runs/{run_name}/{args.exp_name}"
            gameplay_video(
                vis_env=vis_env,
                env_name=env_name,
                seed=args.seed,
                q_networks=q_networks,
                device=device,
                vid_dir=model_path,
                update_steps=global_step,
            )

        # this agents list already handle termination and truncation
        # as dead agents and truncated one are excluded from this list
        if len(envs.agents) == 0:
            obs, _ = envs.magent_reset()
            print(f"An episode ends with length {current_eps_len}")
            print(f"A new episode restarts at step {global_step+1}...")
            if args.random_opponent:
                print("Episode reward of blueteam:", current_rewards_of_blueteam)

            writer.add_scalar("charts/episodic_length", current_eps_len, global_step)
            current_eps_len = 0
            current_rewards_of_blueteam = 0

    model_path = f"runs/{run_name}/{args.exp_name}"
    if args.save_model:
        os.makedirs(model_path, exist_ok=True)
        for handle, q_network in q_networks.items():
            torch.save(q_network.state_dict(), os.path.join(model_path, handle + ".pt"))
        print(f"model saved to {model_path}")

    envs.close()
    del envs
    writer.close()

    # visualize end results
    gameplay_video(
        vis_env=vis_env,
        env_name=env_name,
        seed=args.seed,
        q_networks=q_networks,
        device=device,
        vid_dir=model_path,
    )

    vis_env.close()
