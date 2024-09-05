# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
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
    total_timesteps: int = 5000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments, always fixed with magent"""
    buffer_size: int = 1000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 50
    """the timesteps it takes to update the target network"""
    batch_size: int = 64
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.05
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.5
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 100
    """timestep to start learning"""
    train_frequency: int = 1
    """the frequency of training"""


def make_env(env_id, seed, render=False):
    def thunk():
        import importlib

        importlib.import_module(f"magent2.environments.{env_id}")
        if render:
            env = eval(f"magent2.environments.{env_id}").env(render_mode="rgb_array")
        else:
            env = eval(f"magent2.environments.{env_id}").env()

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
            nn.Flatten(),
        )
        dummy_input = torch.randn(observation_shape).permute(2, 0, 1)
        dummy_output = self.cnn(dummy_input)
        flatten_dim = dummy_output.shape[-1]
        self.network = nn.Sequential(
            nn.Linear(flatten_dim, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, action_shape),
        )

    def forward(self, x):
        x = self.cnn(x)
        return self.network(x)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    args = tyro.cli(Args)
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
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
    envs = make_env(args.env_id, args.seed)()
    env_name = "_".join(args.env_id.split("_")[:-1])

    from magent2.specs import specs

    q_networks = {
        handle: QNetwork(
            specs[env_name]["observation_shape"][handle],
            specs[env_name]["action_shape"][handle],
        ).to(device)
        for handle in specs[env_name]["handle_groups"]
    }
    optimizers = {
        handle: optim.Adam(q_network.parameters(), lr=args.learning_rate)
        for handle, q_network in q_networks.items()
    }
    target_networks = {
        handle: QNetwork(
            specs[env_name]["observation_shape"][handle],
            specs[env_name]["action_shape"][handle],
        ).to(device)
        for handle in specs[env_name]["handle_groups"]
    }
    for handle, target_network in target_networks.items():
        target_network.load_state_dict(q_networks[handle].state_dict())

    rbs = {
        handle: ReplayBuffer(
            args.buffer_size,
            specs[env_name]["observation_shape"][handle],
            1,  # discrete action
            device,
            handle_timeout_termination=False,
        )
        for handle in specs[env_name]["handle_groups"]
    }  # each handle group corresponds to a seperate replay buffer, they share experience within each group
    rewards_count = {}
    current_eps_len = 0

    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    envs.reset(seed=args.seed)
    for global_step, agent in enumerate(envs.agent_iter()):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(
            args.start_e,
            args.end_e,
            args.exploration_fraction * args.total_timesteps,
            global_step,
        )
        current_eps_len += 1
        agent_handle = agent.split("_")[0]
        obs, _, _, _, _ = envs.last()
        if random.random() < epsilon:
            actions = np.random.randint(
                0, specs[env_name]["action_shape"][agent_handle], size=(1,)
            )
        else:
            obs_tensor = torch.Tensor(obs).permute(2, 0, 1).to(device)
            q_values = q_networks[agent_handle](obs_tensor)
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        envs.step(actions[0])
        next_obs, rewards, terminations, truncations, infos = envs.last()
        rewards_count[agent] = rewards + getattr(rewards_count, agent, 0)

        rbs[agent_handle].add(obs, next_obs, actions, rewards, terminations, infos)

        if truncations or terminations:
            envs.reset()
            r = np.sum(np.array(list(rewards_count.values())))
            writer.add_scalar(f"charts/sum_episodic_return", r, global_step)
            writer.add_scalar("charts/episodic_length", current_eps_len, global_step)
            current_eps_len = 0
            rewards_count = {}

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                for handle, rb in rbs.items():
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

                if global_step % 20 == 0:
                    writer.add_scalar("losses/td_loss", loss, global_step)
                    writer.add_scalar(
                        "losses/q_values", old_val.mean().item(), global_step
                    )
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar(
                        "charts/SPS",
                        int(global_step / (time.time() - start_time)),
                        global_step,
                    )

            # update target network
            if global_step % args.target_network_frequency == 0:
                for handle in specs[env_name]["handle_groups"]:
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

        if global_step >= args.total_timesteps:
            break

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}"
        for handle, q_network in q_networks.items():
            torch.save(q_network.state_dict(), os.path.join(model_path, handle + ".pt"))
        print(f"model saved to {model_path}")

    envs.close()
    del envs
    writer.close()

    # visualize results
    vis_env = make_env(args.env_id, args.seed + 1, render=True)()
    vis_env.reset()
    frames = vis_env.render()

    stop = False
    while not stop:
        for agent in vis_env.agent_iter():
            observation, reward, termination, truncation, info = vis_env.last()
            agent_handle = agent_handle = agent.split("_")[0]

            q_value = q_networks[agent_handle](torch.Tensor(observation).to(device))
            action = torch.argmax(q_value, dim=1).cpu().numpy()
            vis_env.step(action)
            frames.append(vis_env.render())
            if terminations or truncations:
                stop = True
                break

    import cv2

    out = cv2.VideoWriter(
        "output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, frames[0].shape[:2]
    )
    for frame in frames:
        out.write(frame)
    out.release()
