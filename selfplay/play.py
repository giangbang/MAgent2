import cv2
import random
import numpy as np
import torch
import os

from magent2.specs import specs


def gameplay_video(
    vis_env, env_name, seed, q_networks, device, vid_dir, update_steps=None
):
    os.makedirs(vid_dir, exist_ok=True)

    # play agains selfplay agents
    vis_env.reset(seed + 1)
    print("Number of agents:", len(vis_env.agents))
    fps = 30
    frames = []

    while len(vis_env.agents) > 0:
        frames.append(vis_env.render())

        actions = {}
        observation = vis_env._compute_observations()

        for agent in vis_env.agents:
            agent_handle = agent.split("_")[0]

            if random.random() < 0.05:  # 5% random actions, similar to atari
                action = vis_env.action_space_of_agent(agent).sample()
            else:
                with torch.no_grad():
                    q_value = q_networks[agent_handle](
                        torch.Tensor(observation[agent]).permute(2, 0, 1).to(device)
                    )
                assert len(q_value.shape) == 2, q_value.shape
                action = torch.argmax(q_value, dim=1).cpu().numpy()[0]

            actions[agent] = action
        vis_env.step(actions)

    height, width, _ = frames[0].shape

    if update_steps is not None:
        update_steps = str(update_steps)
    else:
        update_steps = "final"

    out = cv2.VideoWriter(
        os.path.join(vid_dir, f"play_agains_self_{update_steps}.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    out.release()

    # play agains random agents
    vis_env.reset()
    frames = [vis_env.render()]

    while len(vis_env.agents) > 0:
        frames.append(vis_env.render())

        actions = {}
        observation = vis_env._compute_observations()

        for agent in vis_env.agents:
            agent_handle = agent.split("_")[0]

            # agents with the following handles act as random bot
            random_agent = agent_handle in [
                "prey",
                "red",
                "redmelee",
                "redranged",
                "deer",
            ]

            if (
                random_agent or random.random() < 0.05
            ):  # 5% random actions, similar to atari
                action = vis_env.action_space_of_agent(agent).sample()
            else:
                with torch.no_grad():
                    q_value = q_networks[agent_handle](
                        torch.Tensor(observation[agent]).permute(2, 0, 1).to(device)
                    )
                assert len(q_value.shape) == 2, q_value.shape
                action = torch.argmax(q_value, dim=1).cpu().numpy()[0]

            actions[agent] = action
        vis_env.step(actions)

    height, width, _ = frames[0].shape

    out = cv2.VideoWriter(
        os.path.join(vid_dir, f"play_agains_random_{update_steps}.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    out.release()

    print("Done recording video")


def load_model(model_path, env_id):
    model_path = os.path.join(model_path, env_id)
    env_name = "_".join(env_id.split("_")[:-1])
    from train import QNetwork

    q_networks = {}
    for handle in specs[env_name]["handle_groups"]:
        p = os.path.join(model_path, handle + ".pt")
        q_networks[handle] = torch.load(p, weights_only=False)
    return q_networks


def load_model_state_dict(model_path, env_id):
    model_path = os.path.join(model_path, env_id)
    env_name = "_".join(env_id.split("_")[:-1])
    from train import QNetwork

    q_networks = {}
    for handle in specs[env_name]["handle_groups"]:
        p = os.path.join(model_path, handle + ".pt")
        model = QNetwork(
            specs[env_name]["observation_shape"][handle],
            specs[env_name]["action_shape"][handle],
        )
        model.load_state_dict(torch.load(p, weights_only=True))
        q_networks[handle] = model
    return q_networks


if __name__ == "__main__":
    env_id = "battle_v4"
    env_name = "_".join(env_id.split("_")[:-1])
    q_networks = load_model_state_dict("magent2/pretrained_model", env_id)
    print("Models loaded")
    from train import make_env

    vis_env = make_env(env_name, 1, render=True)()

    gameplay_video(
        vis_env=vis_env,
        env_name=env_name,
        seed=1,
        q_networks=q_networks,
        device="cpu",
        vid_dir="video",
    )
