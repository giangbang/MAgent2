import importlib

import magent2
import numpy as np
import math
import pygame


grid_size = 8

env = importlib.import_module("magent2.environments.adversarial_pursuit_v4")

env = env.parallel_env(render_mode="rgb_array")

print(env)

def draw_arrow_head(surface, start, end, color=(0, 255, 0), arrow_size=8):
    """Draws a small arrowhead pointing from start → end"""
    dx, dy = end[0] - start[0], end[1] - start[1]
    angle = math.atan2(dy, dx)
    # Compute two lines forming the arrow tip
    left = (end[0] - arrow_size * math.cos(angle - math.pi / 6),
            end[1] - arrow_size * math.sin(angle - math.pi / 6))
    right = (end[0] - arrow_size * math.cos(angle + math.pi / 6),
             end[1] - arrow_size * math.sin(angle + math.pi / 6))
    pygame.draw.polygon(surface, color, [end, left, right])

def draw_graph_fn(renderer):
    self = renderer
    n_pursuit = 25  # n_pursuit first agent are pursuit
    graph = np.zeros(75*75).reshape(75, 75) 
    graph[[0]*n_pursuit, range(n_pursuit)] = 1

    # Collect agent positions for later
    agent_positions = {}

    resolution = self.resolution

    view_position = [
            self.map_size[0] / 2 * grid_size - resolution[0] / 2,
            self.map_size[1] / 2 * grid_size - resolution[1] / 2,
        ]

    for agent_id, agent_data in self.new_data[0].items():
        x, y, group_id = agent_data[0], agent_data[1], agent_data[2]
        agent_positions[agent_id] = (
            x * grid_size - view_position[0] + grid_size / 2,
            y * grid_size - view_position[1] + grid_size / 2,
        )

    # Suppose self.graph is your n_agent x n_agent matrix
    # or replace with your own adjacency matrix variable

    n_agent = len(graph)
    arrow_color = (0, 200, 0)  # green arrows

    for i in range(n_agent):
        for j in range(n_agent):
            if graph[i, j] > 0 and i in agent_positions and j in agent_positions:
                start = agent_positions[i]
                end = agent_positions[j]

                # Draw main line
                pygame.draw.line(self.canvas, arrow_color, start, end, 2)

                # Draw arrowhead
                draw_arrow_head(self.canvas, start, end, color=arrow_color)



env.reset()

env.step(np.zeros(75))
import matplotlib.pyplot as plt


img = env.render(draw_graph_fn=draw_graph_fn)
print(img)
print(img.shape)

plt.imshow(img)
plt.show()