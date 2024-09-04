specs = {
    "adversarial_pursuit": {
        "handle_groups": {"predator", "prey"},
        "n_agents": 75,
        "action_shape": {"predator": 9, "prey": 13},
        "observation_shape": {"predator": (9, 9, 5), "prey": (10, 10, 9)},
        "state_shape": (45, 45, 5),
    },
    "battle": {
        "handle_groups": {"red", "blue"},
        "n_agents": 162,
        "action_shape": 21,
        "observation_shape": (13, 13, 5),
        "state_shape": (45, 45, 5),
    },
    "battlefield": {
        "handle_groups": {"red", "blue"},
        "n_agents": 24,
        "action_shape": 21,
        "observation_shape": (13, 13, 5),
        "state_shape": (80, 80, 5),
    },
    "combined_arms": {
        "handle_groups": {"redmelee", "redranged", "bluemelee", "blueranged"},
        "n_agents": 162,
        "action_shape": {
            "redmelee": 9,
            "redranged": 25,
            "bluemelee": 9,
            "blueranged": 25,
        },
        "observation_shape": (13, 13, 9),
        "state_shape": (45, 45, 9),
    },
    "gather": {
        "handle_groups": {"omnivore"},
        "n_agents": 495,
        "action_shape": 33,
        "observation_shape": (15, 15, 5),
        "state_shape": (200, 200, 5),
    },
    "tiger_deer": {
        "handle_groups": {"deer", "tiger"},
        "n_agents": 121,
        "action_shape": {"deer": 5, "tiger": 9},
        "observation_shape": {"deer": (3, 3, 5), "tiger": (9, 9, 5)},
        "state_shape": (45, 45, 5),
    },
}

for envs, spec in specs.items():
    for spec_name in ["action_shape", "observation_shape"]:
        if not isinstance(spec[spec_name], dict):
            spec[spec_name] = {
                handle: spec[spec_name] for handle in spec["handle_groups"]
            }
