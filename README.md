# SMAClite - Starcraft Multi-Agent Challenge lite
This is a repository for the SMAClite environment. It is a (nearly) pure Python reimplementation of the Starcraft Multi-Agent Challenge, using Numpy and OpenAI Gym.

## Two-sided POSG interface: neural-network vs. neural-network play

**支持双方使用神经网络对战。** `SMAClitePOSGEnv` 允许 `team_0` 和 `team_1`
分别由外部神经网络策略控制。双方根据同一时刻的各自局部观测和合法动作掩码选择动作，
再一次性提交双方动作推进战斗。可用于两个独立模型对战、自博弈，或与历史策略进行评估。
模型推理、权重加载和训练由调用方实现。

Both teams can be controlled by external neural-network policies through
`SMAClitePOSGEnv`, enabling independent-policy matches, self-play, and evaluation
against saved opponents. The environment provides:

* Per-unit local observations and legal-action masks for **both** teams.
* Joint action submission as `{"team_0": actions_0, "team_1": actions_1}`.
* Per-team observation/action dimensions via `get_env_info()`, including
  scenarios with different team sizes or unit types.
* An optional global state via `get_state()` for centralized critics, and
  [optional combat telemetry](docs/COMBAT_TELEMETRY.md) for analysis.
* Custom team rewards through `reward_fn`; **all rewards are zero by default**,
  so supply a task-specific reward function when training. Rewards need not be zero-sum.

### Minimal neural-network match (PyTorch)

After [installing the environment](#installation), install the optional PyTorch
dependency with `pip install torch`. This example runs one episode with two
separate, randomly initialized networks. Each network processes its team's units
as a batch; replace its weights with trained weights for a learned-policy match.

```python
import torch
from torch import nn

from smaclite.env import SMAClitePOSGEnv

torch.manual_seed(42)
env = SMAClitePOSGEnv(map_name="MMM2", episode_limit=200)
try:
    schema = env.get_env_info()
    policies = {
        team: nn.Sequential(
            nn.Linear(schema["obs_shape"][team], 64),
            nn.ReLU(),
            nn.Linear(64, schema["n_actions"][team]),
        ).eval()
        for team in env.teams
    }
    observations, info = env.reset(seed=42)
    terminated = truncated = False
    while not (terminated or truncated):
        actions = {}
        with torch.no_grad():
            for team in env.teams:
                obs = torch.as_tensor(observations[team], dtype=torch.float32)
                mask = torch.as_tensor(info["avail_actions"][team], dtype=torch.bool)
                logits = policies[team](obs).masked_fill(~mask, -torch.inf)
                actions[team] = torch.distributions.Categorical(logits=logits).sample().numpy()
        observations, rewards, terminated, truncated, info = env.step(actions)
        # Call env.render() here to view the match.
    print("winner:", info["winner"], "reason:", info["termination_reason"])
finally:
    env.close()
```

Always mask invalid actions before sampling or taking an argmax. Unit slots remain
stable after death; their masks allow only action `0` (no-op). For decentralized
execution, give each unit its own observation row and mask; reserve global state
and combat telemetry for training or evaluation. Shared weights across teams
require compatible observation/action dimensions or an adapter.

For a demo without PyTorch, run the random-policy example:

```bash
python example_posg.py --map MMM2 --episodes 1 --steps 200
# Add --render to view the match.
```

Existing single-sided environments remain available. The POSG interface returns
team reward dictionaries; training frameworks may need an adapter for this
multi-agent interface. See the [POSG interface guide](docs/POSG.md) for reward hooks,
observation layouts, and the full API.

> [!CAUTION]
> The SMAClite environment was updated to support the new [Gymnasium](https://gymnasium.farama.org/) interface in replacement of the deprecated `gym=0.25/0.26` dependency. For backwards compatibility, please see [Gymnasium compatibility documentation](https://gymnasium.farama.org/content/gym_compatibility/) or use version v1.0.0 of the repository.


## Features
The main features of this environment include:
* A fully functional Python implementation of the SMAC environment
* A JSON interface for defining units and scenarios
* Compatibility with the OpenAI Gym API
* (optional) a highly-performant [C++ implementation](https://github.com/micadam/SMAClite-Python-RVO2) of the collision avoidance algorithm

## Available units
The following units are available in this environment:
* baneling
* colossus
* marauder
* marine
* medivac
* spine crawler
* stalker
* zealot
* zergling
## Available scenarios
The following scenarios are available in this environment:
* 10m_vs_11m
* 27m_vs_30m
* 2c_vs_64zg
* 2s3z
* 2s_vs_1sc
* 3s5z
* 3s5z_vs_3s6z
* 3s_vs_5z
* bane_vs_bane
* corridor
* mmm
* mmm2

Note that further scenarios can easily be added by modifying or creating a scenario JSON file.
## Installation
Run
```
pip install .
```
In the SMAClite directory

## Running
As far as we are aware, this project fully adheres to the [OpenAI Gym API](https://www.gymlibrary.dev/), so it can be used with any framework capable of interfacing with Gym-capable environments. We recommend the [ePyMARL](https://github.com/uoe-agents/epymarl) framework, made available in our repository. EPyMARL uses `yaml` files to specify run configurations. To train a model in the `MMM2` scenario using the `MAPPO` algorithm, you can use this example command:
```
python3 src/main.py --config=mappo --env-config=gymma with seed=1 env_args.time_limit=120 env_args.key="smaclite:smaclite/MMM2-v0
```

Note that to use the C++ version of the collision avoidance algorithm, you will have to add the line `use_cpp_rvo2: true` to the `yaml` config file you're referencing, since Sacred does not allow defining new config entries in the command itself.
