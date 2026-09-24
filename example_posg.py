"""External policies driving both teams; run with the mtapro Python interpreter."""

import argparse

import numpy as np

from smaclite.env import SMAClitePOSGEnv
from smaclite.env.maps.map import MapPreset


def random_policy(observations, action_masks, rng):
    """Replace this function with any policy; each row is one agent's input."""
    return np.array([rng.choice(np.flatnonzero(mask))
                     for observation, mask in zip(observations, action_masks)],
                    dtype=np.int64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", default="MMM2",
                        choices=[preset.value.name for preset in MapPreset])
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    # Policies, their random streams and any recurrent memory live outside env.
    policies = {team: random_policy for team in SMAClitePOSGEnv.teams}
    rngs = {team: np.random.default_rng(args.seed + index)
            for index, team in enumerate(SMAClitePOSGEnv.teams)}
    env = SMAClitePOSGEnv(map_name=args.map, episode_limit=args.steps)
    try:
        for episode in range(args.episodes):
            observations, info = env.reset(seed=args.seed + episode)
            terminated = truncated = False
            while not (terminated or truncated):
                actions = {
                    team: policies[team](observations[team],
                                         info["avail_actions"][team], rngs[team])
                    for team in env.teams
                }
                observations, rewards, terminated, truncated, info = env.step(actions)
                if args.render:
                    env.render()
            print(f"episode={episode} steps={info['step_count']} "
                  f"winner={info['winner']} reason={info['termination_reason']} "
                  f"rewards={rewards}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
