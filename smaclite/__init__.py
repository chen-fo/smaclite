import gymnasium as gym
from smaclite.env.maps.map import MapPreset

for preset in MapPreset:
    map_info = preset.value
    gym.register(
        f"smaclite/{map_info.name}-v0",
        entry_point="smaclite.env:SMACliteEnv",
        kwargs={"map_info": map_info},
    )
    # Dict-valued team rewards intentionally differ from Gym's scalar reward API.
    gym.register(
        f"smaclite/{map_info.name}-posg-v0",
        entry_point="smaclite.env:SMAClitePOSGEnv",
        kwargs={"map_info": map_info},
        disable_env_checker=True,
    )
gym.register("smaclite/custom-v0", entry_point="smaclite.env:SMACliteEnv")
gym.register("smaclite/custom-posg-v0", entry_point="smaclite.env:SMAClitePOSGEnv",
             disable_env_checker=True)
