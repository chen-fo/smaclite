"""Exercise both renderer paths even if Windows system font discovery is broken."""

import numpy as np
import pygame
import pytest

from smaclite.env import SMAClitePOSGEnv


@pytest.mark.parametrize("map_name", ["MMM2", "2s3z"])
@pytest.mark.parametrize("render_mode", ["human", "rgb_array"])
def test_render_does_not_depend_on_system_fonts(monkeypatch, map_name, render_mode):
    # Exercise the actual display code without opening test windows.
    pygame.quit()
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")

    def broken_sysfont(*args, **kwargs):
        raise TypeError("expected str, bytes or os.PathLike object, not int")

    monkeypatch.setattr(pygame.font, "SysFont", broken_sysfont)
    env = SMAClitePOSGEnv(map_name=map_name, render_mode=render_mode)
    try:
        env.reset(seed=42)
        first = env.render()
        env.step({team: [1] * size for team, size in env.team_sizes.items()})
        second = env.render()
        if render_mode == "rgb_array":
            for frame in (first, second):
                assert frame.shape == (env.map_info.height * 32,
                                       env.map_info.width * 32, 3)
                assert frame.dtype == np.uint8
                assert np.ptp(frame) > 0
        else:
            assert first is None and second is None
            assert pygame.display.get_surface() is not None
    finally:
        env.close()
        pygame.quit()
