from copy import deepcopy
from dataclasses import replace

import gymnasium as gym
import numpy as np
import pytest

import smaclite  # noqa: F401 -- registers Gymnasium IDs
from smaclite.env import SMACliteEnv, SMAClitePOSGEnv
from smaclite.env.maps.map import Group, MapPreset
from smaclite.env.units.unit_command import StopCommand
from smaclite.env.units.unit_type import StandardUnit as U
from smaclite.env.util.faction import Faction


TEAMS = ("team_0", "team_1")


def make_env(rosters=None, **kwargs):
    rosters = rosters or (((U.MARINE, 1),), ((U.MARINE, 1),))
    types = list(dict.fromkeys(kind for roster in rosters for kind, _ in roster))
    map_info = replace(
        deepcopy(MapPreset.MAP_10M_VS_11M.value), name="test",
        num_allied_units=sum(count for _, count in rosters[0]),
        num_enemy_units=sum(count for _, count in rosters[1]),
        groups=[Group(10, 10, Faction.ALLY, list(rosters[0])),
                Group(14, 10, Faction.ENEMY, list(rosters[1]))],
        num_unit_types=len(types) if len(types) > 1 else 0,
        unit_type_ids={kind: index for index, kind in enumerate(types)},
    )
    return SMAClitePOSGEnv(map_info=map_info, **kwargs)


def position(env, positions):
    for team, points in positions.items():
        units = env.agents if team == "team_0" else env.enemies
        for index, point in enumerate(points):
            units[index].pos[:] = point
    for finder in (env.neighbour_finder_all, env.neighbour_finder_ally,
                   env.neighbour_finder_enemy):
        finder.update()


def stops(env):
    return {team: [1 if index in env._units(team) else 0 for index in range(size)]
            for team, size in env.team_sizes.items()}


@pytest.mark.parametrize("preset", list(MapPreset))
def test_all_maps_accept_both_teams_and_match_spaces(preset):
    env = SMAClitePOSGEnv(map_info=preset.value, episode_limit=4)
    obs, info = env.reset(seed=12)
    assert env.observation_space.contains(obs)
    schema = env.get_env_info()
    assert schema["n_agents"] == {"team_0": preset.value.num_allied_units,
                                  "team_1": preset.value.num_enemy_units}
    rng = np.random.default_rng(1)
    for step in range(4):
        actions = {team: np.array([rng.choice(np.flatnonzero(row))
                                   for row in info["avail_actions"][team]])
                   for team in TEAMS}
        assert env.action_space.contains(actions)
        obs, rewards, terminated, truncated, info = env.step(actions)
        assert env.observation_space.contains(obs)
        assert rewards == {"team_0": 0.0, "team_1": 0.0}
        assert np.isfinite(env.get_state()).all()
        for team in TEAMS:
            assert obs[team].shape[0] == schema["n_agents"][team]
            assert info["avail_actions"][team].shape == (
                schema["n_agents"][team], schema["n_actions"][team])
        if terminated or truncated:
            break
    assert terminated or truncated
    env.close()


def test_stop_does_not_move_or_attack_on_either_side():
    env = make_env()
    env.reset(seed=4)
    before = {uid: (unit.pos.copy(), unit.hp) for uid, unit in env.all_units.items()}
    for _ in range(3):
        env.step(stops(env))
    for uid, (pos, hp) in before.items():
        np.testing.assert_array_equal(env.all_units[uid].pos, pos)
        assert env.all_units[uid].hp == hp
        assert env.all_units[uid].target is None


def test_both_teams_obey_movement_actions():
    env = make_env()
    env.reset(seed=7)
    position(env, {"team_0": [(8, 10)], "team_1": [(20, 10)]})
    env.step({"team_0": [4], "team_1": [5]})  # East and west in world coordinates.
    assert env.agents[0].pos[0] > 8
    assert env.enemies[0].pos[0] < 20
    assert env.agents[0].hp == env.agents[0].max_hp


@pytest.mark.parametrize("attacker", TEAMS)
def test_each_team_can_explicitly_attack_without_retaliation(attacker):
    env = make_env()
    env.reset(seed=8)
    actions = stops(env)
    actions[attacker][0] = 6
    env.step(actions)
    defender = env._other(attacker)
    assert env._units(defender)[0].hp < env._units(defender)[0].max_hp
    assert env._units(attacker)[0].hp == env._units(attacker)[0].max_hp


def test_both_actions_are_applied_in_the_same_step():
    env = make_env()
    env.reset(seed=8)
    env.step({"team_0": [6], "team_1": [6]})
    assert env.agents[0].hp < env.agents[0].max_hp
    assert env.enemies[0].hp < env.enemies[0].max_hp


def test_complete_duel_and_terminal_reward_transition():
    transitions = []

    def reward_fn(transition):
        transitions.append(transition)
        return {team: 0.0 for team in TEAMS}

    env = make_env(reward_fn=reward_fn)
    env.reset(seed=8)
    for _ in range(100):
        _, rewards, terminated, truncated, info = env.step(
            {"team_0": [6], "team_1": [6]})
        if terminated or truncated:
            break
    assert terminated and not truncated
    assert info["winner"] in TEAMS
    assert transitions[-1].winner == info["winner"]
    assert transitions[-1].terminated
    assert transitions[-1].termination_reason == "elimination"
    assert len(transitions) == env.step_count
    assert rewards == {team: 0.0 for team in TEAMS}


def test_masks_respect_unit_mobility_and_target_planes():
    env = make_env((((U.SPINE_CRAWLER, 1),), ((U.MEDIVAC, 1),)))
    env.reset(seed=3)
    # An immobile ground-only attacker cannot move or attack a flying medivac.
    assert np.flatnonzero(env.get_avail_actions("team_0")[0]).tolist() == [1]
    # The opposing healer cannot heal enemies, or itself.
    assert not env.get_avail_actions("team_1")[0, 6:].any()


@pytest.mark.parametrize("team", TEAMS)
def test_hidden_opponents_do_not_leak_through_obs_or_masks(team):
    env = make_env()
    env.reset(seed=8)
    position(env, {"team_0": [(4, 4)], "team_1": [(24, 24)]})
    before_obs = env.get_obs(team)
    before_mask = env.get_avail_actions(team)
    before_state = env.get_state()
    hidden = env._units(env._other(team))[0]
    hidden.hp /= 2
    hidden.pos += np.array([1, 1])
    np.testing.assert_array_equal(env.get_obs(team), before_obs)
    np.testing.assert_array_equal(env.get_avail_actions(team), before_mask)
    assert not np.array_equal(env.get_state(), before_state)
    assert np.count_nonzero(before_obs[0, 4:4 + env.entity_feat_size]) == 0


@pytest.mark.parametrize("healer_team", TEAMS)
def test_healing_uses_stable_friendly_ids_and_never_enemy_ids(healer_team):
    roster = ((U.MEDIVAC, 1), (U.MARINE, 2))
    rosters = (roster, ((U.MARINE, 1),)) if healer_team == "team_0" else (
        ((U.MARINE, 1),), roster)
    env = make_env(rosters)
    env.reset(seed=2)
    friendly = env._units(healer_team)
    friendly[2].hp = 10
    actions = stops(env)
    actions[healer_team][0] = 8  # Heal friendly ID 2, beyond opposing team's size.
    mask = env.get_avail_actions(healer_team)
    assert mask[0, 8]
    assert not mask[0, 6]  # Cannot heal self.
    # Enemy attackable bits must not be mistaken for friendly healing masks.
    assert env.get_obs(healer_team)[0, 4] == 0
    env.step(actions)
    assert friendly[2].hp > 10
    assert friendly[0].target is friendly[2]


def test_dead_slots_are_stable_and_do_not_retarget_automatically():
    env = make_env((((U.MARINE, 2),), ((U.MARINE, 1),)))
    env.reset(seed=2)
    position(env, {"team_0": [(10, 10), (11, 12)], "team_1": [(14, 10)]})
    env.agents[0].hp = 1
    obs, _, terminated, _, _ = env.step({"team_0": [1, 1], "team_1": [6]})
    assert not terminated
    assert set(env.agents) == {1}
    assert not obs["team_0"][0].any()
    assert obs["team_0"][1].any()
    assert np.flatnonzero(env.get_avail_actions("team_0")[0]).tolist() == [0]
    assert isinstance(env.enemies[0].command, StopCommand)
    assert env.agents[1].hp == env.agents[1].max_hp
    env.step({"team_0": [0, 1], "team_1": [1]})
    with pytest.raises(ValueError, match="Invalid action"):
        env.step({"team_0": [1, 1], "team_1": [1]})


@pytest.mark.parametrize("actions", [
    {"team_0": [1]}, {"team_0": [1], "team_1": [1], "extra": [1]},
    {"team_0": [1], "team_1": []}, {"team_0": [1], "team_1": 1},
    {"team_0": [1], "team_1": [-1]}, {"team_0": [1], "team_1": [7]},
    {"team_0": [1], "team_1": [0]}, {"team_0": [1], "team_1": [1.0]},
    {"team_0": [1], "team_1": [True]}, {"team_0": [1], "team_1": [np.nan]},
])
def test_invalid_joint_action_is_rejected_before_any_mutation(actions):
    env = make_env()
    env.reset(seed=11)
    state = env.get_state()
    rng_state = deepcopy(env.np_random.bit_generator.state)
    commands = [u.command for u in env.all_units.values()]
    with pytest.raises(ValueError):
        env.step(actions)
    np.testing.assert_array_equal(env.get_state(), state)
    assert env.np_random.bit_generator.state == rng_state
    assert [u.command for u in env.all_units.values()] == commands


def test_terminal_and_truncated_are_distinct_and_rewards_stay_zero():
    env = make_env(episode_limit=1)
    env.reset(seed=3)
    env.agents[0].hp = 1
    obs, rewards, terminated, truncated, info = env.step(
        {"team_0": [1], "team_1": [6]})
    assert terminated and not truncated
    assert info["winner"] == "team_1"
    assert info["termination_reason"] == "elimination"
    assert rewards == {"team_0": 0.0, "team_1": 0.0}
    assert not obs["team_0"].any()
    assert not env.neighbour_finder_ally.kd_trees  # No stale dead-unit index.
    with pytest.raises(RuntimeError, match="finished"):
        env.step({"team_0": [0], "team_1": [1]})
    env.reset(seed=3)
    _, _, terminated, truncated, info = env.step(stops(env))
    assert not terminated and truncated
    assert info["winner"] is None
    assert info["termination_reason"] == "time_limit"


def test_no_combat_units_is_a_draw():
    env = make_env((((U.MEDIVAC, 1),), ((U.MEDIVAC, 1),)))
    env.reset(seed=2)
    _, _, terminated, truncated, info = env.step(stops(env))
    assert terminated and not truncated
    assert info["winner"] is None
    assert info["termination_reason"] == "no_combat_units"


def test_reward_hook_accepts_non_zero_sum_and_detached_transition():
    transitions = []

    def reward_fn(transition):
        transitions.append(transition)
        assert not transition.state_before.flags.writeable
        assert not transition.state_after.flags.writeable
        with pytest.raises(TypeError):
            transition.actions["team_0"] = (6,)
        return {"team_0": 2.5, "team_1": 4.0}

    env = make_env(reward_fn=reward_fn, episode_limit=2)
    env.reset(seed=2)
    before = env.get_state()
    _, rewards, _, _, _ = env.step(stops(env))
    assert rewards == {"team_0": 2.5, "team_1": 4.0}
    np.testing.assert_array_equal(transitions[0].state_before, before)
    saved_after = transitions[0].state_after.copy()
    env.step(stops(env))
    np.testing.assert_array_equal(transitions[0].state_after, saved_after)
    assert transitions[1].truncated
    assert transitions[1].termination_reason == "time_limit"


@pytest.mark.parametrize("result", [0, {"team_0": 1},
                                     {"team_0": np.nan, "team_1": 0},
                                     {"team_0": [1], "team_1": 0}])
def test_invalid_reward_results_fail_clearly(result):
    env = make_env(reward_fn=lambda transition: result)
    env.reset(seed=2)
    with pytest.raises(ValueError, match="reward_fn"):
        env.step(stops(env))


def test_state_history_and_observations_are_detached_and_seeded():
    env = make_env()
    with pytest.raises(RuntimeError, match="reset"):
        env.step({"team_0": [1], "team_1": [1]})
    assert env.get_env_info()["n_agents"] == {"team_0": 1, "team_1": 1}
    env.reset(seed=9)
    env.step({"team_0": [6], "team_1": [6]})
    state = env.get_state()
    for team in TEAMS:
        layout = env.get_env_info()["state_layout"][f"{team}_last_actions"]
        assert state[layout["offset"] + 6] == 1
    env.reset(seed=9)
    env.step({"team_0": [6], "team_1": [6]})
    np.testing.assert_array_equal(env.get_state(), state)
    obs = env.get_obs_agent("team_0", 0)
    obs[:] = 99
    assert not np.any(env.get_obs_agent("team_0", 0) == 99)
    state[:] = 99
    assert not np.any(env.get_state() == 99)


def test_registered_posg_and_legacy_environment_both_work():
    env = gym.make("smaclite/10m_vs_11m-posg-v0", episode_limit=1)
    obs, _ = env.reset(seed=4)
    assert env.observation_space.contains(obs)
    _, rewards, _, truncated, _ = env.step(stops(env.unwrapped))
    assert isinstance(rewards, dict) and truncated
    env.close()
    legacy = SMACliteEnv(map_info=MapPreset.MAP_10M_VS_11M.value)
    obs, _ = legacy.reset(seed=4)
    assert isinstance(obs, tuple)
    _, reward, _, _, _ = legacy.step([1] * legacy.n_agents)
    assert isinstance(reward, float)
    legacy.close()


def test_combat_telemetry_attributes_damage_to_both_sides_without_kill_bonus():
    env = make_env(record_combat=True)
    _, initial = env.reset(seed=7)
    ally, enemy = env.agents[0], env.enemies[0]
    enemy.hp = 2
    legacy_reward = ally.deal_damage(enemy)
    assert legacy_reward == 12  # Preserve old enemy death bonus, separately.
    ally.deal_damage(enemy)
    assert ally.hp_damage_dealt == 2  # Neither kill bonus nor overkill.
    ally.hp = 3
    assert enemy.deal_damage(ally) == 0  # Legacy code ignores ally damage.
    assert enemy.hp_damage_dealt == 3
    info = env._info()
    assert info["combat"]["team_0"]["hp_damage_dealt"].tolist() == [2]
    assert info["combat"]["team_1"]["hp_damage_dealt"].tolist() == [3]
    assert initial["combat"]["team_0"]["hp_damage_dealt"].tolist() == [0]
    info["combat"]["team_0"]["hp_damage_dealt"][:] = 100
    assert ally.hp_damage_dealt == 2
    env.reset(seed=7)
    assert env._info()["combat"]["team_0"]["hp_damage_dealt"].tolist() == [0]
    env.close()


def test_telemetry_keeps_damage_of_an_attacker_after_it_dies():
    env = make_env((((U.MARINE, 2),), ((U.MARINE, 1),)), record_combat=True)
    env.reset(seed=2)
    env.agents[0].deal_damage(env.enemies[0])
    env.agents[0].hp = 1
    _, _, _, _, info = env.step({"team_0": [1, 1], "team_1": [6]})
    assert 0 not in env.agents
    assert info["combat"]["team_0"]["hp_damage_dealt"].tolist() == [6, 0]
    assert info["combat"]["team_0"]["hp"][0] == 0
    env.close()


def test_privileged_combat_telemetry_is_opt_in():
    env = make_env()
    _, info = env.reset(seed=1)
    assert "combat" not in info
    env.close()
