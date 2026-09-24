"""Two externally controlled teams sharing the SMAClite combat simulator."""

from copy import deepcopy
from collections.abc import Mapping
from numbers import Integral, Real
from types import MappingProxyType

import gymnasium as gym
import numpy as np

from smaclite.env.maps.map import MapPreset
from smaclite.env.rewards import TEAMS, RewardTransition, zero_rewards
from smaclite.env.smaclite import (
    AGENT_SIGHT_RANGE, AGENT_TARGET_RANGE, MOVE_AMOUNT, STEP_MUL, SMACliteEnv,
)
from smaclite.env.terrain.terrain import TerrainType
from smaclite.env.units.combat_type import CombatType
from smaclite.env.units.unit_command import AttackUnitCommand, MoveCommand, StopCommand
from smaclite.env.util.direction import Direction
from smaclite.env.util.faction import Faction


class SMAClitePOSGEnv(SMACliteEnv):
    """Simultaneous joint actions, local observations and optional team rewards.

    Actions are ``{team: sequence_of_ints}``; observations and action masks are
    ``{team: array}`` with one row per stable team-local unit ID. Dead slots stay
    present. Rewards are a dict, so this is a multi-agent interface, not a
    scalar-reward Gymnasium environment (nor a PettingZoo ParallelEnv).
    """

    teams = TEAMS

    def __init__(self, map_info=None, map_file=None, *, map_name=None,
                 episode_limit=200, reward_fn=None, record_combat=False, **kwargs):
        if sum(value is not None for value in (map_info, map_file, map_name)) != 1:
            raise ValueError("Provide exactly one of map_name, map_info or map_file")
        if map_name is not None:
            presets = {p.value.name.lower(): p.value for p in MapPreset}
            if not isinstance(map_name, str) or map_name.lower() not in presets:
                raise ValueError(f"Unknown map_name: {map_name!r}")
            map_info = presets[map_name.lower()]
        if (isinstance(episode_limit, bool) or not isinstance(episode_limit, Integral)
                or episode_limit <= 0):
            raise ValueError("episode_limit must be a positive integer")
        if reward_fn is not None and not callable(reward_fn):
            raise TypeError("reward_fn must be callable")
        super().__init__(map_info=map_info, map_file=map_file, **kwargs)
        self.episode_limit = int(episode_limit)
        self.reward_fn = zero_rewards if reward_fn is None else reward_fn
        self.record_combat = bool(record_combat)
        self.team_sizes = dict(zip(TEAMS, (self.n_agents, self.n_enemies)))
        if min(self.team_sizes.values()) <= 0:
            raise ValueError("Each team must have at least one unit")
        self.n_actions_by_team = {}
        for team, faction in zip(TEAMS, (Faction.ALLY, Faction.ENEMY)):
            has_healer = any(
                kind.combat_type == CombatType.HEALING and count > 0
                for group in self.map_info.groups if group.faction == faction
                for kind, count in group.units
            )
            # Target IDs are never compacted, including a healer at slot zero.
            self.n_actions_by_team[team] = 6 + max(
                self.team_sizes[self._other(team)],
                self.team_sizes[team] if has_healer else 0,
            )
        self.entity_feat_size = 6 + self.map_info.num_unit_types
        self.own_feat_size = 4 + self.map_info.num_unit_types
        self.obs_sizes = {
            team: 4 + (sum(self.team_sizes.values()) - 1) * self.entity_feat_size
            + self.own_feat_size for team in TEAMS
        }
        self.action_space = gym.spaces.Dict({
            team: gym.spaces.MultiDiscrete(
                [self.n_actions_by_team[team]] * self.team_sizes[team]
            ) for team in TEAMS
        })
        self.observation_space = gym.spaces.Dict({
            team: gym.spaces.Box(-np.inf, np.inf,
                                 (self.team_sizes[team], self.obs_sizes[team]),
                                 dtype=np.float32) for team in TEAMS
        })
        self.unit_state_fields = (
            "alive", "health", "shield", "cooldown", "energy", "x", "y"
        ) + tuple(f"unit_type_{i}" for i in range(self.map_info.num_unit_types))
        self.state_layout = {}
        offset = 0
        for team in TEAMS:
            shape = (self.team_sizes[team], len(self.unit_state_fields))
            self.state_layout[team] = {"offset": offset, "shape": shape,
                                       "fields": self.unit_state_fields}
            offset += int(np.prod(shape))
        for team in TEAMS:
            shape = (self.team_sizes[team], self.n_actions_by_team[team])
            self.state_layout[f"{team}_last_actions"] = {
                "offset": offset, "shape": shape,
            }
            offset += int(np.prod(shape))
        self.state_layout["episode_progress"] = {"offset": offset, "shape": (1,)}
        self.state_size = offset + 1
        self.state_space = gym.spaces.Box(-np.inf, np.inf, (self.state_size,),
                                          dtype=np.float32)
        self.step_count = 0
        self._has_reset = False
        self._finished = False

    @staticmethod
    def _other(team):
        if team not in TEAMS:
            raise ValueError(f"Unknown team: {team!r}; expected one of {TEAMS}")
        return TEAMS[1] if team == TEAMS[0] else TEAMS[0]

    def _units(self, team):
        self._other(team)  # Validate rather than silently treating a typo as team_1.
        return self.agents if team == TEAMS[0] else self.enemies

    def _require_reset(self):
        if not self._has_reset:
            raise RuntimeError("Call reset() before using the environment")

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        self._reset_world()
        # Keep stable references after deaths remove units from the live world.
        if self.record_combat:
            self._episode_rosters = {
                team: tuple(self._units(team)[i] for i in range(size))
                for team, size in self.team_sizes.items()
            }
        for unit in self.all_units.values():
            unit.command = StopCommand()
        self.last_actions_by_team = {
            team: np.zeros((size, self.n_actions_by_team[team]), dtype=np.float32)
            for team, size in self.team_sizes.items()
        }
        self.step_count = 0
        self._finished = False
        self._has_reset = True
        return self.get_obs(), self._info()

    def _validate_actions(self, actions):
        if not isinstance(actions, Mapping) or set(actions) != set(TEAMS):
            raise ValueError(f"actions must contain exactly {TEAMS}")
        masks = self.get_avail_actions()
        validated = {}
        for team in TEAMS:
            try:
                values = list(actions[team])
            except TypeError as exc:
                raise ValueError(f"{team} actions must be a sequence") from exc
            if len(values) != self.team_sizes[team]:
                raise ValueError(f"{team} requires {self.team_sizes[team]} actions")
            for index, action in enumerate(values):
                if (isinstance(action, (bool, np.bool_))
                        or not isinstance(action, Integral)
                        or not 0 <= action < self.n_actions_by_team[team]
                        or not masks[team][index, action]):
                    raise ValueError(f"Invalid action for {team}[{index}]: {action!r}")
            validated[team] = tuple(int(a) for a in values)
        return validated

    def step(self, actions):
        self._require_reset()
        if self._finished:
            raise RuntimeError("Episode finished; call reset() before step()")
        # Both sides are checked before changing commands, history or physics.
        actions = self._validate_actions(actions)
        before = self.get_state() if self.reward_fn is not zero_rewards else None
        for team in TEAMS:
            self.last_actions_by_team[team] = np.eye(
                self.n_actions_by_team[team], dtype=np.float32
            )[list(actions[team])]
            for index, unit in self._units(team).items():
                action = actions[team][index]
                if action == 1:
                    command = StopCommand()
                elif action < 6:
                    command = MoveCommand(unit.pos + Direction(action - 2).dx_dy
                                          * MOVE_AMOUNT)
                else:
                    target_team = (team if unit.combat_type == CombatType.HEALING
                                   else self._other(team))
                    command = AttackUnitCommand(self._units(target_team)[action - 6])
                unit.command = command
                unit.attacking = False
        winner, reason = self._outcome()
        for _ in range(STEP_MUL):
            if reason is not None:
                break
            for unit in self.all_units.values():
                if (isinstance(unit.command, AttackUnitCommand)
                        and unit.command.target.hp <= 0):
                    unit.command = StopCommand()
                unit.clean_up_target()
            # No attack-move policies, target selection or legacy reward here.
            self._advance_world()
            winner, reason = self._outcome()
        self.step_count += 1
        terminated = reason is not None
        truncated = not terminated and self.step_count >= self.episode_limit
        if truncated:
            reason = "time_limit"
        self._finished = terminated or truncated
        if self.reward_fn is zero_rewards:
            rewards = {team: 0.0 for team in TEAMS}
        else:
            after = self.get_state()
            before.setflags(write=False)
            after.setflags(write=False)
            transition = RewardTransition(
                state_before=before, state_after=after,
                actions=MappingProxyType(actions), step_count=self.step_count,
                winner=winner, terminated=terminated, truncated=truncated,
                termination_reason=reason,
            )
            rewards = self.reward_fn(transition)
            if not isinstance(rewards, Mapping) or set(rewards) != set(TEAMS):
                raise ValueError(f"reward_fn must return exactly {TEAMS}")
            if any(isinstance(r, (bool, np.bool_)) or not isinstance(r, Real)
                   or not np.isfinite(r) for r in rewards.values()):
                raise ValueError("reward_fn must return finite scalar rewards")
            rewards = {team: float(rewards[team]) for team in TEAMS}
        return (self.get_obs(), rewards, terminated, truncated,
                self._info(winner, reason))

    def _outcome(self):
        alive = [bool(self._units(team)) for team in TEAMS]
        if not any(alive):
            return None, "mutual_elimination"
        if not all(alive):
            return TEAMS[0] if alive[0] else TEAMS[1], "elimination"
        if not any(u.combat_type == CombatType.DAMAGE and u.damage > 0
                   for u in self.all_units.values()):
            return None, "no_combat_units"
        return None, None

    def _info(self, winner=None, reason=None):
        info = {"avail_actions": self.get_avail_actions(),
                "step_count": self.step_count, "winner": winner,
                "termination_reason": reason,
                "battle_won": {team: winner == team for team in TEAMS}}
        if self.record_combat:
            info["combat"] = {
                team: {
                    field: np.array([getattr(unit, field) for unit in roster],
                                    dtype=np.float32)
                    for field in ("hp", "max_hp", "hp_damage_dealt",
                                  "shield_damage_dealt")
                } for team, roster in self._episode_rosters.items()
            }
        return info

    def _can_move(self, unit, direction):
        if unit.max_velocity <= 0:
            return False
        x, y = unit.pos + direction.dx_dy * MOVE_AMOUNT / 2
        return (0 <= x < self.map_info.width and 0 <= y < self.map_info.height
                and self.map_info.terrain[int(y)][int(x)] == TerrainType.NORMAL)

    @staticmethod
    def _can_target(unit, target):
        if unit.hp <= 0 or target.hp <= 0 or unit is target:
            return False
        if target.plane not in unit.valid_targets:
            return False
        if (unit.combat_type == CombatType.HEALING
                and target.combat_type == CombatType.HEALING):
            return False
        return np.linalg.norm(target.pos - unit.pos) <= AGENT_TARGET_RANGE

    def get_avail_actions(self, team=None):
        self._require_reset()
        if team is None:
            return {name: self.get_avail_actions(name) for name in TEAMS}
        units = self._units(team)
        mask = np.zeros((self.team_sizes[team], self.n_actions_by_team[team]),
                        dtype=bool)
        for index in range(self.team_sizes[team]):
            unit = units.get(index)
            if unit is None:
                mask[index, 0] = True
                continue
            mask[index, 1] = True
            for direction in Direction:
                mask[index, 2 + direction.value] = self._can_move(unit, direction)
            targets = (units if unit.combat_type == CombatType.HEALING
                       else self._units(self._other(team)))
            for target_id, target in targets.items():
                mask[index, 6 + target_id] = self._can_target(unit, target)
        return mask

    def _type_features(self, unit):
        result = np.zeros(self.map_info.num_unit_types, dtype=np.float32)
        if len(result):
            result[self.map_info.unit_type_ids[unit.type]] = 1
        return result

    @staticmethod
    def _fraction(value, maximum):
        return value / maximum if maximum else 0.0

    def get_obs(self, team=None):
        self._require_reset()
        if team is None:
            return {name: self.get_obs(name) for name in TEAMS}
        units = self._units(team)
        other = self._other(team)
        masks = self.get_avail_actions(team)
        observations = np.zeros((self.team_sizes[team], self.obs_sizes[team]),
                                dtype=np.float32)
        for index, unit in units.items():
            obs = observations[index]
            obs[:4] = masks[index, 2:6]
            offset = 4
            for target_team in (other, team):
                for target_id in range(self.team_sizes[target_team]):
                    if target_team == team and target_id == index:
                        continue
                    target = self._units(target_team).get(target_id)
                    if target is not None:
                        delta = target.pos - unit.pos
                        distance = np.linalg.norm(delta)
                        if distance < AGENT_SIGHT_RANGE:
                            attackable = (unit.combat_type == CombatType.DAMAGE
                                          and self._can_target(unit, target))
                            obs[offset:offset + 6] = (
                                1.0 if target_team == team else float(attackable),
                                distance / AGENT_SIGHT_RANGE,
                                delta[0] / AGENT_SIGHT_RANGE,
                                delta[1] / AGENT_SIGHT_RANGE,
                                target.hp / target.max_hp,
                                self._fraction(target.shield, target.max_shield),
                            )
                            obs[offset + 6:offset + self.entity_feat_size] = (
                                self._type_features(target)
                            )
                    offset += self.entity_feat_size
            obs[offset:offset + 4] = (
                unit.hp / unit.max_hp, self._fraction(unit.shield, unit.max_shield),
                self._fraction(unit.energy, unit.max_energy),
                self._fraction(unit.cooldown, unit.max_cooldown),
            )
            obs[offset + 4:] = self._type_features(unit)
        return observations

    def get_obs_agent(self, team, agent_id):
        self._other(team)
        if (isinstance(agent_id, bool) or not isinstance(agent_id, Integral)
                or not 0 <= agent_id < self.team_sizes[team]):
            raise ValueError(f"Invalid agent_id for {team}: {agent_id!r}")
        return self.get_obs(team)[agent_id].copy()

    def get_state(self):
        """Global critic features, not a serializable simulator/RNG checkpoint."""
        self._require_reset()
        state = np.zeros(self.state_size, dtype=np.float32)
        for team in TEAMS:
            layout = self.state_layout[team]
            stride = len(self.unit_state_fields)
            for index, unit in self._units(team).items():
                offset = layout["offset"] + index * stride
                state[offset:offset + 7] = (
                    1.0, unit.hp / unit.max_hp,
                    self._fraction(unit.shield, unit.max_shield),
                    self._fraction(unit.cooldown, unit.max_cooldown),
                    self._fraction(unit.energy, unit.max_energy),
                    (unit.pos[0] - self.cx_cy[0]) / self.map_info.width,
                    (unit.pos[1] - self.cx_cy[1]) / self.map_info.height,
                )
                state[offset + 7:offset + stride] = self._type_features(unit)
            start = self.state_layout[f"{team}_last_actions"]["offset"]
            history = self.last_actions_by_team[team].ravel()
            state[start:start + len(history)] = history
        state[-1] = self.step_count / self.episode_limit
        return state

    def get_env_info(self):
        """Static schemas are available before reset; no privileged state."""
        return {"teams": TEAMS, "n_agents": dict(self.team_sizes),
                "agent_ids": {t: tuple(range(n)) for t, n in self.team_sizes.items()},
                "n_actions": dict(self.n_actions_by_team),
                "obs_shape": dict(self.obs_sizes), "state_shape": self.state_size,
                "episode_limit": self.episode_limit,
                "state_layout": deepcopy(self.state_layout),
                "observation_layout": {
                    "movement": ("north", "south", "east", "west"),
                    "opponents": ("attackable", "distance", "dx", "dy", "health",
                                  "shield", "unit_type_one_hot"),
                    "teammates": ("visible", "distance", "dx", "dy", "health",
                                  "shield", "unit_type_one_hot"),
                    "own": ("health", "shield", "energy", "cooldown",
                            "unit_type_one_hot"),
                }}
