# Optional combat telemetry

`SMAClitePOSGEnv(..., record_combat=True)` adds a `combat` mapping to reset/step
info. It is disabled by default and is privileged training/reward data, not a
local observation.

`info["combat"][team]` contains detached float32 arrays, ordered by stable local
unit ID (including dead slots):

- `hp`: current HP; zero after death.
- `max_hp`: maximum HP.
- `hp_damage_dealt`: cumulative actual HP removed by this unit's attacks.
- `shield_damage_dealt`: cumulative actual shield removed by its attacks.

Counters reset each episode. Compute step damage by subtracting consecutive
snapshots. Damage is measured before/after physical hits, respects armor and
remaining health, and excludes overkill and legacy kill bonuses. Area attacks
attribute every hit to their origin unit. Healing is not negative damage.
Retained roster references keep a dead attacker's cumulative contribution.

The legacy scalar reward and default POSG zero reward are unchanged. Reward
definitions and individual/team broadcasting belong to the algorithm project.
With regeneration/healing, cumulative damage may exceed initial HP; the
environment does not cap it or impose a reward scale.
