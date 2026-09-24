# 双边 POSG 环境

`SMAClitePOSGEnv` 接收双方每个单位的动作，复用 SMAClite 的地图、兵种、
移动、碰撞、攻击和治疗机制。双方策略由调用方运行，环境不加载模型，
也不会替任何一方选择动作或自动切换攻击目标。

原来的 `SMACliteEnv` 和 `smaclite/<地图>-v0` 保留单边控制方式。

## 最小接入

```python
from smaclite.env import SMAClitePOSGEnv

env = SMAClitePOSGEnv(map_name="MMM2", episode_limit=200)
schema = env.get_env_info()  # reset 前就可以查询人数、维度、状态布局
obs, info = env.reset(seed=42)

# policy_0 / policy_1 在算法项目里实现。可逐单位调用，也可批量共享网络。
actions = {
    "team_0": policy_0(obs["team_0"], info["avail_actions"]["team_0"]),
    "team_1": policy_1(obs["team_1"], info["avail_actions"]["team_1"]),
}
obs, rewards, terminated, truncated, info = env.step(actions)

state = env.get_state()  # 可选，只用于全局评估或集中式训练
env.close()
```

也可以使用注册入口：

```python
import gymnasium as gym
import smaclite

env = gym.make("smaclite/MMM2-posg-v0", episode_limit=200)
schema = env.unwrapped.get_env_info()
```

这里的 `rewards` 是双队字典，并非 Gymnasium 的单个标量奖励，因此注册时
关闭了其单智能体检查器。返回五元组不代表可以直接交给单智能体训练器；
如需接入特定框架，应在算法项目添加适配层。本接口也不是 PettingZoo Parallel API。

自定义地图使用 `SMAClitePOSGEnv(map_file="path/to/map.json")` 或
`gym.make("smaclite/custom-posg-v0", map_file="path/to/map.json")`。
构造时 `map_name`、`map_info`、`map_file` 三选一。

## 单位与动作

- `team_0` 对应地图里的 `ALLY`，`team_1` 对应 `ENEMY`。
  人数、兵种、出生位置沿用地图，不假定两队相同。
- 每队 ID 为 `0 .. n_agents[team]-1`，按地图组和兵种顺序排列。
  阵亡不会重排 ID；每一步仍需提供双方所有槽位的动作。
- 动作 `0`：死亡单位的 no-op；`1`：停止；`2/3/4/5`：北/南/东/西移动。
  方向使用全局坐标，两队不镜像。
- 动作 `6 + target_id`：战斗单位攻击对方该 ID；治疗单位治疗本队该 ID。
  治疗单位不能治疗自身或其他治疗单位。治疗满血单位可以提交，但不会增加生命值。
- 各队动作维度见 `get_env_info()["n_actions"]`。同队使用统一的填充维度；
  无目标、不可攻击的兵种、超出目标选择范围或无效槽位由 mask 屏蔽。
- `get_avail_actions()` 返回两队布尔矩阵；`get_avail_actions("team_0")`
  只返回该队，形状为 `(人数, 动作维度)`。`reset/step` 的
  `info["avail_actions"]` 已包含下一次决策所需的 mask。
- 动作列表或一维整数 NumPy 数组均可。缺少一方、长度错误、非法索引、浮点或
  被 mask 禁止的动作都会报错；校验失败不会改变环境、历史动作或随机数状态。

双方动作来自同一时刻的观测，全部校验通过后才统一推进最多 8 个物理 tick，
中途若战斗结束则停止。底层保留原有随机单位更新顺序，相同 seed 和相同联合动作
可复现；“同时提交”不表示改写底层逐单位伤害结算机制。
攻击命令可以让单位接近选定目标，目标死亡后停止，不会自动寻找替代目标。

## 局部观测与全局状态

`get_obs()` 返回 `{team: float32数组}`，每队形状为 `(人数, obs_shape[team])`。
也可以调用 `get_obs(team)` 或 `get_obs_agent(team, agent_id)`。
算法应将每一行及对应 mask 传给相应单位的策略；环境返回整个队伍的数组，
不意味着单个单位获准查看队友的完整观测。

每个存活单位的观测依次包含：

1. 四个方向的移动可用性。
2. 按对方固定 ID 排列的可见对手特征：可攻击、距离、相对 x/y、生命、护盾、兵种。
3. 按本队固定 ID 排列、跳过自身的可见队友特征：可见、距离、相对 x/y、生命、护盾、兵种。
4. 自身生命、护盾、能量、冷却和兵种。

视野范围沿用 9，目标选择范围沿用 6；实际攻击射程由兵种决定。
视野外单位和死亡单位对应特征为零，自己的死亡观测也全零。
生命、护盾、能量、冷却以各自最大值归一化，距离与相对坐标除以视野范围，
兵种使用地图定义的 one-hot；单兵种地图的类型特征长度为零。
动作 mask 不利用视野外单位的信息。局部观测的完整字段说明可查询
`get_env_info()["observation_layout"]`。

`get_state()` 单独返回全局 float32 特征向量，不自动放入 `obs` 或 `info`。
它包含双方所有槽位的存活标记、生命、护盾、冷却、能量、归一化位置、兵种，
然后是双方上一轮动作的 one-hot 和 episode 进度。
两队状态使用相同字段，死亡槽位为零；布局及偏移见
`get_env_info()["state_layout"]`。所有返回数组均与环境内部存储分离。
这是供 critic/诊断使用的全局特征，不是用于精确恢复仿真的完整快照
（不包含随机数生成器、全部内部计时器及命令对象）。

## 奖励扩展

默认每一步（包括胜负结束）均返回 `{"team_0": 0.0, "team_1": 0.0}`。
环境不要求奖励零和，也不沿用旧版只计算我方伤害的分数。
使用 `reward_fn` 参数注入自定义函数：

```python
from smaclite.env import SMAClitePOSGEnv
from smaclite.env.rewards import RewardTransition

def reward_fn(transition: RewardTransition):
    # 这里按自己的任务定义两队奖励，不要求互为相反数。
    # 可使用 transition.state_before / state_after / actions 等。
    return {"team_0": 0.0, "team_1": 0.0}

env = SMAClitePOSGEnv(map_name="MMM2", reward_fn=reward_fn)
```

回调每个环境 step 调用一次，接收：前后全局特征、双方动作、步数、获胜队伍、
`terminated`、`truncated`、结束原因。状态为独立只读数组，动作是只读映射中的
整数元组；这些全局数据供奖励计算使用，不应转发给局部执行策略。
返回值必须恰好包含两队，每队一个有限标量。队内奖励分发或个体奖励包装可在
算法层处理。奖励回调在物理推进后执行；如果回调抛错或返回非法奖励，
该步的物理状态已经推进，调试后应重置 episode，不能重试同一步。

## 结束与运行

`terminated` 表示消灭一方、双方同时消灭或双方都没有可造成伤害的单位；
`truncated` 表示达到 `episode_limit`。战斗结束优先于同一步的时间限制。
`info["winner"]` 为获胜队伍或 `None`；`termination_reason` 为
`elimination`、`mutual_elimination`、`no_combat_units` 或 `time_limit`，
未结束时为 `None`。超时不根据剩余血量强行指定胜方。
episode 结束后再次 `step()` 会报错，需要先 `reset()`。

外部随机策略示例（不训练、不需要模型）：

```powershell
conda run -n mtapro python example_posg.py --map MMM2 --episodes 1 --steps 200
conda run -n mtapro python -m pytest -q tests
```

加上 `--render` 可以查看对战。可编辑安装下，另一个算法项目使用同一
`mtapro` 环境即可直接导入修改后的环境，无需复制源码。
