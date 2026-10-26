# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import ray
import gym
import numpy as np

# -----------------------------------------------------------------------------
# Ray remote worker actor -----------------------------------------------------
# -----------------------------------------------------------------------------

class WebshopWorker:
    """Ray remote actor that replaces the worker function.
    Each actor hosts a *WebAgentTextEnv* instance.
    """
    
    def __init__(self, seed, env_kwargs):
        # Lazy import avoids CUDA initialisation issues
        import sys
        import os
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), 'webshop'))
        sys.path.append(project_root)
        from web_agent_site.envs import WebAgentTextEnv  # noqa: WPS433 (runtime import)
        
        env_kwargs['seed'] = seed
        self.env = gym.make('WebAgentTextEnv-v0', **env_kwargs)
    
    def step(self, action):
        """Execute a step in the environment"""
        obs, reward, done, info = self.env.step(action)
        info = dict(info or {})  # make a *copy* so we can mutate safely
        info['available_actions'] = self.env.get_available_actions()
        from .progress_checks import inspect_visible_progress
        info['u_progress']=inspect_visible_progress(self.env)
        info['task_score'] = reward

        # Redefine reward. We only use rule-based reward - win for 10, lose for 0.
        if done and reward == 1.0:
            info['won'] = True
            reward = 10.0
        else:
            info['won'] = False
            reward = 0

        return obs, reward, done, info
    
    def reset(self, idx):
        """Reset the environment with given session index"""
        obs, info = self.env.reset(session=idx)
        info = dict(info or {})
        info['available_actions'] = self.env.get_available_actions()
        from .progress_checks import inspect_visible_progress
        info['u_progress']=inspect_visible_progress(self.env)
        info['won'] = False
        return obs, info
    
    def render(self, mode_for_render):
        """Render the environment"""
        rendered = self.env.render(mode=mode_for_render)
        return rendered
    
    def get_available_actions(self):
        """Get available actions"""
        return self.env.get_available_actions()
    
    def get_goals(self):
        """Get environment goals"""
        return self.env.server.goals
    
    def close(self):
        """Close the environment"""
        self.env.close()


class WebshopGroupWorker:
    """一个 Ray actor 里放 n_slots 个 env，**共享同一份 SimServer 与 JVM**。

    为什么需要它
    ------------
    上面的 WebshopWorker 是「一个 env 一个进程」，每个进程各自 load_products
    并起一个 JVM。实测边际占用 **530 MB/env**（开销主要在 pyserini/spacy/torch
    的 import，与商品数、索引大小无关）。而 GraphGPO 的 WebShop 脚本会同时建
    train(16×8=128) 和 val(128×1=128) 两套，共 256 个进程 ≈ **136 GB**，
    在 62 GB 的机器上根本起不来。共享之后 256 进程 → 32 个 actor，约 17 GB。

    共享为什么不改变结果
    --------------------
    · 训练：原代码 worker i 的 seed 是 ``seed + (i // group_n)``，
      **同一个 group 的 group_n 条 rollout 本来就同 seed**。让一个 actor
      正好装一个 group（n_slots == group_n），它们共用的 SimServer 与原来
      各自建的那份在 seed 上完全一致 —— goals 的 shuffle 顺序相同，
      ``reset(idx)`` 取到的目标也相同。**逐位等价。**
    · 验证：group_n == 1，各 env 原本 seed 不同。但 WebshopMultiProcessEnv.reset
      里的 idx 是**每次验证重新随机抽**的，验证集本来就不固定；共享后仍是
      「从同一池子里随机抽 env_num 个目标」，分布不变。

    唯一要小心的是会话串：SimServer.user_sessions 按会话字符串索引，
    共享 server 后同一个 idx 会撞车。用 session_prefix 隔离 —— 它只改会话 id，
    不影响 session_int（决定用哪个 goal），所以同组的轨迹仍落在同一个任务上。
    这套做法在 recipe/diag/sample_webshop.py 里已经跑过。
    """

    def __init__(self, seeds, env_kwargs, prefix):
        """seeds: 长度 = 本 actor 承载的 env 数，逐个给出每个 env 的 seed。

        **按 seed 分组建 SimServer**：seed 相同的 env 共用一份，seed 不同的各建一份。
        这样一个 actor 可以装下多个 group 而不改变任何语义 —— 省掉的是每进程
        530 MB 的 import 固定开销（pyserini/spacy/torch），不是 seed 语义。

        实测：一个 actor 装 1 个 group(8 env) 占 1.0 GB，其中固定开销 0.53 GB、
        server+8 env 占 0.47 GB。装 2 个 group 只要 1.47 GB 而不是 2.0 GB。
        """
        import sys
        import os as _os
        project_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), 'webshop'))
        sys.path.append(project_root)
        from web_agent_site.envs import WebAgentTextEnv  # noqa: F401,WPS433

        self.envs = []
        servers = {}                      # seed -> SimServer
        for slot, seed in enumerate(seeds):
            kwargs = dict(env_kwargs)
            kwargs['seed'] = seed
            env = gym.make('WebAgentTextEnv-v0', server=servers.get(seed),
                           session_prefix=f'{prefix}{slot}_', **kwargs)
            # SimServer 自己会 `random.seed(seed); random.shuffle(goals)`，
            # 所以同 seed 的那一份复用后 goals 顺序与原来逐位相同。
            servers.setdefault(seed, env.unwrapped.server)
            self.envs.append(env)

    # WebshopWorker.step 的奖励重定义逐字复刻，不能改 -------------------
    def _step_one(self, env, action):
        obs, reward, done, info = env.step(action)
        info = dict(info or {})
        info['available_actions'] = env.get_available_actions()
        from .progress_checks import inspect_visible_progress
        info['u_progress']=inspect_visible_progress(env,inspect_search_results=action.startswith('search['))
        info['task_score'] = reward

        if done and reward == 1.0:
            info['won'] = True
            reward = 10.0
        else:
            info['won'] = False
            reward = 0

        return obs, reward, done, info

    def step(self, actions: list):
        return [self._step_one(env, a) for env, a in zip(self.envs, actions)]

    def reset(self, idxs: list):
        out = []
        for env, idx in zip(self.envs, idxs):
            obs, info = env.reset(session=idx)
            info = dict(info or {})
            info['available_actions'] = env.get_available_actions()
            from .progress_checks import inspect_visible_progress
            info['u_progress']=inspect_visible_progress(env)
            info['won'] = False
            out.append((obs, info))
        return out

    def reset_slot(self, slot, idx):
        env = self.envs[slot]
        obs, info = env.reset(session=idx)
        info = dict(info or {})
        info.update(available_actions=env.get_available_actions(), won=False)
        from .progress_checks import inspect_visible_progress
        info['u_progress']=inspect_visible_progress(env)
        return obs, info

    def step_slot(self, slot, action):
        return self._step_one(self.envs[slot], action)

    def render(self, mode_for_render, slot=None):
        if slot is not None:
            return self.envs[slot].render(mode=mode_for_render)
        return [env.render(mode=mode_for_render) for env in self.envs]

    def get_goals(self):
        return self.envs[0].unwrapped.server.goals

    def close(self):
        for env in self.envs:
            env.close()


# -----------------------------------------------------------------------------
# Vectorised Ray environment --------------------------------------------------
# -----------------------------------------------------------------------------

class WebshopMultiProcessEnv(gym.Env):
    """A vectorised, Ray-based wrapper around *WebAgentTextEnv*.

    ``info`` dictionaries returned by :py:meth:`step` **and** :py:meth:`reset`
    automatically contain the key ``'available_actions'`` so downstream RL code
    can obtain the *legal* action set without extra IPC overhead.
    """
    def __init__(
        self,
        seed: int,
        env_num: int,
        group_n: int,
        resources_per_worker: dict,
        is_train: bool = True,
        env_kwargs: dict = None,
    ) -> None:
        super().__init__()

        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()

        self._seed = seed
        self.group_n = group_n
        self.env_num = env_num
        self.num_processes = env_num * group_n
        self.is_train = is_train
        if not is_train: assert group_n == 1

        self._rng = np.random.RandomState(seed)
        # Explicit opt-in: archived runs retain their original seed/index rules.
        self._task_schedule = None
        self._workers = []
        manifest = os.environ.get('IRGO_TASK_MANIFEST')
        if manifest:
            from recipe.GraphGPO.task_protocol import FixedTaskSchedule
            self._task_schedule = FixedTaskSchedule(manifest, is_train, env_num, seed)
            self._task_schedule.verify_sources(os.path.join(os.path.dirname(__file__), 'webshop'))
        all_seeds = [
            self._task_schedule.environment_seed if self._task_schedule else seed + i // group_n
            for i in range(self.num_processes)
        ]

        self._env_kwargs = env_kwargs if env_kwargs is not None else {'observation_mode': 'text', 'num_products': None}

        # -------------------------- Ray actors setup --------------------------
        # WEBSHOP_ENVS_PER_ACTOR：一个 actor 里塞几个 env（共享 SimServer/JVM）。
        # 不设或设成 1 时，下面走的是**完全原始**的一 env 一进程路径。
        #
        # 训练侧必须设成 group_n：一个 actor 正好装一个 group，组内本来就同 seed，
        # 这样才与原语义逐位等价（见 WebshopGroupWorker 的注释）。
        #
        # 验证侧 group_n == 1，没有这个约束，可以塞得更满。
        # 实测每个 actor 的固定开销约 530 MB（pyserini/spacy/torch 的 import），
        # 而共享 server 后每多一个 env 只加约 40 MB —— 所以 actor 数才是内存主项：
        #     128 env / 8  = 16 actor × 0.85 GB = 13.6 GB
        #     128 env / 32 =  4 actor × 1.80 GB =  7.2 GB   省 6.4 GB
        # 代价是同一 actor 内的 env 串行 step，而 WebShop 单步是毫秒级，
        # 相对 LLM 生成的耗时可以忽略。
        _key = 'WEBSHOP_ENVS_PER_ACTOR' if is_train else 'WEBSHOP_ENVS_PER_ACTOR_VAL'
        per_actor = int(os.environ.get(_key,
                        os.environ.get('WEBSHOP_ENVS_PER_ACTOR', '1')))
        if per_actor > 1 and self.num_processes % per_actor:
            print(f'[webshop] num_processes={self.num_processes} 不能被 '
                  f'WEBSHOP_ENVS_PER_ACTOR={per_actor} 整除，退回一 env 一进程')
            per_actor = 1
        self._per_actor = per_actor

        if per_actor == 1:
            env_worker = ray.remote(**resources_per_worker)(WebshopWorker)
            self._workers = [
                env_worker.remote(all_seeds[i], self._env_kwargs)
                for i in range(self.num_processes)
            ]
        else:
            # CPU 预留按 env 数放大，保持与原来相同的总量
            res = dict(resources_per_worker)
            if 'num_cpus' in res:
                res['num_cpus'] = res['num_cpus'] * per_actor
            env_worker = ray.remote(**res)(WebshopGroupWorker)
            n_actors = self.num_processes // per_actor
            # 每个 env 的 seed 与原始路径逐位一致：seed + (i // group_n)。
            # actor 内部再按 seed 分组建 SimServer，所以 per_actor 可以大于
            # group_n（一个 actor 装多个 group），语义不受影响。
            self._workers = [
                env_worker.remote(all_seeds[a * per_actor:(a + 1) * per_actor],
                                  self._env_kwargs, f'a{a}_')
                for a in range(n_actors)
            ]
            print(f'[webshop] {"train" if is_train else "val"}: '
                  f'{self.num_processes} 个 env → {n_actors} 个 actor '
                  f'(每个 {per_actor} 个，共享 SimServer)')

        # Get goals from the first worker
        goals_future = self._workers[0].get_goals.remote()
        goals = ray.get(goals_future)

        # ------- original ----------#
        # if args.num is None:
        #     if split == 'test':
        #         self.goal_idxs = range(500)
        #     elif split == 'eval':
        #         self.goal_idxs = range(500, 1500)
        #     elif split == 'train':
        #         self.goal_idxs = range(1500, len(self.env.server.goals))
        # else:
        #     self.goal_idxs = range(len(self.env.server.goals))

        if self._task_schedule:
            self._task_schedule.verify_goals(goals)
            self.goal_idxs = sorted(self._task_schedule.allowed_indices)
        elif not self.is_train:
            self.goal_idxs = range(500)
        else:
            self.goal_idxs = range(500, len(goals))
            
        print(f'[webshop] eligible goals: {len(self.goal_idxs)}; frozen protocol: {bool(self._task_schedule)}')

    # ------------------------------------------------------------------
    # Base API ----------------------------------------------------------
    # ------------------------------------------------------------------

    def step(self, actions: list[str]):
        if len(actions) != self.num_processes:
            raise ValueError(
                f'Expected {self.num_processes} actions, got {len(actions)}',
            )

        # Send step commands to all workers
        if self._per_actor == 1:
            futures = []
            for worker, action in zip(self._workers, actions):
                future = worker.step.remote(action)
                futures.append(future)
            results = ray.get(futures)
        else:
            e = self._per_actor
            futures = [w.step.remote(actions[a * e:(a + 1) * e])
                       for a, w in enumerate(self._workers)]
            results = [r for chunk in ray.get(futures) for r in chunk]

        # Collect results
        obs_list, reward_list, done_list, info_list = [], [], [], []
        for obs, reward, done, info in results:
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)

        return obs_list, reward_list, done_list, info_list

    def reset(self):
        idx = (self._task_schedule.next_indices() if self._task_schedule else
               self._rng.choice(self.goal_idxs, size=self.env_num, replace=False))
        if self._task_schedule:
            task_ids, trajectory_ids = self._task_schedule.identities(idx, self.group_n)
        idx = np.repeat(idx, self.group_n).tolist()

        # Send reset commands to all workers
        if self._per_actor == 1:
            futures = []
            for worker, i in zip(self._workers, idx):
                future = worker.reset.remote(i)
                futures.append(future)
            results = ray.get(futures)
        else:
            e = self._per_actor
            futures = [w.reset.remote(idx[a * e:(a + 1) * e])
                       for a, w in enumerate(self._workers)]
            results = [r for chunk in ray.get(futures) for r in chunk]

        # Collect results
        obs_list, info_list = [], []
        for slot, (obs, info) in enumerate(results):
            info = dict(info)
            info['u_goal_index'] = int(idx[slot])
            info['u_environment_seed'] = (self._task_schedule.environment_seed if self._task_schedule
                                           else self._seed + slot // self.group_n)
            if self._task_schedule:
                info['irgo_task_instance_id'] = task_ids[slot]
                info['irgo_trajectory_id'] = trajectory_ids[slot]
            obs_list.append(obs)
            info_list.append(info)

        return obs_list, info_list

    # ------------------------------------------------------------------
    # Convenience helpers ----------------------------------------------
    # ------------------------------------------------------------------

    def branch(self, slot, goal_index):
        if not self.is_train or goal_index not in self.goal_idxs:
            raise ValueError('Intervention must stay in the training split')
        if not 0 <= slot < self.num_processes:
            raise ValueError('Invalid intervention slot')
        return WebshopBranch(self, slot, goal_index)

    def render(self, mode: str = 'text', env_idx: int = None):
        if env_idx is not None:
            if self._per_actor == 1:
                future = self._workers[env_idx].render.remote(mode)
            else:
                e = self._per_actor
                future = self._workers[env_idx // e].render.remote(mode, env_idx % e)
            return ray.get(future)

        futures = []
        for worker in self._workers:
            future = worker.render.remote(mode)
            futures.append(future)

        results = ray.get(futures)
        if self._per_actor == 1:
            return results
        return [r for chunk in results for r in chunk]

    # ------------------------------------------------------------------
    # Clean‑up ----------------------------------------------------------
    # ------------------------------------------------------------------

    def close(self):
        if getattr(self, '_closed', False):
            return

        # Close all workers and kill Ray actors
        close_futures = []
        for worker in self._workers:
            future = worker.close.remote()
            close_futures.append(future)
        
        # Wait for all workers to close
        ray.get(close_futures)
        
        # Kill all Ray actors
        for worker in self._workers:
            ray.kill(worker)
            
        self._closed = True

    def __del__(self):  # noqa: D401
        self.close()


# -----------------------------------------------------------------------------
# Factory helper --------------------------------------------------------------
# -----------------------------------------------------------------------------

def build_webshop_envs(
    seed: int,
    env_num: int,
    group_n: int,
    resources_per_worker: dict,
    is_train: bool = True,
    env_kwargs: dict = None,
):
    """Mirror *build_sokoban_envs* so higher‑level code can swap seamlessly."""
    return WebshopMultiProcessEnv(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        resources_per_worker=resources_per_worker,
        is_train=is_train,
        env_kwargs=env_kwargs,
    )

class WebshopBranch:
    """One-slot view; reset/step never touches sibling environments or goal RNG."""
    num_processes = 1

    def __init__(self, parent, slot, goal_index):
        self.parent, self.slot, self.goal_index = parent, slot, goal_index

    def reset(self):
        p = self.parent
        if p._per_actor == 1:
            obs, info = ray.get(p._workers[self.slot].reset.remote(self.goal_index))
        else:
            worker = p._workers[self.slot // p._per_actor]
            obs, info = ray.get(worker.reset_slot.remote(self.slot % p._per_actor, self.goal_index))
        return [obs], [info]

    def step(self, actions):
        if len(actions) != 1:
            raise ValueError('Probe must contain exactly one physical environment')
        p = self.parent
        if p._per_actor == 1:
            row = ray.get(p._workers[self.slot].step.remote(actions[0]))
        else:
            worker = p._workers[self.slot // p._per_actor]
            row = ray.get(worker.step_slot.remote(self.slot % p._per_actor, actions[0]))
        return tuple([x] for x in row)
