"""WebShop and DataProto adapter for the U controller; no optimizer in probes."""

import copy
import json
import hashlib
from pathlib import Path
import numpy as np
import torch
from .prob_u import UConfig, UController, digest, context_key, credit_key


def plain(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    return x


class TraceCapture:
    """记录当前策略真实执行的数据及原始 token，用于精确回放与信用核验。"""

    def __init__(self, config, gen_batch, envs, obs, infos, uids, tids, meta):
        self.max_steps = int(config.env.max_steps)
        self.credit_state_mode = config.algorithm.u.get("credit_state_mode", "page")
        self.gen_batch = gen_batch
        self.previous_infos = copy.deepcopy(infos)
        self.traces = []
        self.physical_calls = 0
        self.generated_tokens = 0
        self.generated_input_tokens = 0
        self.generated_requests = 0
        for i, info in enumerate(infos):
            if "u_goal_index" not in info or "u_environment_seed" not in info:
                raise ValueError("U requires reset identity metadata from WebShop")
            goal = digest(
                [envs.tasks[i], info["u_environment_seed"], meta["environment_version"]]
            )
            self.traces.append(
                dict(
                    id=str(tids[i]),
                    task_id=str(uids[i]),
                    goal_id=goal,
                    task_text=envs.tasks[i],
                    goal_index=int(info["u_goal_index"]),
                    environment_seed=int(info["u_environment_seed"]),
                    initial_anchor=obs["anchor"][i],
                    slot=i,
                    source="natural",
                    split="train",
                    max_steps=self.max_steps,
                    steps=[],
                    credit_start=0,
                    credit_state_mode=self.credit_state_mode,
                    **meta,
                )
            )

    def before(self, batch, step):
        self.generated_input_tokens += int(batch.batch["attention_mask"].sum().item())
        ids = plain(batch.non_tensor_batch["raw_prompt_ids"])
        self.prompts = ids
        keys = [
            context_key(
                t["goal_id"],
                ids[i],
                batch.non_tensor_batch["anchor_obs"][i],
                self.max_steps - step,
            )
            for i, t in enumerate(self.traces)
        ]
        credit_keys = [
            credit_key(
                {"key": key, "anchor": batch.non_tensor_batch["anchor_obs"][i]},
                self.credit_state_mode,
            )
            for i, key in enumerate(keys)
        ]
        # u_key remains the legacy exact-context field, never the new graph key.
        batch.non_tensor_batch["u_key"] = np.array(keys, dtype=object)
        batch.non_tensor_batch["u_context_key"] = np.array(keys, dtype=object)
        batch.non_tensor_batch["u_credit_key"] = np.array(credit_keys, dtype=object)
        batch.non_tensor_batch["u_step"] = np.full(len(keys), step + 1, dtype=np.int64)
        batch.non_tensor_batch["u_goal_id"] = np.array(
            [t["goal_id"] for t in self.traces], dtype=object
        )
        self.keys = keys
        self.credit_keys = credit_keys

    def after(
        self,
        step,
        obs,
        raw,
        actions,
        active,
        nxt,
        rewards,
        dones,
        infos,
        response_ids,
        response_mask,
    ):
        self.physical_calls += len(self.traces)
        self.generated_tokens += int(response_mask.sum().item())
        self.generated_requests += len(self.traces)
        for i, t in enumerate(self.traces):
            if not active[i]:
                continue
            tokens = plain(response_ids[i])
            mask = plain(response_mask[i])
            t["steps"].append(
                dict(
                    step=step + 1,
                    key=self.keys[i],
                    remaining=self.max_steps - step,
                    context_key=self.keys[i],
                    credit_key=self.credit_keys[i],
                    prompt_ids=self.prompts[i],
                    anchor=obs["anchor"][i],
                    raw_response=raw[i],
                    action=actions[i],
                    next_anchor=nxt["anchor"][i],
                    reward=float(rewards[i]),
                    done=bool(dones[i]),
                    won=bool(infos[i]["won"]),
                    valid=bool(np.asarray(infos[i]["is_action_valid"]).item()),
                    score=float(infos[i].get("task_score", 0.0)),
                    forced=False,
                    response_ids=[int(v) for v, m in zip(tokens, mask) if m],
                    clickables_before=list(
                        self.previous_infos[i]
                        .get("available_actions", {})
                        .get("clickables", [])
                    ),
                    checked_before=self.previous_infos[i].get("u_progress", {}),
                    checked_after=infos[i].get("u_progress", {}),
                )
            )
        self.previous_infos = copy.deepcopy(infos)

    def finish(self):
        for t in self.traces:
            s = t["steps"][-1]
            t.update(
                success=bool(s["won"]),
                score=s["score"],
                stop_reason=(
                    "success"
                    if s["won"]
                    else "terminal_failure" if s["done"] else "budget_exhausted"
                ),
            )


class ProbeRunner:
    """恢复同一任务前缀，只替换分支首动作，后续由当前策略自然生成。

    信用图可以按页面合并；回放必须保留完整输入与剩余步数，不能将
    一个相似页面误当成相同的可执行现场。前缀不重复提供信用证据。
    """

    def __init__(self, collector, actor, envs):
        self.collector = collector
        self.actor = actor
        self.envs = envs

    def propose_queries(self, natural, pre, cfg, stage):
        from .full_runtime import propose_queries

        return propose_queries(self, natural, pre, cfg, stage)

    def __call__(self, ref, branch, action, probe_id):
        from .graphgpo_env_manager import WebshopEnvironmentManager
        from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

        vector = self.envs.envs.branch(ref["slot"], ref["goal_index"])
        manager = WebshopEnvironmentManager(
            vector, self.envs.projection_f, self.collector.config
        )
        # preprocess uses the observation to build the prompt. Keep the original
        # slot's dataset metadata as well, rather than substituting another task.
        gen = self.collector.u_capture.gen_batch[ref["slot"] : ref["slot"] + 1]
        obs, infos = manager.reset(kwargs=None)
        if obs["anchor"][0] != ref["initial_anchor"]:
            raise ValueError("Probe initial task mismatch")
        t = {k: copy.deepcopy(v) for k, v in ref.items() if k != "steps"}
        t.update(
            id=probe_id,
            source="probe",
            steps=[],
            credit_start=branch["step"] - 1,
            parent_trajectory_id=ref["id"],
        )
        total_calls = 0
        generated_tokens = 0
        generated_input_tokens = 0
        generated_requests = 0
        for step in range(1, ref["max_steps"] + 1):
            batch = self.collector.preprocess_batch(gen, obs)
            ids = plain(batch.non_tensor_batch["raw_prompt_ids"][0])
            key = context_key(
                ref["goal_id"], ids, obs["anchor"][0], ref["max_steps"] - step + 1
            )
            seed = None
            response_ids = []
            if step < branch["step"]:
                expected = ref["steps"][step - 1]
                if key != expected["key"]:
                    raise ValueError("Probe prefix context mismatch")
                raw = expected["raw_response"]
                mode = "prefix"
            elif step == branch["step"]:
                if key != branch["key"] or ids != branch["prompt_ids"]:
                    raise ValueError("Probe branch full-context mismatch")
                raw = "<think></think><action>" + action + "</action>"
                mode = "forced"
                response_ids = self.collector.tokenizer.encode(
                    raw, add_special_tokens=False
                ) + [self.collector.tokenizer.eos_token_id]
            else:
                inp = batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt_ids"],
                )
                seed = int(digest([probe_id, step])[:8], 16) % (2**31 - 1)
                inp.meta_info = {
                    **gen.meta_info,
                    "validate": False,
                    "do_sample": True,
                    "u_sampling_seed": seed,
                }
                padded, n = pad_dataproto_to_divisor(inp, self.actor.world_size)
                generated_input_tokens += int(
                    padded.batch["attention_mask"].sum().item()
                )
                padded_result = self.actor.generate_sequences(padded)
                width = padded_result.batch["responses"].shape[1]
                generated_tokens += int(
                    padded_result.batch["attention_mask"][:, -width:].sum().item()
                )
                generated_requests += len(padded_result)
                result = unpad_dataproto(padded_result, pad_size=n)
                responses = result.batch["responses"]
                raw = self.collector.tokenizer.batch_decode(
                    responses, skip_special_tokens=True
                )[0]
                mask = result.batch["attention_mask"][0, -responses.shape[1] :]
                response_ids = plain(responses[0][mask.bool()])
                mode = "continuation"
            raw_list = [raw]
            projected, valid = self.envs.projection_f(raw_list.copy())
            if mode == "forced" and (not valid[0] or projected[0] != action):
                raise ValueError("Forced candidate projection mismatch")
            nxt, rewards, dones, infos = manager.step(raw_list.copy())
            total_calls += 1
            s = dict(
                step=step,
                key=key,
                prompt_ids=ids,
                anchor=obs["anchor"][0],
                remaining=ref["max_steps"] - step + 1,
                raw_response=raw,
                action=projected[0],
                next_anchor=nxt["anchor"][0],
                reward=float(np.asarray(rewards).reshape(-1)[0]),
                done=bool(np.asarray(dones).reshape(-1)[0]),
                won=bool(infos[0]["won"]),
                valid=bool(np.asarray(infos[0]["is_action_valid"]).item()),
                score=float(infos[0].get("task_score", 0.0)),
                forced=mode == "forced",
                mode=mode,
                sampling_seed=seed,
                response_ids=response_ids,
            )
            s["context_key"] = key
            s["credit_key"] = credit_key(
                s, self.collector.config.algorithm.u.get("credit_state_mode", "page")
            )
            t["steps"].append(s)
            if s["done"]:
                if step < branch["step"]:
                    raise ValueError("Probe ended before intervention")
                break
            obs = nxt
        s = t["steps"][-1]
        t.update(
            success=s["won"],
            score=s["score"],
            generated_tokens=generated_tokens,
            generated_input_tokens=generated_input_tokens,
            generated_requests=generated_requests,
            stop_reason=(
                "success"
                if s["won"]
                else "terminal_failure" if s["done"] else "budget_exhausted"
            ),
        )
        return t, total_calls


class UTrainingBridge:
    """对接 GraphGPO 的真实训练批次，保留自然 PPO 与辅助 NLL 的来源区别。"""

    def __init__(self, config, collector, actor, envs):
        from omegaconf import OmegaConf

        settings = OmegaConf.to_container(config.algorithm.u, resolve=True)
        if not settings.pop("enabled"):
            raise ValueError("U bridge requested while disabled")
        self.environment_version = settings.pop("environment_version")
        if not self.environment_version or self.environment_version == "REQUIRED":
            raise ValueError(
                "Set U environment_version to the verified data/code identity"
            )
        if (
            config.algorithm.adv_estimator != "prob"
            or config.algorithm.filter_groups.enable
        ):
            raise ValueError("U requires Prob and forbids dynamic repeat-sampling")
        if int(config.actor_rollout_ref.rollout.n) != 1:
            raise ValueError("U requires one response per generation request")
        if config.env.env_name != "webshop/WebAgentTextEnv":
            raise ValueError("U currently supports WebShop only")
        if int(config.algorithm.prob.horizon) != int(settings["horizon"]):
            raise ValueError("Prob/U horizon mismatch")
        self.collector = collector
        self.actor = actor
        self.envs = envs
        self.output = Path(config.trainer.default_local_dir) / "u_evidence"
        self.controller = UController(UConfig(**settings), self.output)
        self.config = config
        if int(config.trainer.critic_warmup) > 1:
            raise ValueError("U requires an actor update in every stage")
        manifest = {
            "config": OmegaConf.to_container(config, resolve=True),
            "implementation": "prob-u-full150-page-checked-budget-success-learning",
            "base_commit": "a48589b1fb3fab23d45342064306a112dc10744e",
            "source_sha256": {
                name: hashlib.sha256(
                    (Path(__file__).parent / name).read_bytes()
                ).hexdigest()
                for name in (
                    "prob_u.py",
                    "prob_u_runtime.py",
                    "graphgpo_ray_trainer.py",
                    "rollout_loop.py",
                )
            },
        }
        if not (self.output / "run_manifest.json").exists():
            self.controller.write(self.output / "run_manifest.json", manifest)
        self.last_metrics = {}

    def start(self, stage):
        self.stage = stage
        self.collector.u_meta = dict(
            stage=stage,
            policy_version=f"pre_update_{stage}",
            environment_version=self.environment_version,
        )

    def process(self, batch):
        keys = list(
            zip(
                batch.non_tensor_batch["traj_uid"],
                map(int, batch.non_tensor_batch["u_step"]),
            )
        )
        steps = {
            (t["id"], s["step"]): s
            for t in self.collector.u_capture.traces
            for s in t["steps"]
        }
        expected = [
            credit_key(steps[k], self.controller.cfg.credit_state_mode) for k in keys
        ]
        if list(batch.non_tensor_batch.get("u_credit_key", [])) != expected:
            raise ValueError("U graph and normalization credit keys disagree")
        self.before, self.after, metrics = self.controller.process(
            self.collector.u_capture.traces,
            ProbeRunner(self.collector, self.actor, self.envs),
            self.stage,
        )
        self.last_metrics = metrics
        self.last_metrics["environment/u_natural_physical_calls"] = (
            self.collector.u_capture.physical_calls
        )
        self.last_metrics["u/natural_generated_tokens"] = (
            self.collector.u_capture.generated_tokens
        )
        self.last_metrics["u/natural_input_tokens"] = (
            self.collector.u_capture.generated_input_tokens
        )
        self.last_metrics["environment/u_natural_effective_steps"] = sum(
            len(t["steps"]) for t in self.collector.u_capture.traces
        )
        self.last_metrics["u/natural_generated_requests"] = (
            self.collector.u_capture.generated_requests
        )
        before = torch.tensor(
            [self.before[k]["a"] for k in keys],
            device=batch.batch["responses"].device,
            dtype=torch.float32,
        )
        after = torch.tensor(
            [self.after[k]["a"] for k in keys],
            device=before.device,
            dtype=torch.float32,
        )
        self.raw_delta = after - before
        batch.batch["step_rewards"] = after
        self.last_metrics["history/observer_active"] = 1
        self.last_metrics["environment/u_total_physical_calls"] = (
            self.collector.u_capture.physical_calls
            + metrics["u/extra_physical_env_calls"]
        )
        prior = [
            json.loads(p.read_text())
            for p in self.output.glob("stage_*/metrics.json")
            if int(p.parent.name.split("_")[1]) < self.stage
            and (p.parent / "complete.json").exists()
        ]
        self.last_metrics["environment/u_cumulative_physical_calls"] = (
            self.last_metrics["environment/u_total_physical_calls"]
            + sum(m.get("environment/u_total_physical_calls", 0) for m in prior)
        )
        return metrics

    def compose_observation(self, batch, compose):
        """Hold all penalties/episode terms fixed; change only raw step credit."""
        from verl import DataProto

        # DataProto deepcopy invokes its serializer and can lock the original
        # TensorDict. Clone tensors explicitly to keep the live batch writable.
        base = DataProto(
            batch=batch.batch.clone(),
            non_tensor_batch=copy.deepcopy(batch.non_tensor_batch),
            meta_info=copy.deepcopy(batch.meta_info),
        )
        base.batch["step_rewards"] = base.batch["step_rewards"] - self.raw_delta
        base = compose(base)
        batch = compose(batch)
        mask = batch.batch["response_mask"].bool()
        delta = (batch.batch["advantages"] - base.batch["advantages"])[mask]
        self.last_metrics.update(
            {
                "prob/final_token_advantage_sign_changed_ratio": (
                    float(
                        (
                            (
                                batch.batch["advantages"].sign()
                                != base.batch["advantages"].sign()
                            )[mask]
                        )
                        .float()
                        .mean()
                    )
                    if delta.numel()
                    else 0.0
                ),
                "prob/final_token_advantage_changed_ratio": (
                    float((delta.abs() > 1e-8).float().mean()) if delta.numel() else 0.0
                ),
                "prob/final_token_advantage_abs_delta_mean": (
                    float(delta.abs().mean()) if delta.numel() else 0.0
                ),
            }
        )
        from .prob_u import failed_trajectory_ids

        failed_ids = failed_trajectory_ids(self.collector.u_capture.traces, limit=8)
        failed_rows = torch.tensor(
            [str(t) in failed_ids for t in batch.non_tensor_batch["traj_uid"]],
            device=mask.device,
        )
        self.last_metrics.update(
            {
                "prob/final_token_advantage_changed_gt_001": (
                    float((delta.abs() > 0.01).float().mean()) if delta.numel() else 0.0
                ),
                "tasks/all_failed_natural_steps": int(failed_rows.sum()),
                "tasks/all_failed_nonzero_final_advantage_steps": int(
                    ((batch.batch["advantages"].abs() > 1e-8) & mask)
                    .any(-1)[failed_rows]
                    .sum()
                ),
            }
        )
        folder = self.controller.pending
        if not torch.isfinite(batch.batch["advantages"]).all():
            raise ValueError("Nonfinite U advantages")
        # Save a keyed full snapshot before batch replication or balancing.
        torch.save(
            {
                "trajectory_ids": list(batch.non_tensor_batch["traj_uid"]),
                "step_ids": list(map(int, batch.non_tensor_batch["u_step"])),
                "mask": batch.batch["response_mask"].cpu(),
                "before": base.batch["advantages"].cpu(),
                "after": batch.batch["advantages"].cpu(),
            },
            folder / "final_advantages.pt",
        )
        self.controller.write(folder / "metrics.json", self.last_metrics)
        return batch

    def attach_auxiliary(self, batch):
        from .full_runtime import attach_auxiliary

        epochs = int(self.config.actor_rollout_ref.actor.ppo_epochs)
        self.last_metrics.update(
            {
                "cost/ppo_rows_over_epochs": len(batch) * epochs,
                "cost/ppo_input_output_tokens_over_epochs": int(
                    batch.batch["attention_mask"].sum().item()
                )
                * epochs,
                "cost/ppo_response_tokens_over_epochs": int(
                    batch.batch["response_mask"].sum().item()
                )
                * epochs,
            }
        )
        attach_auxiliary(self, batch)

    def save_actor_input(self, batch):
        # Exact post-padding/post-balancing batch and old policy probabilities.
        natural_ids = {t["id"] for t in self.collector.u_capture.traces}
        if any(t not in natural_ids for t in batch.non_tensor_batch["traj_uid"]):
            raise ValueError("Non-natural trajectory reached actor update")
        torch.save(
            {
                "tensors": {k: v.detach().cpu() for k, v in batch.batch.items()},
                "non_tensor": plain(batch.non_tensor_batch),
            },
            self.controller.pending / "actor_input.pt",
        )

    def after_update(self, metrics):
        from .full_runtime import compact_stage_evidence

        folder = self.controller.pending
        compact_stage_evidence(folder)
        self.controller.finish(
            {
                "actor_update_completed": True,
                "stage": self.stage,
                "metrics": plain(metrics),
            }
        )
        self.collector.u_meta = None

    def log_tables(self, logger, stage, metrics):
        folder = self.output / f"stage_{stage:06d}"
        self.controller.write(folder / "logged_metrics.json", plain(metrics))
        backend = logger.logger.get("wandb")
        if backend is None:
            return
        folder = self.output / f"stage_{stage:06d}"
        rows = json.loads((folder / "tasks.json").read_text())
        columns = ["task_id", "goal_id", "trajectories", "successes", "probes"]
        shadow = json.loads((folder / "history_shadow.json").read_text())
        shadow_cols = [
            "goal_id",
            "trajectory_id",
            "action",
            "current_support",
            "history_support",
            "current_prediction",
            "history_prediction",
            "current_brier",
            "history_brier",
            "gain",
            "stale_policy_count",
        ]
        payload = {
            "u/tasks": backend.Table(
                columns=columns, data=[[r[c] for c in columns] for r in rows]
            )
        }
        if shadow:
            payload["history/shadow_examples"] = backend.Table(
                columns=shadow_cols,
                data=[[r[c] for c in shadow_cols] for r in shadow[:256]],
            )
        backend.log(payload, step=stage, commit=False)
