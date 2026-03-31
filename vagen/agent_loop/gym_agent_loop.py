# Copyright 2025 Bytedance Ltd.
# Licensed under the Apache License, Version 2.0

import asyncio
import json
import logging
import os
import re
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from PIL import Image
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from ..envs.gym_image_env import GymImageEnv
from omegaconf import OmegaConf
import traceback
import importlib
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

def _flatten_text_only_content(msg):
    """
    convert message['content'] from multimodal list to plain text
    - only allow type == 'text'
    - concatenate multiple text blocks in order
    """
    content = msg.get("content")

    if isinstance(content, str):
        return msg

    if not isinstance(content, list):
        raise TypeError(f"Unsupported content type: {type(content)}")

    texts = []
    for block in content:
        if not isinstance(block, dict):
            raise TypeError(f"Invalid content block: {block}")

        block_type = block.get("type")
        if block_type != "text":
            raise AssertionError(
                f"Non-text block found in text-only tokenizer path: {block_type}"
            )
        texts.append(block.get("text", ""))

    new_msg = dict(msg)
    new_msg["content"] = "".join(texts)
    return new_msg


class AgentState(Enum):
    PENDING = "pending"
    GENERATING = "generating"
    INTERACTING = "interacting"
    TERMINATED = "terminated"


class AgentData:
    """Container for all mutable trajectory state."""
    def __init__(
        self,
        messages: List[Dict[str, Any]],
        image_data: List[Image.Image],
        metrics: Dict[str, Any],
        request_id: str,
        env: GymImageEnv,
        response_limit: int,
        env_name: str,
    ):
        self.messages = messages
        self.image_data = image_data
        self.metrics = metrics
        self.request_id = request_id
        self.env = env
        self.response_limit = response_limit
        self.env_name = env_name

        # Token buffers
        self.prompt_ids: List[int] = []
        self.response_ids: List[int] = []
        self.response_mask: List[int] = []
        self.response_logprobs: List[float] = []

        # Env stats
        self.env_rewards: List[float] = []
        self.traj_success: bool = False
        self.env_turns: int = 0
        self.graph_states: List[Dict[str, Any]] = []
        # Reward component accumulators
        self.total_step_penalty: float = 0.0
        self.total_invalid_penalty: float = 0.0
        self.last_info: Dict[str, Any] = {}

        # Cached assistant text to step env
        self.last_assistant_text: Optional[str] = None


# -------------------- MM helpers --------------------

def _normalize_images(imgs: List[Image.Image]) -> List[Image.Image]:
    """Ensure PIL RGB and drop Nones."""
    out: List[Image.Image] = []
    for im in imgs or []:
        if im is None:
            continue
        out.append(im.convert("RGB") if isinstance(im, Image.Image) else im)
    return out


def _get_forbidden_vision_token_ids(tokenizer, processor) -> set[int]:
    """Resolve multimodal control token ids that must never appear in assistant output."""
    token_names = (
        "<|vision_start|>",
        "<|vision_end|>",
        "<|image_pad|>",
        "<|video_pad|>",
    )
    resolved_ids: set[int] = set()

    candidates = [processor, getattr(processor, "tokenizer", None), tokenizer]
    for candidate in candidates:
        if candidate is None or not hasattr(candidate, "convert_tokens_to_ids"):
            continue
        for token_name in token_names:
            token_id = candidate.convert_tokens_to_ids(token_name)
            if isinstance(token_id, int) and token_id >= 0:
                unk_id = getattr(candidate, "unk_token_id", None)
                if unk_id is None or token_id != unk_id:
                    resolved_ids.add(token_id)
    return resolved_ids


def _strip_forbidden_vision_tokens(
    token_ids: List[int],
    forbidden_token_ids: set[int],
    tokenizer,
    log_probs: Optional[List[float]] = None,
) -> tuple[List[int], Optional[List[float]], List[str]]:
    """Remove multimodal control tokens from generated assistant responses."""
    if not token_ids or not forbidden_token_ids:
        return list(token_ids), list(log_probs) if log_probs is not None else None, []

    sanitized_ids: List[int] = []
    sanitized_log_probs: Optional[List[float]] = [] if log_probs is not None else None
    removed_token_names: List[str] = []

    for idx, token_id in enumerate(token_ids):
        if token_id in forbidden_token_ids:
            removed_token_names.append(str(tokenizer.convert_ids_to_tokens(token_id)))
            continue
        sanitized_ids.append(token_id)
        if sanitized_log_probs is not None:
            sanitized_log_probs.append(log_probs[idx])

    return sanitized_ids, sanitized_log_probs, removed_token_names


def extract_success(info: Dict[str, Any], success_keys: str = "success|is_success") -> bool:
    """Extract success flag from env info dict."""
    for key in success_keys.split("|"):
        if key in info:
            return bool(info[key])
    return False

def convert_obs_to_content(
    obs: Dict[str, Any],
    obs_text_key: str = "obs_str",
    image_placeholder: str = "<image>",
    video_placeholder: str = "<video>",
    multi_modal_key: str = "multi_modal_input",
    **kwargs,
) -> List[Dict[str, Any]]:
    """Convert obs['obs_str'] containing <image>/<video> into structured content."""
    text = obs[obs_text_key]
    mmi = obs.get(multi_modal_key, {}) or {}

    # Simple strict consistency check
    num_img_tok = text.count(image_placeholder)
    num_vid_tok = text.count(video_placeholder)
    num_imgs = len(mmi.get(image_placeholder, []) or [])
    num_vids = len(mmi.get(video_placeholder, []) or [])
    assert num_img_tok == num_imgs, f"#images ({num_imgs}) != #{image_placeholder} ({num_img_tok})"
    assert num_vid_tok == num_vids, f"#videos ({num_vids}) != #{video_placeholder} ({num_vid_tok})"

    # Split and keep tokens
    pattern = f"({re.escape(image_placeholder)}|{re.escape(video_placeholder)})"
    segments = re.split(pattern, text)

    content: List[Dict[str, Any]] = []
    for seg in segments:
        if not seg:
            continue
        if seg == image_placeholder:
            content.append({"type": "image"})
        elif seg == video_placeholder:
            content.append({"type": "video"})
        else:
            content.append({"type": "text", "text": seg})
    return content


# -------------------- Gym Agent Loop --------------------

class GymAgentLoop(AgentLoopBase):
    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        print("Performing class-level GymAgentLoop initialization")

        cls.tokenizer = tokenizer
        cls.processor = processor
        cls.multi_turn_cfg = config.actor_rollout_ref.rollout.multi_turn
        
        # Store module paths for lazy loading; environments are imported on first use
        cls.env_registry_paths = dict(config.env_registry.items())
        cls.env_registry = {}
            
        cls.apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})
        cls.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        cls.response_length = config.actor_rollout_ref.rollout.response_length
        cls.forbidden_response_token_ids = _get_forbidden_vision_token_ids(tokenizer, processor)

        _placeholder = [{"role": "system", "content": "placeholder"}]
        if processor is not None:
            _prefix_text = processor.apply_chat_template(
                _placeholder, add_generation_prompt=False, tokenize=False, **cls.apply_chat_template_kwargs
            )
            cls.system_prompt_prefix = processor(text=[_prefix_text], return_tensors="pt")["input_ids"].squeeze(0).tolist()
        else:
            cls.system_prompt_prefix = tokenizer.apply_chat_template(
                _placeholder, add_generation_prompt=False, tokenize=True, return_dict=False, **cls.apply_chat_template_kwargs
            )

    @rollout_trace_op
    async def run(self, sampling_params: Dict[str, Any], **kwargs) -> AgentLoopOutput:
        metrics: Dict[str, Any] = {}
        request_id = uuid4().hex

        # Build env (lazy import on first use)
        env_name = kwargs["env_name"]
        if env_name not in self.env_registry:
            if env_name not in self.env_registry_paths:
                raise KeyError(f"Unknown env: {env_name}. Available: {list(self.env_registry_paths.keys())}")
            module_path, class_name = self.env_registry_paths[env_name].rsplit(".", 1)
            module = importlib.import_module(module_path)
            self.env_registry[env_name] = getattr(module, class_name)
        env_cls = self.env_registry[env_name]
        env_config = kwargs["config"]
        seed = kwargs["seed"]
        self.env_max_turns = kwargs.get("max_turns", None)
        env: GymImageEnv = env_cls(env_config=env_config)

        # Bootstrap: reset -> system_prompt (message order: system, then initial user)
        init_obs, info = await env.reset(seed=seed)
        sys_obs = await env.system_prompt()

        messages: List[Dict[str, Any]] = []
        image_data: List[Image.Image] = []

        if sys_obs:
            messages.append({"role": "system", "content": convert_obs_to_content(sys_obs, **kwargs)})
            sys_imgs = sys_obs.get("multi_modal_input", {}).get("<image>", []) or []
            image_data.extend(_normalize_images(sys_imgs))
        if init_obs:
            messages.append({"role": "user", "content": convert_obs_to_content(init_obs, **kwargs)})
            init_imgs = init_obs.get("multi_modal_input", {}).get("<image>", []) or []
            image_data.extend(_normalize_images(init_imgs))

        per_turn_response_limit = int(kwargs.get("response_length_per_turn") or self.response_length)
        per_turn_response_limit = min(per_turn_response_limit, self.response_length)
        if per_turn_response_limit <= 0:
            per_turn_response_limit = 1

        agent_data = AgentData(
            messages=messages,
            image_data=image_data,
            metrics=metrics,
            request_id=request_id,
            env=env,
            response_limit=per_turn_response_limit,
            env_name=kwargs["env_name"],
        )
        init_graph_state = info.get("graph_state") if isinstance(info, dict) else None
        if init_graph_state is not None:
            agent_data.graph_states.append(init_graph_state)

        # State machine: always GENERATE -> INTERACT, and decide termination inside INTERACT
        state = AgentState.PENDING
        while state != AgentState.TERMINATED:
            if state == AgentState.PENDING:
                state = await self._handle_pending_state(agent_data, sampling_params)
            elif state == AgentState.GENERATING:
                state = await self._handle_generating_state(agent_data, sampling_params)
            elif state == AgentState.INTERACTING:
                state = await self._handle_env_state(agent_data, **kwargs)
            else:
                logger.error(f"Invalid state: {state}")
                state = AgentState.TERMINATED

        # Close env after loop
        await env.close()

        # Finalize output
        resp_len = len(agent_data.response_mask)
        response_ids = agent_data.prompt_ids[-resp_len:] if resp_len else []
        prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - resp_len]
        multi_modal_data = {"image": agent_data.image_data} if agent_data.image_data else {}

        if len(prompt_ids) > self.prompt_length:
            logger.warning(
                f"In env:{agent_data.env_name}, prompt_ids length {len(prompt_ids)} exceeds prompt_length {self.prompt_length}",
            )
        if len(response_ids) > self.response_length:
            logger.warning(
                f"In env:{agent_data.env_name}, response_ids length {len(response_ids)} exceeds response_length {self.response_length}",
            )

        output = AgentLoopOutput(
            prompt_ids=prompt_ids[-self.prompt_length:],
            response_ids=response_ids[: self.response_length],
            response_mask=agent_data.response_mask[: self.response_length],
            multi_modal_data=multi_modal_data,
            response_logprobs=(
                agent_data.response_logprobs[: self.response_length] if agent_data.response_logprobs else None
            ),
            reward_score=sum(agent_data.env_rewards) if agent_data.env_rewards else 0.0,
            num_turns=agent_data.env_turns,
            metrics=agent_data.metrics,
            extra_fields={
                "image_data": agent_data.image_data,
                "reward_extra_info": {
                    "traj_success": float(agent_data.traj_success),
                    "graph_states": json.dumps(list(agent_data.graph_states)),
                    "step_penalty": agent_data.total_step_penalty,
                    "invalid_penalty": agent_data.total_invalid_penalty,
                    "cogmap_reward": float(agent_data.last_info.get("cogmap_score", 0.0)) * 10.0,
                    "cogmap_dir": float(agent_data.last_info.get("cogmap_dir", 0.0)),
                    "cogmap_facing": float(agent_data.last_info.get("cogmap_facing", 0.0)),
                    "cogmap_pos": float(agent_data.last_info.get("cogmap_pos", 0.0)),
                    "exp_coverage": float(agent_data.last_info.get("cogmap_exploration_coverage", 0.0)),
                    "eval_task_reward": float(agent_data.last_info.get("eval_task_reward", 0.0)),
                    "eval_task_mean": float(agent_data.last_info.get("eval_task_mean", 0.0)),
                    # Per-task eval scores (eval_dir, eval_rot, etc.) forwarded from env info
                    **{k: float(v) for k, v in agent_data.last_info.items()
                       if k.startswith("eval_") and k not in ("eval_task_reward", "eval_task_scores", "eval_task_mean")},
                },
            },
        )
        return output

    async def _handle_pending_state(self, agent_data: AgentData, sampling_params: Dict[str, Any]) -> AgentState:
        """Encode initial (system + first user) messages into prompt_ids."""
        if self.processor is not None:
            raw_prompt = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    agent_data.messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            model_inputs = self.processor(text=[raw_prompt], images=agent_data.image_data or None, return_tensors="pt")
            agent_data.prompt_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            if agent_data.image_data:
                raise ValueError("Environment returned images but `processor` is None.")

            flat_messages = [_flatten_text_only_content(msg) for msg in agent_data.messages]
            agent_data.prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    flat_messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
        
        if len(agent_data.prompt_ids)>self.prompt_length:
            logger.warning(f"In env:{agent_data.env_name}, initial prompt length {len(agent_data.prompt_ids)} exceeds prompt_length {self.prompt_length}")
        return AgentState.GENERATING

    async def _handle_generating_state(
        self, agent_data: AgentData, sampling_params: Dict[str, Any]
    ) -> AgentState:
        """Generate assistant output and mark generated tokens with mask=1."""
        sampling_params_for_turn = sampling_params.copy()
        max_new_tokens=sampling_params_for_turn.get("max_new_tokens", None) or agent_data.response_limit
        max_new_tokens = min(max_new_tokens, agent_data.response_limit)
        sampling_params_for_turn["max_new_tokens"] = max_new_tokens
            

        with simple_timer("generate_sequences", agent_data.metrics):
            output = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,
                sampling_params=sampling_params_for_turn,
                image_data=agent_data.image_data,
            )

        response_ids, response_logprobs, removed_tokens = _strip_forbidden_vision_tokens(
            list(output.token_ids),
            self.forbidden_response_token_ids,
            self.tokenizer,
            list(output.log_probs) if output.log_probs is not None else None,
        )
        if removed_tokens:
            logger.warning(
                "Removed multimodal control tokens from assistant response in env:%s request:%s tokens=%s",
                agent_data.env_name,
                agent_data.request_id,
                removed_tokens,
            )

        agent_data.response_ids = response_ids
        if len(agent_data.response_ids)>agent_data.response_limit:
            logger.warning(f"In env:{agent_data.env_name}, generated response length {len(agent_data.response_ids)} exceeds per-turn response_limit {agent_data.response_limit}")
        agent_data.prompt_ids += agent_data.response_ids
        agent_data.response_mask += [1] * len(agent_data.response_ids)
        if response_logprobs is not None:
            agent_data.response_logprobs += response_logprobs

        # Cache assistant text and add assistant message (text-only)
        assistant_message = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
        )
        agent_data.last_assistant_text = assistant_message
        agent_data.messages.append({"role": "assistant", "content": assistant_message})
        return AgentState.INTERACTING

    async def _handle_env_state(self, agent_data: AgentData, **kwargs) -> AgentState:
        """
        Step the environment with last assistant action; always collect reward first.
        If terminal (done/success/turn-limit/token-limit), stop WITHOUT appending user suffix,
        so the episode ends on an assistant turn.
        """
        action_str = agent_data.last_assistant_text or ""
        try:
            obs, reward, done, info = await agent_data.env.step(action_str)
            # traceback
        except Exception as exc:
            logger.error(
                "Environment step failed in '%s' with action %r: %s",
                agent_data.env_name,
                action_str,
                exc,
            )
            logger.error("Environment traceback:\n%s", traceback.format_exc())
            obs, reward, done, info = {"obs_str":"Environment Error"}, 0.0, True, {"traj_success": False}

        agent_data.env_rewards.append(float(reward))
        graph_state = info.get("graph_state") if isinstance(info, dict) else None
        if graph_state is not None:
            agent_data.graph_states.append(graph_state)
        agent_data.traj_success = extract_success(info)
        # Validation turns (e.g. perception checks) don't count toward the turn budget or penalties
        if not (isinstance(info, dict) and info.get("is_validation_turn", False)):
            agent_data.env_turns += 1
        agent_data.last_info = info if isinstance(info, dict) else {}
        # Accumulate reward components for exploration (non-terminal) turns
        if not done and not (isinstance(info, dict) and info.get("is_validation_turn", False)):
            agent_data.total_step_penalty += -0.1
            if not info.get("is_valid_action", True):
                agent_data.total_invalid_penalty += -0.5
        # Termination rule #3: env done or success
        if done or agent_data.traj_success:
            return AgentState.TERMINATED

        # Termination rule #2: env turn-limit (if set)
        if self.env_max_turns is not None and agent_data.env_turns >= int(self.env_max_turns):
            return AgentState.TERMINATED

        # Termination rule #1: response token-limit
        if len(agent_data.response_mask) >= self.response_length:
            return AgentState.TERMINATED

        # Not terminal -> append user suffix for next turn
        user_content = convert_obs_to_content(obs, **kwargs)
        user_msg = {"role": "user", "content": user_content}
        agent_data.messages.append(user_msg)

        new_images = obs.get("multi_modal_input", {}).get("<image>", []) or []
        new_images = _normalize_images(new_images)

        _placeholder = {"role": "system", "content": "placeholder"}
        if self.processor is not None:
            raw_user_suffix = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    [_placeholder, user_msg],
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            model_inputs = self.processor(text=[raw_user_suffix], images=new_images or None, return_tensors="pt")
            response_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            if new_images:
                raise ValueError("Environment returned images but `processor` is None.")

            flat_user_msg = _flatten_text_only_content(user_msg)
            response_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    [_placeholder, flat_user_msg], add_generation_prompt=True,
                    tokenize=True, return_dict=False, **self.apply_chat_template_kwargs
                ),
            )
        response_ids = response_ids[len(self.system_prompt_prefix):]
        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)

        if new_images:
            agent_data.image_data.extend(new_images)

        return AgentState.GENERATING
