import numpy as np
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple

from vagen.envs.gym_image_env import GymImageEnv
from .env_config import SpatialGymConfig
from .managers.exploration_manager import ExplorationManager
from .managers.cognitive_map_manager import CognitiveMapManager
from .actions.base import BaseAction
from .managers.agent_proxy import get_agent_proxy
from .prompts import PromptManager
from .utils.room_utils import initialize_room_from_json
from .utils.utils import parse_llm_response, execute_exploration_action, get_agent_view
from .utils.image_handler import ImageHandler
from .actions.actions import configure_actions
from .evaluation.task_types import EvalTaskType
from gymnasium.utils import seeding


class EnvPhase(Enum):
    EXPLORATION_PERCEPTION = "exploration_perception"
    EXPLORATION_ACTION = "exploration_action"
    COGMAP = "cogmap"
    EVAL_TASK = "eval_task"
    DONE = "done"


class SpatialGym(GymImageEnv):
    """
    Spatial Gym Environment.
    Supports optional perception validation and post-exploration eval tasks.
    """
    def __init__(self, env_config: Dict[str, Any]):
        super().__init__(env_config)
        if isinstance(env_config, dict):
            try:
                valid_fields = SpatialGymConfig.__dataclass_fields__.keys()
                filtered_config = {k: v for k, v in env_config.items() if k in valid_fields}
                self.config = SpatialGymConfig(**filtered_config)
            except Exception as e:
                print(f"Warning: partial config init failed: {e}. Using default config.")
                self.config = SpatialGymConfig()
        else:
            self.config = env_config

        self.prompter: PromptManager = PromptManager(self.config)
        self.action_classes = configure_actions('exploration')

        # State
        self.phase: EnvPhase = EnvPhase.EXPLORATION_ACTION
        self.remaining_exp_steps: int = 0
        self.render_cache = None
        self.current_turn_number: int = 0      # total step() calls (including perception)
        self.effective_turns: int = 0           # only action + cogmap + eval turns (NOT perception)
        self.observed_image_paths: List[str] = []
        self.forced_term_occurred: bool = False

        # Room / exploration
        self.initial_room = None
        self.initial_agent = None
        self.exploration_manager = None

        # Perception state
        self.perception_retries_left: int = 0
        self.best_perception_score: float = 0.0

        # Eval task state
        self.eval_task_queue: List[Tuple] = []  # [(task, question), ...]
        self.eval_task_scores: List[float] = []

        # Tracks whether last action turn had observe + visible objects
        self._last_action_had_observe: bool = False
        self._last_visible_objects: List[str] = []

    def _generate_initial_observation(self) -> Tuple[Dict[str, Any], Any]:
        """Generate initial observation based on exploration type."""
        exp_history = {}
        images = []
        final_loc = None
        if self.config.exp_type == 'passive':
            proxy = get_agent_proxy(
                self.config.proxy_agent,
                self.initial_room,
                self.agent,
                grid_size=self.config.grid_size if hasattr(self.config, 'grid_size') else None,
            )
            proxy.run()
            if self.config.render_mode == 'vision':
                obs_str = proxy.to_text(self.config.image_placeholder)
                image_paths = []
                for t in proxy.turns:
                    if any('observe' in result.action_type for result in t.actions):
                        image, image_path = get_agent_view(
                            proxy.mgr, t.pos, t.ori, self.image_handler, seed=self.current_seed
                        )
                        images.append(image)
                        image_paths.append(image_path)
                assert images, "No images captured for vision render mode"
                exp_history['multi_modal_data'] = {self.config.image_placeholder: images}
                exp_history['multi_modal_data_paths'] = image_paths
            else:
                obs_str = proxy.to_text()
            exp_history['obs_str'] = obs_str
            self.exploration_manager = proxy.mgr
            final_loc = (list(proxy.turns[-1].pos), list(proxy.turns[-1].ori))

        obs_dict, self.observed_image_paths = self.prompter.get_initial_observation_prompt(
            room=self.initial_room,
            agent=self.agent,
            exp_history=exp_history,
        )

        obs = {'obs_str': obs_dict['obs_str']}
        mm_data = exp_history.get('multi_modal_data') or obs_dict.get('multi_modal_data') or {}
        if mm_data:
            obs['multi_modal_input'] = mm_data

        return obs, final_loc

    async def system_prompt(self) -> Dict[str, Any]:
        return {'obs_str': self.prompter.system_prompt()}

    async def reset(self, seed: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset environment for a new episode."""
        self.current_seed = seed
        self.np_random, seed = seeding.np_random(seed)

        self.action_classes = configure_actions('exploration')

        self.image_handler = ImageHandler(self.config.data_dir, seed, image_size=self.config.image_size)
        self.json_data = self.image_handler.json_data
        self.prompter = PromptManager(self.config, self.np_random, self.image_handler)

        self.initial_room, self.agent = initialize_room_from_json(self.json_data)
        self.initial_agent = self.agent.copy()

        self.remaining_exp_steps = self.config.max_exp_steps
        self.current_turn_number = 0
        self.effective_turns = 0
        self.observed_image_paths = []
        self.forced_term_occurred = False
        self.phase = EnvPhase.EXPLORATION_ACTION

        # Perception state reset
        self.perception_retries_left = 0
        self.best_perception_score = 0.0
        self._last_action_had_observe = False
        self._last_visible_objects = []

        # Eval task state reset
        self.eval_task_queue = []
        self.eval_task_scores = []

        BaseAction.set_field_of_view(self.config.field_of_view)
        self.exploration_manager = ExplorationManager(
            self.initial_room, self.agent,
            grid_size=(self.config.grid_size if hasattr(self.config, 'grid_size') else None),
            seed=seed,
        )

        info = {}
        if self.config.exp_type == 'passive':
            info['finish'] = True

        obs, _ = self._generate_initial_observation()
        self.render_cache = obs
        self.observed_image_paths = []

        # For active exploration, check if initial FOV has visible objects for perception
        if self.config.exp_type == 'active' and self._should_do_perception_initial():
            self.phase = EnvPhase.EXPLORATION_PERCEPTION
            self.perception_retries_left = self.config.max_perception_retries
            self.best_perception_score = 0.0
            # Append current FOV image + perception prompt.
            self._append_fov_image(obs)
            obs['obs_str'] += '\n\n' + self.prompter.get_perception_prompt()
        elif self.config.exp_type == 'active':
            # No perception needed — show action format directly.
            obs['obs_str'] += '\n\n' + self.prompter.get_format_footer(True)

        return obs, info

    def _should_do_perception_initial(self) -> bool:
        """Check if perception should trigger on the very first turn."""
        if not self.config.require_perception:
            return False
        # Check if initial FOV has visible objects
        from .actions.actions import ObserveAction
        obs_result = ObserveAction().execute(
            self.exploration_manager.exploration_room, self.exploration_manager.agent
        )
        visible = obs_result.data.get('visible_objects', []) if obs_result.success else []
        if visible:
            self._last_action_had_observe = True
            self._last_visible_objects = list(visible)
            return True
        return False

    def _should_do_perception(self) -> bool:
        """Check if perception should trigger before next action turn."""
        return (
            self.config.require_perception
            and self._last_action_had_observe
            and len(self._last_visible_objects) > 0
        )

    def _append_fov_image(self, obs: dict) -> None:
        """Append current FOV image to *obs* in-place (text placeholder + pixel data)."""
        if self.config.render_mode != 'vision' or self.image_handler is None:
            return
        image, image_path = get_agent_view(
            self.exploration_manager,
            self.exploration_manager.agent.pos,
            self.exploration_manager.agent.ori,
            self.image_handler,
            seed=self.current_seed,
        )
        mm = obs.get('multi_modal_input', {})
        imgs = list(mm.get(self.config.image_placeholder, []))
        imgs.append(image)
        obs['multi_modal_input'] = {self.config.image_placeholder: imgs}
        obs['obs_str'] += f'\n\nYour current view:\n{self.config.image_placeholder}'
        if image_path:
            self.observed_image_paths.append(image_path)

    # ------------------------------------------------------------------
    # step() dispatcher
    # ------------------------------------------------------------------

    async def step(self, action_str: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        self.current_turn_number += 1
        is_perception = (self.phase == EnvPhase.EXPLORATION_PERCEPTION)

        if is_perception:
            turn_category = "perception"
            result = self._handle_perception(action_str)
        elif self.phase == EnvPhase.EXPLORATION_ACTION:
            turn_category = "exploration"
            self.effective_turns += 1
            result = self._handle_action(action_str)
        elif self.phase == EnvPhase.COGMAP:
            turn_category = "exploration"  # cogmap is part of exploration flow
            self.effective_turns += 1
            result = self._handle_cogmap(action_str)
        elif self.phase == EnvPhase.EVAL_TASK:
            turn_category = "evaluation"
            self.effective_turns += 1
            result = self._handle_eval_task(action_str)
        else:
            raise RuntimeError(f"step() called in unexpected phase: {self.phase}")

        obs, reward, done, info = result
        info['effective_turns'] = self.effective_turns
        info['is_validation_turn'] = is_perception
        info['turn_category'] = turn_category
        return obs, reward, done, info

    # ------------------------------------------------------------------
    # Phase handlers
    # ------------------------------------------------------------------

    def _handle_perception(self, action_str: str):
        """Validate agent's local cogmap perception of current FOV."""
        _, perception_answer, _ = parse_llm_response(
            action_str, enable_think=bool(self.config.prompt_config.get('enable_think', True))
        )
        cogmap_str = perception_answer if perception_answer else action_str

        scores = CognitiveMapManager.score_local_cogmap(
            cogmap_str,
            self.exploration_manager.exploration_room,
            self.exploration_manager.agent,
        )
        overall = scores['overall']
        self.best_perception_score = max(self.best_perception_score, overall)
        passed = overall >= self.config.perception_pass_threshold

        # Count how many objects the agent reported vs ground truth visible
        n_visible = len(self._last_visible_objects)
        # Estimate reported count from the JSON
        n_reported = self._count_reported_objects(cogmap_str)

        if passed:
            # Proceed to action turn
            self.phase = EnvPhase.EXPLORATION_ACTION
            feedback = self.prompter.get_perception_feedback(
                True, overall, self.config.perception_pass_threshold, n_visible, n_reported, self.perception_retries_left
            )
            obs = {'obs_str': feedback + '\n' + self.prompter.steps_left_message(self.remaining_exp_steps) + '\n' + self.prompter.get_format_footer(True)}
            # Re-attach the FOV image so agent can plan actions
            self._append_fov_image(obs)
            self.render_cache = obs
            return obs, 0.0, False, {'perception_passed': True, 'perception_score': overall}

        # Failed
        self.perception_retries_left -= 1
        if self.perception_retries_left >= 0:
            # Retry with same FOV
            feedback = self.prompter.get_perception_feedback(
                False, overall, self.config.perception_pass_threshold, n_visible, n_reported, self.perception_retries_left
            )
            obs = {'obs_str': feedback + '\n\n' + self.prompter.get_perception_prompt()}
            # Re-show FOV image (placeholder + pixel data)
            self._append_fov_image(obs)
            self.render_cache = obs
            return obs, 0.0, False, {'perception_passed': False, 'perception_score': overall, 'perception_retry': True}

        # All retries exhausted — consume 1 step, skip action
        self.remaining_exp_steps -= 1
        reward = -0.1 - self.config.perception_fail_penalty

        if self.remaining_exp_steps <= 0:
            # Budget exhausted, go to cogmap
            return self._enter_cogmap_phase(forced_term=True, extra_reward=reward)

        # Stay in place, next turn. Check if perception should trigger again (same pos, same FOV)
        if self._should_do_perception():
            self.phase = EnvPhase.EXPLORATION_PERCEPTION
            self.perception_retries_left = self.config.max_perception_retries
            self.best_perception_score = 0.0
            feedback = self.prompter.get_perception_feedback(
                False, overall, self.config.perception_pass_threshold, n_visible, n_reported, 0
            )
            obs_str = feedback + f"\n{self.prompter.steps_left_message(self.remaining_exp_steps)}"
            obs_str += '\n\n' + self.prompter.get_perception_prompt()
            obs = {'obs_str': obs_str}
            self._append_fov_image(obs)
        else:
            self.phase = EnvPhase.EXPLORATION_ACTION
            obs = {'obs_str': self.prompter.steps_left_message(self.remaining_exp_steps) + '\n' + self.prompter.get_format_footer(True)}
            self._append_fov_image(obs)

        self.render_cache = obs
        return obs, reward, False, {'perception_passed': False, 'perception_score': overall, 'perception_exhausted': True}

    def _handle_action(self, action_str: str):
        """Execute exploration action (existing logic, extracted from old step())."""
        _, action, _ = parse_llm_response(
            action_str, enable_think=bool(self.config.prompt_config.get('enable_think', True))
        )

        obs, reward, done, info, exp_log, self.remaining_exp_steps, awaiting_cogmap, image_path = (
            execute_exploration_action(
                action,
                self.exploration_manager,
                self.action_classes,
                self.remaining_exp_steps,
                self.config,
                self.prompter,
                image_handler=self.image_handler if self.config.render_mode == 'vision' else None,
                seed=self.current_seed,
            )
        )

        if info.get('forced_term'):
            self.forced_term_occurred = True

        if image_path:
            self.observed_image_paths.append(image_path)
        else:
            self.observed_image_paths = []

        # Track whether this action ended with observe and had visible objects
        self._last_action_had_observe = False
        self._last_visible_objects = []
        if exp_log and not awaiting_cogmap:
            self._last_visible_objects = list(exp_log.visible_objects or [])
            self._last_action_had_observe = len(self._last_visible_objects) > 0 or bool(exp_log.visible_objects is not None)
            # More precise: check if an observe action was in the executed actions
            executed = info.get('action_executed', [])
            self._last_action_had_observe = any('Observe' in str(a) for a in executed)

        if awaiting_cogmap:
            self.phase = EnvPhase.COGMAP
        elif not done and self._should_do_perception():
            self.phase = EnvPhase.EXPLORATION_PERCEPTION
            self.perception_retries_left = self.config.max_perception_retries
            self.best_perception_score = 0.0
            obs['obs_str'] += '\n\n' + self.prompter.get_perception_prompt()
        elif not done and not awaiting_cogmap:
            obs['obs_str'] += '\n' + self.prompter.get_format_footer(True)

        self.render_cache = obs
        return obs, reward, done, info

    def _handle_cogmap(self, action_str: str):
        """Score global cognitive map (existing logic, extracted from old step())."""
        _, cogmap_answer, _ = parse_llm_response(
            action_str, enable_think=bool(self.config.prompt_config.get('enable_think', True))
        )
        cogmap_str = cogmap_answer if cogmap_answer else action_str
        cogmap_scores = CognitiveMapManager.score_global_cogmap(
            cogmap_str,
            self.exploration_manager.exploration_room,
            self.exploration_manager.agent,
            list(self.exploration_manager.observed_items),
        )
        n_total = len(self.exploration_manager.node_names)
        n_observed = len(self.exploration_manager.observed_nodes)
        exploration_coverage = n_observed / n_total if n_total > 0 else 1.0
        _, reward, info = CognitiveMapManager.compute_cogmap_reward(
            cogmap_scores, exploration_coverage, forced_term=self.forced_term_occurred,
        )

        # Check if we should do eval tasks
        if self.config.enable_eval_tasks:
            self._prepare_eval_task_queue()
            if self.eval_task_queue:
                # Serve first eval task
                self.phase = EnvPhase.EVAL_TASK
                task, question = self.eval_task_queue[0]
                obs = {'obs_str': self.prompter.get_eval_task_prompt(question)}
                # Attach image if vision eval task needs it
                obs = self._attach_eval_task_image(obs, task)
                self.render_cache = obs
                # Store cogmap reward to add later
                self._cogmap_reward = reward
                self._cogmap_info = info
                return obs, 0.0, False, {'cogmap_done': True, **info}

        obs = {'obs_str': self.prompter.task_finished_message()}
        self.render_cache = obs
        self.phase = EnvPhase.DONE
        return obs, reward, True, info

    def _handle_eval_task(self, action_str: str):
        """Evaluate agent's answer to current eval task, serve next or finish."""
        _, answer, _ = parse_llm_response(
            action_str, enable_think=bool(self.config.prompt_config.get('enable_think', True))
        )
        answer_str = answer if answer else action_str

        # Evaluate current task
        task, _ = self.eval_task_queue.pop(0)
        try:
            score, eval_info = task.evaluate(answer_str)
            # Keep continuous score (float); bool True/False → 1.0/0.0
            self.eval_task_scores.append(float(score) if score is not None else 0.0)
        except Exception:
            self.eval_task_scores.append(0.0)
            eval_info = {}

        if self.eval_task_queue:
            # Serve next task
            next_task, next_question = self.eval_task_queue[0]
            obs = {'obs_str': self.prompter.get_eval_task_prompt(next_question)}
            obs = self._attach_eval_task_image(obs, next_task)
            self.render_cache = obs
            return obs, 0.0, False, {'eval_task_correct': bool(self.eval_task_scores[-1]), **eval_info}

        # All tasks done — compute final reward
        eval_reward = 0.0
        if self.eval_task_scores:
            mean_score = sum(self.eval_task_scores) / len(self.eval_task_scores)
            eval_reward = self.config.eval_task_reward_scale * mean_score

        total_reward = self._cogmap_reward + eval_reward
        # Build per-task score dict: eval_dir, eval_rot, etc.
        eval_per_task = {}
        for i, task_config in enumerate(self.config.eval_tasks):
            if i < len(self.eval_task_scores):
                eval_per_task[f"eval_{task_config['task_type']}"] = self.eval_task_scores[i]
        info = {
            **self._cogmap_info,
            'eval_task_scores': self.eval_task_scores,
            'eval_task_reward': eval_reward,
            'eval_task_mean': mean_score if self.eval_task_scores else 0.0,
            **eval_per_task,
        }

        obs = {'obs_str': self.prompter.task_finished_message()}
        self.render_cache = obs
        self.phase = EnvPhase.DONE
        return obs, total_reward, True, info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _enter_cogmap_phase(self, forced_term: bool = False, extra_reward: float = 0.0):
        """Transition into cogmap phase."""
        self.phase = EnvPhase.COGMAP
        if forced_term:
            self.forced_term_occurred = True
        obs = {'obs_str': self.prompter.get_cogmap_output_prompt()}
        self.render_cache = obs
        info = {'forced_term': forced_term} if forced_term else {}
        return obs, extra_reward, False, info

    def _prepare_eval_task_queue(self):
        """Instantiate eval tasks from config."""
        self.eval_task_queue = []
        for task_config in self.config.eval_tasks:
            task_type = task_config['task_type']
            task_kwargs = task_config.get('task_kwargs', {}) or {}
            try:
                task = EvalTaskType.create_task(
                    task_type, self.np_random,
                    self.exploration_manager.exploration_room,
                    self.exploration_manager.agent,
                    task_kwargs,
                )
                question = task.generate_question()
                self.eval_task_queue.append((task, question))
            except (ValueError, Exception):
                continue  # Skip tasks that fail to generate

    def _attach_eval_task_image(self, obs: dict, task) -> dict:
        """Attach image to obs if the eval task question references <image>.

        For vision eval tasks the image must match the pose described in the
        question, which is not always ``task.agent`` (create_task resets to
        init).  We check ``eval_data.answer`` for an explicit final pose first.
        """
        if self.config.image_placeholder not in str(task.eval_data.question):
            return obs
        if self.image_handler is None:
            return obs
        # Determine the correct viewing pose for the image
        answer = task.eval_data.answer
        if isinstance(answer, dict) and 'final_pos' in answer and 'final_ori' in answer:
            import numpy as np
            pos = np.asarray(answer['final_pos'], dtype=float)
            ori = np.asarray(answer['final_ori'], dtype=float)
        else:
            pos = task.agent.pos
            ori = task.agent.ori
        try:
            image, _ = get_agent_view(
                self.exploration_manager, pos, ori,
                self.image_handler, seed=self.current_seed,
            )
            obs['multi_modal_input'] = {self.config.image_placeholder: [image]}
        except Exception:
            pass
        return obs

    def _count_reported_objects(self, cogmap_str: str) -> int:
        """Estimate how many objects the agent reported in their local cogmap JSON."""
        import json, re
        try:
            # Try to extract JSON
            match = re.search(r'\{.*\}', cogmap_str, re.DOTALL)
            if match:
                data = json.loads(match.group())
                objects = data.get('objects', data)
                if isinstance(objects, dict):
                    return len([k for k in objects.keys() if k != 'origin'])
        except Exception:
            pass
        return 0

    def render(self):
        return self.render_cache

    async def close(self):
        return

    def get_exp_summary(self):
        """Get exploration efficiency metrics."""
        return self.exploration_manager.get_exp_summary() if self.exploration_manager else ExplorationManager.DEFAULT_EXP_SUMMARY

    def _get_env_info(self):
        """Get environment state information."""
        return {
            "config": self.config.to_dict(),
            "initial_room": self.initial_room.to_dict(),
            "initial_agent": self.initial_agent.to_dict(),
        }
