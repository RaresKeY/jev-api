#!/usr/bin/env python3
"""
Odysseus AI Workstation: Jev System One Model Simulation & Evaluation Suite

This script demonstrates how Jev (TypeSafe AI's non-autoregressive, calibrated decision model)
operates across critical agentic lifecycle gates in an AI workstation environment:

1. Context Compaction & Short-Reply Follow-up Detection
2. Dynamic Tool Selection & Capability Pruning
3. Task Difficulty Scoring & Model Tier Routing
4. Pre-Execution Safety & Human-in-the-Loop (HITL) Gating
5. Fast, Cost-Effective Completion Verification
6. Computer Use / GUI Automation Action Verification & Alignment

Usage:
    # Run against a managed llama.cpp server:
    .venv/bin/python odysseus_workstation_sim.py

    # Run with live Jev API:
    export TYPESAFE_API_KEY="your-api-key"
    .venv/bin/python odysseus_workstation_sim.py --backend jev
"""

import os
import sys
import json
import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

# Check if typesafe-sdk is installed
try:
    from typesafe_sdk import TypeSafeClient, Choice, Score, Noul
    from typesafe_sdk import SystemOneResponse
    TYPESAFE_SDK_AVAILABLE = True
except ImportError:
    TYPESAFE_SDK_AVAILABLE = False

    class Question:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class Choice(Question):
        type = "choice"

    class Score(Question):
        type = "score"

    class Noul(Question):
        type = "noul"



# ==============================================================================
# Explicit Mock Engine (not calibrated; never an automatic fallback)
# ==============================================================================

class MockSystemOneResponse:
    """Mock SystemOneResponse matching typesafe_sdk schema."""
    def __init__(self, choices=None, scores=None, nouls=None, model="jev-latest"):
        self.model = model
        self.usage = {"input_tokens": 142, "output_tokens": 0}
        self.choices = choices or {}
        self.scores = scores or {}
        self.nouls = nouls or {}

@dataclass
class MockChoiceAnswer:
    choice: str
    confidence: float
    probabilities: Dict[str, float]

@dataclass
class MockScoreAnswer:
    score: float
    confidence: float
    legend: List[str]
    probabilities: List[float]

@dataclass
class MockNoulAnswer:
    probability: float  # p(yes)


class MockTypeSafeClient:
    """Demo heuristics with fabricated scores and probabilities."""
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def system_one(self, state: Dict[str, Any], questions: Dict[str, Any], **kwargs):
        # Add realistic sub-150ms simulated latency
        time.sleep(0.08)
        
        state_str = json.dumps(state).lower()
        mock_choices = {}
        mock_scores = {}
        mock_nouls = {}

        for q_name, q_obj in questions.items():
            q_type = q_obj.type if hasattr(q_obj, 'type') else q_obj.get('type')
            instructions = (q_obj.instructions if hasattr(q_obj, 'instructions') else q_obj.get('instructions', '')).lower()
            
            # --- NOUL (Binary calibrated probability) ---
            if q_type == 'noul':
                p = 0.5
                if any(w in instructions or w in q_name for w in ["dangerous", "destructive", "harm", "delete", "wipe"]):
                    if any(k in state_str for k in ["rm -rf", "drop database", "wipe", "prune -a", "format"]):
                        p = 0.98
                    elif any(k in state_str for k in ["restart", "deploy", "overwrite"]):
                        p = 0.65
                    else:
                        p = 0.04
                elif any(w in instructions or w in q_name for w in ["continuation", "follow-up"]):
                    if any(k in state_str for k in ["looks good", "now deploy", "do it", "try again", "yes"]):
                        p = 0.94
                    else:
                        p = 0.12
                elif any(w in instructions or w in q_name for w in ["compact", "can_compact"]):
                    p = 0.1  # Summary alone cannot establish safe history compaction
                elif any(w in instructions or w in q_name for w in ["bash", "terminal"]):
                    p = 0.92 if any(k in state_str for k in ["restart", "run", "docker", "server", "prune"]) else 0.05
                elif any(w in instructions or w in q_name for w in ["filesystem", "file"]):
                    p = 0.95 if any(k in state_str for k in ["edit", "code", "file", "redis", "fix"]) else 0.10
                elif any(w in instructions or w in q_name for w in ["gui", "desktop", "mouse"]):
                    p = 0.96 if any(k in state_str for k in ["click", "browser", "window", "desktop", "button"]) else 0.02
                elif any(w in instructions or w in q_name for w in ["satisfied", "completed"]):
                    p = 0.91 if ("success" in state_str or "passed" in state_str or "resolved" in state_str) else 0.25
                elif any(w in instructions or w in q_name for w in ["advance", "alignment"]):
                    p = 0.95 if ("click" in state_str and "button" in state_str) or "chrome" in state_str else 0.40
                elif any(w in instructions or w in q_name for w in ["human guidance", "intervention"]):
                    p = 0.85 if ("destructive" in state_str or "confirm" in state_str or "captcha" in state_str) else 0.08
                else:
                    p = 0.82
                mock_nouls[q_name] = MockNoulAnswer(probability=round(p, 3))

            # --- CHOICE (Categorical selection) ---
            elif q_type == 'choice':
                criteria = q_obj.criteria if hasattr(q_obj, 'criteria') else q_obj.get('criteria', {})
                options = list(criteria.keys()) if isinstance(criteria, dict) else list(criteria)
                
                selected = options[0]
                probs = {opt: round(1.0 / len(options), 3) for opt in options}
                
                # Context compaction / reply intent
                if "intent_type" in q_name:
                    if any(k in state_str for k in ["looks good", "lgtm", "yes", "do it"]):
                        selected = "confirmation" if "confirmation" in options else options[0]
                    elif any(k in state_str for k in ["now deploy", "next", "also add"]):
                        selected = "continuation" if "continuation" in options else options[0]
                    elif any(k in state_str for k in ["thanks", "cool", "hello"]):
                        selected = "casual_low_signal" if "casual_low_signal" in options else options[0]
                    else:
                        selected = "new_task" if "new_task" in options else options[0]
                
                # Primary domain / tool selection
                elif "primary_domain" in q_name:
                    if "click" in state_str or "window" in state_str:
                        selected = "computer_use_gui"
                    elif "fix" in state_str or "redis" in state_str or "code" in state_str:
                        selected = "code_editing"
                    elif "wipe" in state_str or "docker" in state_str:
                        selected = "terminal_execution"
                    elif "search" in state_str or "find" in state_str:
                        selected = "web_research"
                    else:
                        selected = options[0]

                # Model routing
                elif "tier" in q_name or "recommended_tier" in q_name:
                    if any(k in state_str for k in ["race condition", "architecture", "refactor", "concurrency"]):
                        selected = "tier_3_heavy"
                    elif any(k in state_str for k in ["restart", "docker", "deploy", "edit"]):
                        selected = "tier_2_balanced"
                    else:
                        selected = "tier_1_fast"

                # Next step verification
                elif "next_step" in q_name:
                    selected = "finish_and_respond" if ("passed" in state_str or "fixed" in state_str or "resolved" in state_str) else "retry_with_correction"

                # Safety gating
                elif "gating" in q_name:
                    if any(k in state_str for k in ["rm -rf", "drop database", "wipe", "prune -a"]):
                        selected = "block_until_explicit_user_confirmation"
                    elif any(k in state_str for k in ["deploy", "restart server"]):
                        selected = "notify_user_async"
                    else:
                        selected = "auto_approve"

                # Computer use execution strategy
                elif "strategy" in q_name or "execution_strategy" in q_name:
                    selected = "abort_and_ask_user" if any(k in state_str for k in ["captcha", "2fa", "crash", "confirm purchase"]) else "proceed_with_mouse_keyboard"

                # Distribute high probability to selected
                for opt in options:
                    probs[opt] = 0.05
                probs[selected] = 0.85
                total = sum(probs.values())
                probs = {k: round(v / total, 3) for k, v in probs.items()}

                mock_choices[q_name] = MockChoiceAnswer(
                    choice=selected,
                    confidence=0.89,
                    probabilities=probs
                )

            # --- SCORE (Ordered rubric ranking) ---
            elif q_type == 'score':
                criteria = q_obj.criteria if hasattr(q_obj, 'criteria') else q_obj.get('criteria', [])
                num_levels = len(criteria) if criteria else 5
                
                # Determine score based on signals
                if "complexity" in q_name or "difficulty" in instructions:
                    if any(k in state_str for k in ["race condition", "concurrency", "distributed"]):
                        score_val = 4.2  # High complexity
                    elif any(k in state_str for k in ["docker", "deploy", "restart"]):
                        score_val = 2.8
                    else:
                        score_val = 1.1
                elif "risk" in q_name:
                    if any(k in state_str for k in ["rm -rf", "wipe", "prune -a"]):
                        score_val = 4.8  # Critical risk
                    elif any(k in state_str for k in ["deploy", "restart"]):
                        score_val = 2.7
                    else:
                        score_val = 0.4
                elif "ambiguity" in q_name:
                    score_val = 0.3 if "clear" in state_str or "button" in state_str else 2.5
                else:
                    score_val = 3.0

                score_val = max(0.0, min(score_val, num_levels - 1))
                probs = [0.0] * num_levels
                idx = min(int(round(score_val)), num_levels - 1)
                probs[idx] = 0.70
                if idx > 0:
                    probs[idx - 1] = 0.15
                if idx < num_levels - 1:
                    probs[idx + 1] = 0.15

                total = sum(probs)
                probs = [p / total for p in probs]
                mock_scores[q_name] = MockScoreAnswer(
                    score=score_val,
                    confidence=0.88,
                    legend=criteria if criteria else [f"Level {i}" for i in range(num_levels)],
                    probabilities=probs
                )

        return MockSystemOneResponse(
            choices=mock_choices,
            scores=mock_scores,
            nouls=mock_nouls
        )


# ==============================================================================
# Workstation Decision Engine Wrapping Jev
# ==============================================================================

class OdysseusDecisionEngine:
    """Encapsulates all Jev System One gates used by the Odysseus AI Workstation."""

    def __init__(self, api_key: Optional[str] = None, backend=None):
        from pathlib import Path
        from decision_lab.env import load_env
        load_env(Path(__file__).resolve().parent / '.env')
        self.api_key = api_key or os.getenv("JEV_API") or os.getenv("TYPESAFE_API_KEY")
        backend = backend or os.getenv("DECISION_BACKEND", "llama")
        self.is_live = backend != "mock"
        if backend == "llama":
            from local_llama import LlamaServerClient
            self.client = LlamaServerClient()
            print(f"[Decision Engine] llama.cpp: {self.client.base_url} (uncalibrated estimates)")
        elif backend == "jev":
            from decision_lab.jev import JevClient
            self.client = JevClient(key=self.api_key)
        elif backend == "mock":
            print("[Decision Engine] Mock heuristics; fabricated probabilities and latency")
            self.client = MockTypeSafeClient()
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def _call_jev(self, state: Dict[str, Any], questions: Dict[str, Any]) -> Any:
        start_t = time.perf_counter()
        response = self.client.system_one(state=state, questions=questions)
        duration_ms = (time.perf_counter() - start_t) * 1000
        return response, duration_ms

    # --------------------------------------------------------------------------
    # Gate 1: Smart Context Compaction & Short-Reply Detection
    # --------------------------------------------------------------------------
    def evaluate_reply_and_context(self, active_task_summary: str, user_reply: str):
        """Classifies short user replies (e.g. 'yes', 'do it', 'lgtm') to decide

        if it continues current work or pivots, and whether context can be compacted.
        """
        state = {
            "active_task_summary": active_task_summary,
            "user_reply": user_reply
        }
        questions = {
            "intent_type": Choice(
                instructions="Classify the intent of this user response relative to the active task.",
                criteria={
                    "confirmation": "Direct confirmation or approval to proceed (e.g., 'yes', 'do it', 'lgtm')",
                    "continuation": "Follow-up question, parameter tweak, or next step in the same task",
                    "new_task": "A distinct new instruction or prompt starting a new goal",
                    "casual_low_signal": "Social chit-chat, thanks, emoji, or non-actionable remarks"
                }
            ),
            "is_continuation": Noul(
                instructions="Does this message refer directly to continuing or refining the ongoing active task?"
            ),
            "can_compact_history": Noul(
                instructions="Can prior turn history before the current task summary be safely summarized/compacted without losing critical constraints?"
            )
        }
        res, latency = self._call_jev(state, questions)
        return {
            "gate": "Context Compaction & Follow-up Detection",
            "latency_ms": round(latency, 1),
            "intent": res.choices["intent_type"].choice,
            "intent_confidence": res.choices["intent_type"].confidence,
            "is_continuation": res.nouls["is_continuation"].probability >= 0.5,
            "is_continuation_prob": res.nouls["is_continuation"].probability,
            "can_compact_history": res.nouls["can_compact_history"].probability >= 0.5,
        }

    # --------------------------------------------------------------------------
    # Gate 2: Tool Selection & Capability Pruning
    # --------------------------------------------------------------------------
    def select_tools_and_domain(self, user_instruction: str, workspace_context: str):
        """Given a large registry of 20+ workstation tools, Jev filters down

        the exact sub-inventory needed for this prompt to minimize LLM token bloat.
        """
        state = {
            "instruction": user_instruction,
            "workspace_context": workspace_context
        }
        questions = {
            "primary_domain": Choice(
                instructions="What is the primary action domain required to fulfill this request?",
                criteria={
                    "code_editing": "Inspecting, editing, or refactoring code files",
                    "terminal_execution": "Running shell commands, package managers, or dev servers",
                    "web_research": "Searching documentation, fetching URLs, or web search",
                    "computer_use_gui": "Controlling the desktop GUI, mouse clicks, keystrokes, window focus",
                    "no_tools_needed": "Pure reasoning or direct conversational explanation"
                }
            ),
            "needs_bash": Noul(instructions="Does fulfilling this request require executing shell/terminal commands?"),
            "needs_filesystem": Noul(instructions="Does this request require reading, editing, or creating files?"),
            "needs_browser": Noul(instructions="Does this request require browsing URLs or performing web searches?"),
            "needs_gui": Noul(instructions="Does this request require GUI/mouse/keyboard desktop automation?")
        }
        res, latency = self._call_jev(state, questions)
        
        # Build active tool list from filtered capabilities
        active_tools = []
        if res.nouls["needs_filesystem"].probability >= 0.5:
            active_tools.extend(["view_file", "replace_file_content", "write_to_file"])
        if res.nouls["needs_bash"].probability >= 0.5:
            active_tools.extend(["run_command", "manage_task"])
        if res.nouls["needs_browser"].probability >= 0.5:
            active_tools.extend(["read_url_content", "search_web"])
        if res.nouls["needs_gui"].probability >= 0.5:
            active_tools.extend(["gui_click", "gui_type", "gui_screenshot", "gui_hotkey"])

        return {
            "gate": "Tool Selection & Capability Pruning",
            "latency_ms": round(latency, 1),
            "primary_domain": res.choices["primary_domain"].choice,
            "needs_bash": res.nouls["needs_bash"].probability,
            "needs_filesystem": res.nouls["needs_filesystem"].probability,
            "needs_browser": res.nouls["needs_browser"].probability,
            "needs_gui": res.nouls["needs_gui"].probability,
            "pruned_active_tools": active_tools
        }

    # --------------------------------------------------------------------------
    # Gate 3: Task Difficulty & Model Tier Routing
    # --------------------------------------------------------------------------
    def route_model_by_difficulty(self, user_instruction: str, context_summary: str):
        """Scores difficulty to route to the optimal model tier:

        Tier 1 (Flash/Haiku) -> Tier 2 (Sonnet/Pro) -> Tier 3 (Reasoning/o1/Opus).
        """
        state = {
            "instruction": user_instruction,
            "context_summary": context_summary
        }
        questions = {
            "complexity_score": Score(
                instructions="Rate the cognitive and architectural complexity of the task.",
                criteria=[
                    "Trivial: Single-line edit, direct question, or simple file lookup",
                    "Simple: Standard function modification or basic search",
                    "Moderate: Multi-file coordination or standard debugging",
                    "Complex: Cross-module refactoring or architecture design",
                    "Extreme: Concurrency race condition, deep algorithms, or high-risk orchestration"
                ]
            ),
            "recommended_tier": Choice(
                instructions="Recommend the model capability tier needed to solve this reliably.",
                criteria={
                    "tier_1_fast": "Fast, lightweight model (e.g. Gemini Flash, Claude Haiku)",
                    "tier_2_balanced": "Balanced production model (e.g. Claude Sonnet, Gemini Pro)",
                    "tier_3_heavy": "Heavy frontier reasoning model (e.g. o1, Claude Opus)"
                }
            )
        }
        res, latency = self._call_jev(state, questions)
        return {
            "gate": "Task Difficulty & Model Routing",
            "latency_ms": round(latency, 1),
            "complexity_score": res.scores["complexity_score"].score,
            "recommended_tier": res.choices["recommended_tier"].choice,
            "confidence": res.choices["recommended_tier"].confidence
        }

    # --------------------------------------------------------------------------
    # Gate 4: Pre-Execution Safety & Human-in-the-Loop Gating
    # --------------------------------------------------------------------------
    def evaluate_pre_execution_safety(self, user_prompt: str, proposed_tool: str, tool_args: Dict[str, Any]):
        """Evaluates whether a planned tool call is destructive, irreversible,

        or security-critical before running it.
        """
        state = {
            "user_prompt": user_prompt,
            "proposed_tool": proposed_tool,
            "tool_args": tool_args
        }
        questions = {
            "is_destructive": Noul(
                instructions="Will this proposed action delete, wipe, or permanently alter critical data or files?"
            ),
            "is_external_impact": Noul(
                instructions="Does this action affect external production environments, deployment servers, or network APIs?"
            ),
            "risk_score": Score(
                instructions="Assess the overall execution risk level.",
                criteria=[
                    "Safe: Read-only inspection or harmless calculation",
                    "Low: Easily reversible local edits or standard tests",
                    "Medium: Dependency installation or dev server restart",
                    "High: Service deployment, file deletion, or sensitive config edit",
                    "Critical: Database drop, disk wipe, or root command execution"
                ]
            ),
            "gating_action": Choice(
                instructions="What authorization policy should apply before executing this tool call?",
                criteria={
                    "auto_approve": "Run immediately in background without blocking",
                    "notify_user_async": "Execute while notifying user in the activity stream",
                    "block_until_explicit_user_confirmation": "Halt and require explicit human-in-the-loop modal click"
                }
            )
        }
        res, latency = self._call_jev(state, questions)
        return {
            "gate": "Pre-Execution Safety & HITL Gating",
            "latency_ms": round(latency, 1),
            "risk_score": res.scores["risk_score"].score,
            "is_destructive_prob": res.nouls["is_destructive"].probability,
            "is_external_prob": res.nouls["is_external_impact"].probability,
            "gating_decision": res.choices["gating_action"].choice
        }

    # --------------------------------------------------------------------------
    # Gate 5: Fast, Cheap Completion Verification
    # --------------------------------------------------------------------------
    def verify_completion(self, user_instruction: str, actions_taken: List[Dict[str, Any]], final_output: str):
        """Sub-500ms verification to check if the LLM actually finished the user's ask."""
        state = {
            "request": user_instruction,
            "actions_taken": actions_taken,
            "final_output": final_output
        }
        questions = {
            "is_task_satisfied": Noul(
                instructions="Did the actions taken and final output completely fulfill the user's original request?"
            ),
            "quality_score": Score(
                instructions="Rate the completion quality.",
                criteria=[
                    "Failed: Did not perform the requested task",
                    "Partial: Attempted but left errors or missed instructions",
                    "Adequate: Completed core instruction without extra validation",
                    "Robust: Fully implemented and verified with tests or diagnostics"
                ]
            ),
            "next_step": Choice(
                instructions="Determine the next agent loop action.",
                criteria={
                    "finish_and_respond": "Task is complete; present final response to user",
                    "retry_with_correction": "Task incomplete or failed; re-prompt model to fix defects",
                    "ask_clarification": "Ambiguity encountered; ask user for direction"
                }
            )
        }
        res, latency = self._call_jev(state, questions)
        return {
            "gate": "Fast Completion Verification",
            "latency_ms": round(latency, 1),
            "is_satisfied": res.nouls["is_task_satisfied"].probability >= 0.5,
            "satisfaction_probability": res.nouls["is_task_satisfied"].probability,
            "quality_score": res.scores["quality_score"].score,
            "next_step": res.choices["next_step"].choice
        }

    # --------------------------------------------------------------------------
    # Gate 6: Computer Use & Desktop GUI Exploration
    # --------------------------------------------------------------------------
    def evaluate_gui_action(self, goal: str, screen_context: str, proposed_action: Dict[str, Any]):
        """Evaluates OS desktop / GUI automation commands before dispatching mouse/keys."""
        state = {
            "user_goal": goal,
            "screen_context": screen_context,
            "proposed_action": proposed_action
        }
        questions = {
            "action_alignment": Noul(
                instructions="Does this proposed GUI action directly advance the stated goal on the current screen?"
            ),
            "requires_human_guidance": Noul(
                instructions="Is the GUI in an unexpected state (e.g. CAPTCHA, 2FA prompt, crash modal) requiring human takeover?"
            ),
            "ambiguity_score": Score(
                instructions="Assess the visual ambiguity of the click/interaction target.",
                criteria=[
                    "Clear: Unique, high-contrast button/element clearly identified",
                    "Low: Slight variation but recognizable target",
                    "Moderate: Multiple similar elements on screen",
                    "High: Target occluded or requires scrolling",
                    "Severe: Target not visible on current screen"
                ]
            ),
            "execution_strategy": Choice(
                instructions="How should the workstation execute this computer use step?",
                criteria={
                    "proceed_with_mouse_keyboard": "Dispatch mouse click / keystrokes immediately",
                    "wait_for_render": "Delay action for 500ms to allow UI animation to settle",
                    "abort_and_ask_user": "Pause computer use and request human intervention"
                }
            )
        }
        res, latency = self._call_jev(state, questions)
        return {
            "gate": "Computer Use Action Alignment & Visual Safety",
            "latency_ms": round(latency, 1),
            "alignment_prob": res.nouls["action_alignment"].probability,
            "needs_human_prob": res.nouls["requires_human_guidance"].probability,
            "ambiguity_score": res.scores["ambiguity_score"].score,
            "strategy": res.choices["execution_strategy"].choice
        }


# ==============================================================================
# Simulation Scenarios (Workstation prompts mimicking real AI developer tasks)
# ==============================================================================

def run_simulation():
    engine = OdysseusDecisionEngine()

    print("\n" + "=" * 80)
    print("🤖 ODYSSEUS AI WORKSTATION: JEV SYSTEM ONE INTEGRATION TEST")
    print("=" * 80)

    # --------------------------------------------------------------------------
    # Scenario 1: Multi-Tool Complex Task (Routing, Capability Pruning, Verification)
    # --------------------------------------------------------------------------
    print("\n[SCENARIO 1] Complex Task: Distributed Bug Fix & Server Restart")
    prompt_1 = "Fix the race condition in the Redis session store in src/auth.py and restart the dev server on port 6379."
    workspace_ctx = "Repo: backend-api, Python 3.12, Redis 7.0, 142 files in workspace"

    print(f"👤 User: \"{prompt_1}\"")

    # 1. Capability & Tool Selection
    tool_eval = engine.select_tools_and_domain(prompt_1, workspace_ctx)
    print(f"\n  ⚙️ [Gate: Tool Filtering] ({tool_eval['latency_ms']}ms)")
    print(f"     Domain: {tool_eval['primary_domain']}")
    print(f"     Pruned Capabilities: {tool_eval['pruned_active_tools']}")
    print(f"     Token savings: Only {len(tool_eval['pruned_active_tools'])} tools passed to LLM instead of all 24 workstation tools.")

    # 2. Model Routing by Difficulty
    route_eval = engine.route_model_by_difficulty(prompt_1, workspace_ctx)
    print(f"\n  🎯 [Gate: Model Routing] ({route_eval['latency_ms']}ms)")
    print(f"     Complexity: {route_eval['complexity_score']:.1f}/4.0")
    confidence = route_eval['confidence']
    confidence_text = 'unavailable' if confidence is None else f'{confidence:.2f}'
    print(f"     Routed Model: {route_eval['recommended_tier']} (Confidence: {confidence_text})")

    # 3. Simulated LLM Output & Completion Verification
    simulated_actions = [
        {"tool": "replace_file_content", "file": "src/auth.py", "status": "success"},
        {"tool": "run_command", "command": "pytest tests/test_session_race.py", "status": "3 passed"},
        {"tool": "run_command", "command": "systemctl restart redis-dev", "status": "active (running)"}
    ]
    simulated_reply = "I resolved the race condition in src/auth.py using an atomic Redis Lua script, verified with tests, and restarted the server on port 6379."
    
    verify_eval = engine.verify_completion(prompt_1, simulated_actions, simulated_reply)
    print(f"\n  ✅ [Gate: Fast Completion Verification] ({verify_eval['latency_ms']}ms)")
    print(f"     Task Satisfied: {verify_eval['is_satisfied']} (p = {verify_eval['satisfaction_probability']})")
    print(f"     Quality Score: {verify_eval['quality_score']:.1f}/3.0")
    print(f"     Next Action: {verify_eval['next_step']}")

    # --------------------------------------------------------------------------
    # Scenario 2: Smart Context Compaction & Short-Reply Follow-up
    # --------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("[SCENARIO 2] Context Compaction: Short Follow-up Reply")
    active_summary = "Resolved Redis race condition in src/auth.py and restarted dev server."
    short_reply = "Looks good, now deploy it to staging."

    print(f"📌 Active Context: \"{active_summary}\"")
    print(f"👤 User Reply: \"{short_reply}\"")

    reply_eval = engine.evaluate_reply_and_context(active_summary, short_reply)
    print(f"\n  🧠 [Gate: Follow-up & Compaction] ({reply_eval['latency_ms']}ms)")
    print(f"     Classified Intent: {reply_eval['intent']}")
    print(f"     Is Continuation: {reply_eval['is_continuation']} (p = {reply_eval['is_continuation_prob']})")
    print(f"     Compaction Safe: {reply_eval['can_compact_history']} (recommendation only; history unchanged)")

    # --------------------------------------------------------------------------
    # Scenario 3: Pre-Execution Safety & Human-in-the-Loop (Destructive Command)
    # --------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("[SCENARIO 3] Pre-Execution Safety Gating: Dangerous Action")
    danger_prompt = "Clean up the environment completely."
    dangerous_tool = "run_command"
    dangerous_args = {"command": "docker system prune -a --volumes -f && rm -rf /var/lib/testdb/*"}

    print(f"👤 User: \"{danger_prompt}\"")
    print(f"⚠️ Proposed Tool: {dangerous_tool}({dangerous_args['command']})")

    safety_eval = engine.evaluate_pre_execution_safety(danger_prompt, dangerous_tool, dangerous_args)
    print(f"\n  🛡️ [Gate: Pre-Execution Safety] ({safety_eval['latency_ms']}ms)")
    print(f"     Destructive Probability: {safety_eval['is_destructive_prob']}")
    print(f"     Risk Score: {safety_eval['risk_score']:.1f}/4.0")
    print(f"     Gating Decision: >>> {safety_eval['gating_decision'].upper()} <<<")
    if safety_eval['gating_decision'] == 'block_until_explicit_user_confirmation':
        print("     Confirmation recommended; this demo executes no actions.")

    # --------------------------------------------------------------------------
    # Scenario 4: Computer Use / Desktop GUI Action Evaluation
    # --------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("[SCENARIO 4] Computer Use: OS Desktop & GUI Action Verification")
    gui_goal = "Open Chrome and login to AWS console."
    screen_summary = "Desktop with dock visible. Chrome icon present at (45, 1020). No browser currently open."
    proposed_gui_action = {"action": "mouse_click", "coordinates": [45, 1020], "target": "Chrome Dock Icon"}

    print(f"🎯 GUI Goal: \"{gui_goal}\"")
    print(f"🖥️ Screen State: {screen_summary}")
    print(f"🖱️ Proposed GUI Step: {proposed_gui_action}")

    gui_eval = engine.evaluate_gui_action(gui_goal, screen_summary, proposed_gui_action)
    print(f"\n  🖥️ [Gate: Computer Use Alignment & Verification] ({gui_eval['latency_ms']}ms)")
    print(f"     Goal Alignment: {gui_eval['alignment_prob'] * 100:.0f}%")
    print(f"     Target Ambiguity: {gui_eval['ambiguity_score']:.1f}/4.0")
    print(f"     Execution Strategy: {gui_eval['strategy']}")

    print("\n" + "=" * 80)
    print("Decision demo complete: all six gates evaluated; no actions executed.")
    print("=" * 80)


def print_payloads():
    """Prints raw JSON requests and cURL snippets for all 6 Jev gates."""
    print("\n" + "=" * 80)
    print("📋 JEV SYSTEM ONE: RAW PAYLOAD & CURL SPECIFICATIONS")
    print("=" * 80)

    gates = [
        (
            "Gate 1: Context Compaction & Short-Reply Detection",
            {
                "model": "jev-latest",
                "state": {
                    "active_task_summary": "Resolved Redis race condition in src/auth.py and restarted dev server.",
                    "user_reply": "Looks good, now deploy it to staging."
                },
                "questions": {
                    "intent_type": {
                        "type": "choice",
                        "instructions": "Classify the intent of this user response relative to the active task.",
                        "criteria": {
                            "confirmation": "Direct confirmation or approval to proceed (e.g., 'yes', 'do it', 'lgtm')",
                            "continuation": "Follow-up question, parameter tweak, or next step in the same task",
                            "new_task": "A distinct new instruction or prompt starting a new goal",
                            "casual_low_signal": "Social chit-chat, thanks, emoji, or non-actionable remarks"
                        }
                    },
                    "is_continuation": {
                        "type": "noul",
                        "instructions": "Does this message refer directly to continuing or refining the ongoing active task?"
                    },
                    "can_compact_history": {
                        "type": "noul",
                        "instructions": "Can prior turn history before the current task summary be safely summarized/compacted without losing critical constraints?"
                    }
                }
            }
        ),
        (
            "Gate 2: Dynamic Tool Selection & Capability Pruning",
            {
                "model": "jev-latest",
                "state": {
                    "instruction": "Fix the race condition in the Redis session store in src/auth.py and restart the dev server on port 6379.",
                    "workspace_context": "Repo: backend-api, Python 3.12, Redis 7.0, 142 files in workspace"
                },
                "questions": {
                    "primary_domain": {
                        "type": "choice",
                        "instructions": "What is the primary action domain required to fulfill this request?",
                        "criteria": {
                            "code_editing": "Inspecting, editing, or refactoring code files",
                            "terminal_execution": "Running shell commands, package managers, or dev servers",
                            "web_research": "Searching documentation, fetching URLs, or web search",
                            "computer_use_gui": "Controlling the desktop GUI, mouse clicks, keystrokes, window focus",
                            "no_tools_needed": "Pure reasoning or direct conversational explanation"
                        }
                    },
                    "needs_bash": {"type": "noul", "instructions": "Does fulfilling this request require executing shell/terminal commands?"},
                    "needs_filesystem": {"type": "noul", "instructions": "Does this request require reading, editing, or creating files?"},
                    "needs_browser": {"type": "noul", "instructions": "Does this request require browsing URLs or performing web searches?"},
                    "needs_gui": {"type": "noul", "instructions": "Does this request require GUI/mouse/keyboard desktop automation?"}
                }
            }
        ),
        (
            "Gate 3: Task Difficulty & Model Routing",
            {
                "model": "jev-latest",
                "state": {
                    "instruction": "Fix the race condition in the Redis session store in src/auth.py and restart the dev server on port 6379.",
                    "context_summary": "Repo: backend-api, Python 3.12, Redis 7.0, 142 files in workspace"
                },
                "questions": {
                    "complexity_score": {
                        "type": "score",
                        "instructions": "Rate the cognitive and architectural complexity of the task.",
                        "criteria": [
                            "Trivial: Single-line edit, direct question, or simple file lookup",
                            "Simple: Standard function modification or basic search",
                            "Moderate: Multi-file coordination or standard debugging",
                            "Complex: Cross-module refactoring or architecture design",
                            "Extreme: Concurrency race condition, deep algorithms, or high-risk orchestration"
                        ]
                    },
                    "recommended_tier": {
                        "type": "choice",
                        "instructions": "Recommend the model capability tier needed to solve this reliably.",
                        "criteria": {
                            "tier_1_fast": "Fast, lightweight model (e.g. Gemini Flash, Claude Haiku)",
                            "tier_2_balanced": "Balanced production model (e.g. Claude Sonnet, Gemini Pro)",
                            "tier_3_heavy": "Heavy frontier reasoning model (e.g. o1, Claude Opus)"
                        }
                    }
                }
            }
        ),
        (
            "Gate 4: Pre-Execution Safety & Human-in-the-Loop Gating",
            {
                "model": "jev-latest",
                "state": {
                    "user_prompt": "Clean up the environment completely.",
                    "proposed_tool": "run_command",
                    "tool_args": {"command": "docker system prune -a --volumes -f && rm -rf /var/lib/testdb/*"}
                },
                "questions": {
                    "is_destructive": {
                        "type": "noul",
                        "instructions": "Will this proposed action delete, wipe, or permanently alter critical data or files?"
                    },
                    "is_external_impact": {
                        "type": "noul",
                        "instructions": "Does this action affect external production environments, deployment servers, or network APIs?"
                    },
                    "risk_score": {
                        "type": "score",
                        "instructions": "Assess the overall execution risk level.",
                        "criteria": [
                            "Safe: Read-only inspection or harmless calculation",
                            "Low: Easily reversible local edits or standard tests",
                            "Medium: Dependency installation or dev server restart",
                            "High: Service deployment, file deletion, or sensitive config edit",
                            "Critical: Database drop, disk wipe, or root command execution"
                        ]
                    },
                    "gating_action": {
                        "type": "choice",
                        "instructions": "What authorization policy should apply before executing this tool call?",
                        "criteria": {
                            "auto_approve": "Run immediately in background without blocking",
                            "notify_user_async": "Execute while notifying user in the activity stream",
                            "block_until_explicit_user_confirmation": "Halt and require explicit human-in-the-loop modal click"
                        }
                    }
                }
            }
        ),
        (
            "Gate 5: Fast Completion Verification",
            {
                "model": "jev-latest",
                "state": {
                    "request": "Fix the race condition in the Redis session store in src/auth.py and restart the dev server on port 6379.",
                    "actions_taken": [
                        {"tool": "replace_file_content", "file": "src/auth.py", "status": "success"},
                        {"tool": "run_command", "command": "pytest tests/test_session_race.py", "status": "3 passed"},
                        {"tool": "run_command", "command": "systemctl restart redis-dev", "status": "active (running)"}
                    ],
                    "final_output": "I resolved the race condition in src/auth.py using an atomic Redis Lua script, verified with tests, and restarted the server on port 6379."
                },
                "questions": {
                    "is_task_satisfied": {
                        "type": "noul",
                        "instructions": "Did the actions taken and final output completely fulfill the user's original request?"
                    },
                    "quality_score": {
                        "type": "score",
                        "instructions": "Rate the completion quality.",
                        "criteria": [
                            "Failed: Did not perform the requested task",
                            "Partial: Attempted but left errors or missed instructions",
                            "Adequate: Completed core instruction without extra validation",
                            "Robust: Fully implemented and verified with tests or diagnostics"
                        ]
                    },
                    "next_step": {
                        "type": "choice",
                        "instructions": "Determine the next agent loop action.",
                        "criteria": {
                            "finish_and_respond": "Task is complete; present final response to user",
                            "retry_with_correction": "Task incomplete or failed; re-prompt model to fix defects",
                            "ask_clarification": "Ambiguity encountered; ask user for direction"
                        }
                    }
                }
            }
        ),
        (
            "Gate 6: Computer Use & Desktop GUI Action Alignment",
            {
                "model": "jev-latest",
                "state": {
                    "user_goal": "Open Chrome and login to AWS console.",
                    "screen_context": "Desktop with dock visible. Chrome icon present at (45, 1020). No browser currently open.",
                    "proposed_action": {"action": "mouse_click", "coordinates": [45, 1020], "target": "Chrome Dock Icon"}
                },
                "questions": {
                    "action_alignment": {
                        "type": "noul",
                        "instructions": "Does this proposed GUI action directly advance the stated goal on the current screen?"
                    },
                    "requires_human_guidance": {
                        "type": "noul",
                        "instructions": "Is the GUI in an unexpected state (e.g. CAPTCHA, 2FA prompt, crash modal) requiring human takeover?"
                    },
                    "ambiguity_score": {
                        "type": "score",
                        "instructions": "Assess the visual ambiguity of the click/interaction target.",
                        "criteria": [
                            "Clear: Unique, high-contrast button/element clearly identified",
                            "Low: Slight variation but recognizable target",
                            "Moderate: Multiple similar elements on screen",
                            "High: Target occluded or requires scrolling",
                            "Severe: Target not visible on current screen"
                        ]
                    },
                    "execution_strategy": {
                        "type": "choice",
                        "instructions": "How should the workstation execute this computer use step?",
                        "criteria": {
                            "proceed_with_mouse_keyboard": "Dispatch mouse click / keystrokes immediately",
                            "wait_for_render": "Delay action for 500ms to allow UI animation to settle",
                            "abort_and_ask_user": "Pause computer use and request human intervention"
                        }
                    }
                }
            }
        )
    ]

    for title, payload in gates:
        print(f"\n### {title}")
        print("```bash")
        print("curl -X POST https://api.typesafe.ai/v1/systemone \\")
        print("  -H \"Authorization: Bearer $TYPESAFE_API_KEY\" \\")
        print("  -H \"Content-Type: application/json\" \\")
        payload_json = json.dumps(payload, indent=2)
        # Escape single quotes if any
        print(f"  -d '{payload_json}'")
        print("```")


def run_interactive():
    """Allows user to test custom prompts interactively against the Jev decision gates."""
    engine = OdysseusDecisionEngine()
    print("\n" + "=" * 80)
    print("💬 INTERACTIVE ODYSSEUS WORKSTATION TESTER")
    print("=" * 80)
    print("Enter any prompt to evaluate it through Jev's routing and safety gates (or 'exit' to quit).\n")

    while True:
        try:
            prompt = input("\n📝 Enter workstation prompt: ").strip()
            if not prompt or prompt.lower() in ["exit", "quit", "q"]:
                break

            print("\n  🔍 Analyzing with Jev System One...")
            
            # Run Gate 2: Tool selection
            tool_eval = engine.select_tools_and_domain(prompt, "Interactive Terminal Session")
            print(f"  ⚙️  [Domain & Tools] ({tool_eval['latency_ms']}ms)")
            print(f"     Domain: {tool_eval['primary_domain']}")
            print(f"     Pruned Tool Inventory: {tool_eval['pruned_active_tools']}")

            # Run Gate 3: Routing
            route_eval = engine.route_model_by_difficulty(prompt, "Interactive Terminal Session")
            print(f"  🎯 [Model Routing] ({route_eval['latency_ms']}ms)")
            print(f"     Complexity: {route_eval['complexity_score']:.1f}/4.0 -> {route_eval['recommended_tier']}")

            # Run Gate 4: Safety Check
            safety_eval = engine.evaluate_pre_execution_safety(prompt, "simulated_action", {"input": prompt})
            print(f"  🛡️  [Safety Gating] ({safety_eval['latency_ms']}ms)")
            print(f"     Risk Score: {safety_eval['risk_score']:.1f}/4.0 | Destructive prob: {safety_eval['is_destructive_prob']}")
            print(f"     Gating Action: >>> {safety_eval['gating_decision']} <<<")

        except (KeyboardInterrupt, EOFError):
            print("\nExiting interactive mode.")
            break


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Odysseus Workstation Jev Simulation")
    parser.add_argument("--dump-payloads", action="store_true", help="Print exact cURL payloads for all gates")
    parser.add_argument("--interactive", action="store_true", help="Run interactive prompt tester")
    parser.add_argument("--backend", choices=["llama", "jev", "mock"], default=os.getenv("DECISION_BACKEND", "llama"))
    parser.add_argument("--llama-url", default=os.getenv("LLAMA_BASE_URL", "http://127.0.0.1:8080"))
    args = parser.parse_args()
    os.environ["DECISION_BACKEND"] = args.backend
    os.environ["LLAMA_BASE_URL"] = args.llama_url

    try:
        if args.dump_payloads:
            print_payloads()
        elif args.interactive:
            run_interactive()
        else:
            run_simulation()
    except (RuntimeError, ValueError, KeyError, OSError) as error:
        parser.exit(2, f"Decision backend failed: {error}\n")
