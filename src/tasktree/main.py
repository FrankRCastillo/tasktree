#!/usr/bin/env python3
"""
tasktree: A strictly local, contract-first recursive reasoning harness for LLMs.

Every node is an agent that either does its task itself (worker) or splits it
across subordinates (manager). Context flows down, reports flow up: a worker
sees its chain of command's intent, the shared invariants, and what its peers
are doing, so its work fits beside theirs. A manager never rewrites that work;
it judges its subordinates from their summaries and objective measurements,
sends targeted revisions, and its deliverable is their content joined in order.
No single model call ever has to hold the whole project.
"""

import argparse
import ast
import configparser
import dataclasses
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Literal

from importlib.resources import files

from openai import OpenAI
from pydantic import BaseModel, Field
from rich.console import Console
from rich.live import Live
from rich.text import Text

# How much of the preceding section a worker sees so its content can continue from it
PRECEDING_TAIL_CHARS = 1500

# Output budget estimates. Deliberately conservative: overshooting the budget costs a whole wasted call.
CHARS_PER_TOKEN = 3.5          # prompt size estimate for English prose and code
WORDS_PER_TOKEN = 0.75         # deliverable words per output token
RESPONSE_OVERHEAD_TOKENS = 800  # JSON keys, the reasoning and summary fields
THINKING_RESERVE = 0.5         # share of the budget left for hidden reasoning when thinking is on
MIN_RESPONSE_TOKENS = 256


# ---------------------------------------------------------------------------
# Structured Output Schemas (reasoning first so the model can think before acting)
# ---------------------------------------------------------------------------

class SubtaskSpec(BaseModel):
    title: str
    description: str
    test_criteria: str
    depends_on: List[int] = Field(
        default_factory=list,
        description="0-based indices into this subtasks list (the first sibling is 0) that must finish first."
    )
    file_path: Optional[str] = None
    verify_command: Optional[str] = None
    read_files: List[str] = Field(default_factory=list)


class AgentResponse(BaseModel):
    reasoning: str = Field(description="Brief explanation of why you chose worker or manager mode.")
    mode: Literal["worker", "manager"]
    shared_invariants: str = Field("", description="(manager only) Global rules or conventions all subtasks must follow.")
    subtasks: List[SubtaskSpec] = Field(default_factory=list, description="(manager only) Ordered list of child subtasks.")
    content: str = Field("", description="(worker only) Complete text or source code of the deliverable.")
    summary: str = Field("", description="(worker only) Concise report of what was produced, for your manager.")
    file_path: Optional[str] = Field(None, description="(worker only, optional) Relative workspace path if writing a named file.")
    verify_command: Optional[str] = Field(None, description="(worker only, optional) Shell command to verify code/script output.")


class Revision(BaseModel):
    node_id: str
    feedback: str


class ManagerReviewResponse(BaseModel):
    reasoning: str = Field(description="Evaluation of the subordinates' reports against your task and test criteria.")
    approved: bool
    summary: str = Field(description="Concise report of what your branch accomplished, to pass up to your manager.")
    revisions: List[Revision] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# In-Memory Task Tree Node
# ---------------------------------------------------------------------------
@dataclass
class TaskNode:
    id: str
    title: str
    description: str
    test_criteria: str
    parent_id: Optional[str] = None
    depth: int = 0
    scope_weight: float = 100.0
    status: str = "PENDING"
    depends_on: List[str] = field(default_factory=list)
    children_ids: List[str] = field(default_factory=list)
    read_files: List[str] = field(default_factory=list)
    file_path: Optional[str] = None
    verify_command: Optional[str] = None
    shared_invariants: str = ""
    output: str = ""  # Worker: its content. Manager: its children's aggregated content.
    summary: str = ""
    verify_status: str = ""  # "", "passed" or "failed"
    verify_output: str = ""
    last_feedback: str = ""
    live_tokens: int = 0


# ---------------------------------------------------------------------------
# Helper Utilities (AST Stubbing, Git Checkpoints, JSON Parsing)
# ---------------------------------------------------------------------------

def extract_python_stub(file_path: Path) -> str:
    """Strips a Python file down to imports, class/function signatures, and docstrings."""
    if not file_path.exists():
        return ""
    source = file_path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source[:1500]

    lines: List[str] = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            lines.append(ast.unparse(node))
        elif isinstance(node, ast.ClassDef):
            bases = ", ".join(ast.unparse(b) for b in node.bases)
            lines.append(f"\nclass {node.name}({bases}):" if bases else f"\nclass {node.name}:")
            doc = ast.get_docstring(node)
            if doc:
                lines.append(f'    """{doc}"""')
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    args = ast.unparse(item.args)
                    ret = f" -> {ast.unparse(item.returns)}" if item.returns else ""
                    prefix = "async def" if isinstance(item, ast.AsyncFunctionDef) else "def"
                    lines.append(f"    {prefix} {item.name}({args}){ret}: ...")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = ast.unparse(node.args)
            ret = f" -> {ast.unparse(node.returns)}" if node.returns else ""
            prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
            lines.append(f"\n{prefix} {node.name}({args}){ret}: ...")
            doc = ast.get_docstring(node)
            if doc:
                lines.append(f'    """{doc}"""')
    return "\n".join(lines).strip()

def get_directory_tree(root: Path, max_files: int = 150) -> str:
    """Returns a compact list of relative file paths, skipping noise like .git and node_modules."""
    ignore = {".git", "__pycache__", ".venv", "venv", "node_modules", ".mypy_cache", ".pytest_cache"}
    paths: List[str] = []
    for path in sorted(root.rglob("*")):
        if any(part in ignore for part in path.parts):
            continue
        if path.is_file():
            paths.append(str(path.relative_to(root)))
            if len(paths) >= max_files:
                paths.append("... (truncated)")
                break
    return "\n".join(paths) if paths else "(empty directory)"


def resolve_within(root: Path, rel: str) -> Optional[Path]:
    """Resolves rel under root, or returns None if it would escape root (via '..', an absolute path, or a symlink)."""
    root = root.resolve()
    target = (root / rel).resolve()
    return target if target.is_relative_to(root) and target != root else None


def read_context_files(context_dir: Optional[Path], rel_paths: List[str], max_chars_per_file: int = 8000) -> str:
    if not context_dir or not rel_paths:
        return "None."
    blocks = []
    for rel in rel_paths:
        target = resolve_within(context_dir, rel)
        if target is None or not target.is_file():
            continue
        try:
            text = target.read_text(encoding="utf-8", errors="replace")
            if len(text) > max_chars_per_file:
                text = text[:max_chars_per_file] + "\n... [truncated]"
            blocks.append(f"--- File: {rel} ---\n{text}")
        except Exception as e:
            blocks.append(f"--- File: {rel} (unreadable: {e}) ---")
    return "\n\n".join(blocks) if blocks else "None."

def word_count(text: str) -> int:
    """Matches `wc -w`: whitespace-separated tokens."""
    return len(text.split())

def init_git_workspace(workspace: Path) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    if not (workspace / ".git").exists():
        subprocess.run(["git", "init"], cwd=workspace, capture_output=True)
        subprocess.run(["git", "config", "user.name", "tasktree"], cwd=workspace, capture_output=True)
        subprocess.run(["git", "config", "user.email", "tasktree@local"], cwd=workspace, capture_output=True)


def git_commit(workspace: Path, message: str) -> None:
    subprocess.run(["git", "add", "."], cwd=workspace, capture_output=True)
    subprocess.run(["git", "commit", "-m", message], cwd=workspace, capture_output=True)


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def schema_text(schema: type[BaseModel]) -> str:
    return json.dumps(schema.model_json_schema(), indent=2)


def detect_context_window(base_url: str) -> Optional[int]:
    """Asks a llama.cpp server for its per-slot context size. Returns None for servers without /props."""
    root = base_url.rstrip("/").removesuffix("/v1")
    try:
        with urllib.request.urlopen(f"{root}/props", timeout=3) as resp:
            n_ctx = json.load(resp).get("default_generation_settings", {}).get("n_ctx")
        return int(n_ctx) if n_ctx else None
    except Exception:
        return None


class OutputTruncated(Exception):
    """The response hit its token limit before finishing, so the deliverable is too big for one pass."""

    def __init__(self, response_tokens: int):
        self.words = int(response_tokens * WORDS_PER_TOKEN)
        super().__init__(f"response cut off at its token limit after about {self.words} words")


class RunLog:
    """Append-only, human-readable stream of every model call, written as it arrives so it can be followed with tail -f."""

    def __init__(self, path: Optional[Path]):
        self.path = path
        self.file = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.file = open(path, "a", encoding="utf-8")
        self.current_stream: Optional[str] = None

    def write(self, text: str) -> None:
        if self.file is not None:
            self.file.write(text)
            self.file.flush()

    def line(self, text: str) -> None:
        self.current_stream = None
        self.write(f"{text}\n")

    def stream(self, kind: str, text: str) -> None:
        """Appends streamed text, with a header whenever the stream switches between prompt/thinking/response."""
        if kind != self.current_stream:
            self.write(f"\n--- {kind} ---\n")
            self.current_stream = kind
        self.write(text)

    def close(self) -> None:
        if self.file is not None:
            self.file.close()
            self.file = None


def extract_json_payload(raw_text: str) -> Dict[str, Any]:
    """Safely extracts JSON even if a local model wraps it in markdown fences."""
    cleaned = raw_text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1)
    else:
        brace_match = re.search(r"(\{.*\})", cleaned, re.DOTALL)
        if brace_match:
            cleaned = brace_match.group(1)
    return json.loads(cleaned)


# ---------------------------------------------------------------------------
# Core Harness
# ---------------------------------------------------------------------------

class TaskTreeHarness:
    def __init__(
        self,
        config: Dict[str, Any],
        root_prompt: Optional[str],
        output_file: Optional[str] = None,
        context_dir: Optional[str] = None,
        resume: bool = False,
    ):
        self.config = config
        self.workspace = Path(config["workspace"]).resolve()
        self.output_file = output_file
        self.checkpoint_path = self.workspace / ".tasktree_state.json"

        self.client = OpenAI(
            base_url=config["base_url"],
            api_key=config["api_key"],
        )
        self.nodes: Dict[str, TaskNode] = {}
        self.root_id = "1"
        self.context_dir = Path(context_dir).resolve() if context_dir else None
        self.context_window = config["context_window"] or detect_context_window(config["base_url"])
        self.log = RunLog(None)

        if resume:
            self.load_checkpoint()
        else:
            if not root_prompt:
                raise ValueError("root_prompt is required unless resuming")
            self.root_prompt = root_prompt
            self.nodes[self.root_id] = TaskNode(
                id=self.root_id,
                title=root_prompt,
                description=root_prompt,
                test_criteria=f"Fully and accurately satisfies the user's root request: {root_prompt}",
                depth=0,
                scope_weight=100.0,
            )

        init_git_workspace(self.workspace)
        if config["log_file"]:
            log_path = Path(config["log_file"])
            log_path = log_path if log_path.is_absolute() else self.workspace / log_path
            self.exclude_from_workspace_git(log_path)
            self.log = RunLog(log_path)
        self.log.line(
            f"\n##### [{timestamp()}] tasktree {'resumed' if resume else 'started'}: {self.root_prompt}\n"
            f"##### model {config['model']} | context window {self.context_window or 'unknown'} "
            f"| max_tokens {config['max_tokens']} | thinking {'on' if config['enable_thinking'] else 'off'}"
        )

    def exclude_from_workspace_git(self, path: Path) -> None:
        """Keeps the log out of the workspace's per-node commits."""
        try:
            rel = path.resolve().relative_to(self.workspace)
        except ValueError:
            return
        exclude = self.workspace / ".git" / "info" / "exclude"
        if not exclude.parent.is_dir():
            return
        existing = exclude.read_text(encoding="utf-8") if exclude.exists() else ""
        if str(rel) not in existing.splitlines():
            with open(exclude, "a", encoding="utf-8") as f:
                f.write(("" if existing.endswith("\n") or not existing else "\n") + f"{rel}\n")

    def save_checkpoint(self) -> None:
        data = {
            "root_prompt": self.root_prompt,
            "context_dir": str(self.context_dir) if self.context_dir else None,
            "nodes": {nid: asdict(node) for nid, node in self.nodes.items()},
        }
        self.checkpoint_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load_checkpoint(self) -> None:
        """Restores a saved run. DONE nodes are kept; everything else is reset to PENDING and re-run.
        Managers keep the subtasks they already created."""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint found at {self.checkpoint_path}")
        data = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
        if "nodes" not in data:
            raise ValueError(f"{self.checkpoint_path} predates resume support and cannot be resumed")
        self.root_prompt = data["root_prompt"]
        if self.context_dir is None and data.get("context_dir"):
            self.context_dir = Path(data["context_dir"])
        known = {f.name for f in dataclasses.fields(TaskNode)}
        self.nodes = {
            nid: TaskNode(**{k: v for k, v in fields.items() if k in known})
            for nid, fields in data["nodes"].items()
        }
        for node in self.nodes.values():
            node.live_tokens = 0
            if node.status != "DONE":
                node.status = "PENDING"
                # Ancestors aggregated incomplete work, so they must re-run too
                for ancestor in self.ancestors(node):
                    ancestor.status = "PENDING"

    def warn(self, live: Optional[Live], message: str) -> None:
        self.log.line(f"[{timestamp()}] warning: {message}")
        if live is not None:
            live.console.print(f"[yellow]{message}[/yellow]")

    def log_outcome(self, node: TaskNode) -> None:
        self.log.line(f"[{timestamp()}] node {node.id} {node.status}: {node.title}")

    # --- Output Budget ---

    def output_token_budget(self, prompt_chars: int) -> int:
        """Tokens the response can use: max_tokens, capped by what the context window has left."""
        budget = self.config["max_tokens"]
        if self.context_window:
            budget = min(budget, self.context_window - int(prompt_chars / CHARS_PER_TOKEN))
        return max(budget, MIN_RESPONSE_TOKENS)

    def content_word_budget(self, prompt_chars: int) -> int:
        """Words of deliverable that fit in one response, after JSON overhead and room for thinking."""
        tokens = self.output_token_budget(prompt_chars) - RESPONSE_OVERHEAD_TOKENS
        if self.config["enable_thinking"]:
            tokens = int(tokens * (1 - THINKING_RESERVE))
        return max(int(tokens * WORDS_PER_TOKEN), 0)

    # --- Model Calls ---

    def call_model(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: type[BaseModel],
        label: str,
        temp: Optional[float] = None,
        node: Optional[TaskNode] = None,
        live: Optional[Live] = None,
    ) -> BaseModel:
        """Stateless single-turn call with streaming token counter, logging, and schema validation.
        Raises OutputTruncated if the response itself overflows its budget."""
        full_system = (
            f"{system_prompt}\n\n"
            f"You MUST respond with valid JSON strictly matching this schema:\n{schema_text(schema)}"
        )
        temperature = self.config["temperature"] if temp is None else temp
        thinking = self.config["enable_thinking"]

        last_err = None
        for _ in range(3):
            content, finish_reason, thinking_tokens, response_tokens = self.stream_completion(
                full_system, user_prompt, temperature, thinking, label, node, live
            )
            if finish_reason == "length":
                if thinking and thinking_tokens > response_tokens:
                    # Reasoning used up the budget before the answer: that says nothing about the
                    # task's size, so let the model answer directly instead
                    self.warn(live, f"{label} call for node {node.id if node else '?'} ran out of room while "
                                    f"thinking ({thinking_tokens} tokens); retrying with thinking off.")
                    thinking = False
                    continue
                raise OutputTruncated(response_tokens)
            try:
                parsed = extract_json_payload(content)
                return schema.model_validate(parsed)
            except Exception as e:
                last_err = e
                user_prompt += f"\n\nYour previous JSON was invalid ({e}). Return ONLY valid JSON matching the schema."
        raise RuntimeError(f"Model failed to return valid structured output: {last_err}")

    def stream_completion(
        self,
        full_system: str,
        user_prompt: str,
        temperature: float,
        thinking: bool,
        label: str,
        node: Optional[TaskNode],
        live: Optional[Live],
    ) -> tuple[str, Optional[str], int, int]:
        """Streams one completion. Returns (content, finish_reason, thinking tokens, response tokens);
        token counts are streamed chunks, which servers like llama.cpp send one token at a time."""
        max_tokens = self.output_token_budget(len(full_system) + len(user_prompt))
        # Only sent when disabling, so servers and templates without the option see an unchanged request
        extra_body = {} if thinking else {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}

        node_name = f"node {node.id}" if node else "harness"
        started = time.monotonic()
        self.log.line(
            f"\n===== [{timestamp()}] {node_name} | {label} | thinking {'on' if thinking else 'off'} "
            f"| max_tokens {max_tokens} ====="
        )
        self.log.stream("prompt", user_prompt)

        if node is not None:
            node.live_tokens = 0
        stream = self.client.chat.completions.create(
            model=self.config["model"],
            messages=[
                {"role": "system", "content": full_system},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            stream=True,
            **extra_body,
        )

        chunks: List[str] = []
        finish_reason = None
        thinking_tokens = response_tokens = 0
        for chunk in stream:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            finish_reason = getattr(choice, "finish_reason", None) or finish_reason
            delta = getattr(choice, "delta", None)
            if delta is None:
                continue
            # Thinking models stream hidden reasoning separately; count it so progress stays visible
            reasoning = getattr(delta, "reasoning_content", None)
            text = getattr(delta, "content", None)
            if reasoning:
                thinking_tokens += 1
                self.log.stream("thinking", reasoning)
            if text:
                response_tokens += 1
                chunks.append(text)
                self.log.stream("response", text)
            if (reasoning or text) and node is not None:
                node.live_tokens += 1
                if live is not None and node.live_tokens % 5 == 0:
                    live.update(self.render_tree())

        if node is not None:
            node.live_tokens = 0
            if live is not None:
                live.update(self.render_tree())
        self.log.line(
            f"\n===== end {node_name} | {label} | finish: {finish_reason} | ~{thinking_tokens} thinking + "
            f"~{response_tokens} response tokens | {time.monotonic() - started:.0f}s ====="
        )
        return "".join(chunks), finish_reason, thinking_tokens, response_tokens

    # --- Live Indented Number Renderer ---

    def render_tree(self) -> Text:
        status_styles = {
            "PENDING": "[dim]PENDING[/dim]",
            "WORKING": "[bold yellow]WORKING[/bold yellow]",
            "REVISING": "[bold orange1]REVISING[/bold orange1]",
            "TESTING": "[magenta]TESTING[/magenta]",
            "MANAGING": "[cyan]MANAGING[/cyan]",
            "REVIEWING": "[bold cyan]REVIEWING[/bold cyan]",
            "DONE": "[bold green]DONE[/bold green]",
            "FAILED": "[bold red]FAILED[/bold red]",
        }

        lines: List[str] = []

        def add_lines(node_id: str) -> None:
            node = self.nodes[node_id]
            tag = status_styles.get(node.status, node.status)

            # Show live token counter on active calls, or subordinate progress on managers
            if node.live_tokens > 0:
                tag = f"{tag} [dim]({node.live_tokens} tok)[/dim]"
            elif node.children_ids and node.status in ("PENDING", "MANAGING", "REVIEWING"):
                done_count = sum(1 for cid in node.children_ids if self.nodes[cid].status == "DONE")
                tag = f"{tag} [dim]({done_count}/{len(node.children_ids)})[/dim]"

            file_hint = f" [dim]({node.file_path})[/dim]" if node.file_path else ""

            lines.append(f"[bold]{node.id}.[/bold] {node.title}{file_hint}  {tag}")

            for cid in node.children_ids:
                add_lines(cid)

        add_lines(self.root_id)

        return Text.from_markup("\n".join(lines))

    # --- Context Flowing Down: intent, invariants, peers ---

    def ancestors(self, node: TaskNode) -> List[TaskNode]:
        """Returns the chain of command above node, root first."""
        chain: List[TaskNode] = []
        curr = node.parent_id
        while curr:
            chain.append(self.nodes[curr])
            curr = self.nodes[curr].parent_id
        return list(reversed(chain))

    def chain_of_command(self, node: TaskNode) -> str:
        chain = self.ancestors(node)
        if not chain:
            return "None (you are the root task)."
        return "\n".join(
            f"{a.id}. {a.title}\n   Intent: {a.description}\n   Success Criteria: {a.test_criteria}"
            for a in chain
        )

    def inherited_invariants(self, node: TaskNode) -> str:
        """Collects shared_invariants set by every ancestor, root first."""
        chain = [
            f"From {a.id} ({a.title}): {a.shared_invariants}"
            for a in self.ancestors(node)
            if a.shared_invariants
        ]
        return "\n".join(chain) if chain else "None."

    def sibling_overview(self, node: TaskNode) -> str:
        """Lists the node's peers with their scopes, and the reports of peers that have finished."""
        if not node.parent_id:
            return "None (you are the root task)."
        lines: List[str] = []
        for sid in self.nodes[node.parent_id].children_ids:
            s = self.nodes[sid]
            marker = "  <-- you" if sid == node.id else ""
            target = f" -> {s.file_path}" if s.file_path else ""
            lines.append(f"{s.id}. {s.title}{target} [{s.status}]{marker}\n   Scope: {s.description}")
            if sid != node.id and s.status == "DONE" and s.summary:
                lines.append(f"   Report: {s.summary}")
        return "\n".join(lines)

    def preceding_content(self, node: TaskNode) -> str:
        """Returns the end of the content that will sit directly before this node's in the aggregated
        output, crossing up the tree when the node is the first section of its branch."""
        curr = node
        while curr.parent_id:
            parent = self.nodes[curr.parent_id]
            siblings = parent.children_ids
            for sid in reversed(siblings[: siblings.index(curr.id)]):
                s = self.nodes[sid]
                if not s.file_path:
                    return s.output[-PRECEDING_TAIL_CHARS:]
            if parent.file_path:
                return ""  # The parent's aggregate starts a separate file
            curr = parent
        return ""

    def gather_dependency_stubs(self, node: TaskNode) -> str:
        """Pulls lightweight AST stubs (for code) or summaries (for prose) from completed dependencies."""
        if not node.depends_on:
            return "None."
        blocks = []
        for dep_id in node.depends_on:
            dep = self.nodes[dep_id]
            target = resolve_within(self.workspace, dep.file_path) if dep.file_path else None
            if target is not None and dep.file_path.endswith(".py"):
                stub = extract_python_stub(target)
                blocks.append(f"--- Interface Stub for {dep.file_path} ({dep.title}) ---\n{stub}")
            else:
                blocks.append(f"--- Summary of Dependency '{dep.title}' ---\n{dep.summary}")
        return "\n\n".join(blocks)

    def agent_system_prompt(self) -> str:
        # replace rather than format, so literal braces in a custom prompt don't break it
        return self.config["prompts"]["agent"].replace("{max_children}", str(self.config["max_children"]))

    def agent_prompt(self, node: TaskNode, feedback: str = "", problem: str = "") -> str:
        max_depth = self.config["max_depth"]
        sections = [
            f"Root Goal: {self.root_prompt}",
            f"Your Position: node {node.id}, depth {node.depth} of max {max_depth} | "
            f"Estimated Share of Total Scope: {node.scope_weight:.1f}%",
        ]
        if node.depth >= max_depth:
            sections.append('You are at the maximum depth: you MUST choose mode="worker".')
        sections += [
            f"Chain of Command (your superiors' intent, root first):\n{self.chain_of_command(node)}",
            f"Shared Invariants (must be honored):\n{self.inherited_invariants(node)}",
            f"Sibling Overview (your peers under the same manager):\n{self.sibling_overview(node)}",
            f"Current Task: {node.title}\n"
            f"Description: {node.description}\n"
            f"Test Criteria: {node.test_criteria}\n"
            f"Assigned File: {node.file_path or 'None'}\n"
            f"Assigned Verify Command: {node.verify_command or 'None'}",
            f"Dependencies (completed peers you build on):\n{self.gather_dependency_stubs(node)}",
        ]
        preceding = self.preceding_content(node)
        if preceding:
            sections.append(f"End of the Preceding Section (your content follows it directly):\n{preceding}")
        if self.context_dir:
            sections.append(f"Read-Only Context Directory Files:\n{get_directory_tree(self.context_dir)}")
        sections.append(f"Reference Files to Inspect:\n{read_context_files(self.context_dir, node.read_files)}")
        if (feedback or problem) and node.output:
            sections.append(f"Your Previous Submission:\n{node.output}")
        if feedback:
            sections.append(f"Revision Feedback from your manager (address this directly):\n{feedback}")
        if problem:
            sections.append(f"Problem With Your Previous Attempt (fix this):\n{problem}")

        # The budget depends on how much room the rest of the prompt leaves, so it is measured last
        prompt_chars = len(self.agent_system_prompt()) + len(schema_text(AgentResponse)) + len("\n\n".join(sections))
        words = self.content_word_budget(prompt_chars)
        if node.depth >= max_depth:
            consequence = "you cannot split, so keep your content within it."
        else:
            consequence = 'if the complete deliverable would need more, you cannot do it as a worker: choose mode="manager".'
        sections.insert(2, f"Output Budget: this response can hold at most about {words:,} words of content; {consequence}")
        return "\n\n".join(sections)

    # --- Running Nodes ---

    def run_node(self, node_id: str, live: Live, feedback: str = "") -> None:
        node = self.nodes[node_id]
        if node.status == "DONE" and not feedback:
            return  # Already completed (e.g. in a resumed run)
        if node.children_ids:
            self.run_manager(node, live, upstream_feedback=feedback)
        else:
            self.run_agent(node, live, feedback)

    def run_agent(self, node: TaskNode, live: Live, feedback: str = "") -> None:
        """Lets the node either do its task itself (worker) or split it (manager)."""
        problem = ""
        for attempt in range(1, self.config["max_retries"] + 1):
            node.status = "REVISING" if (feedback or problem) else "WORKING"
            live.update(self.render_tree())
            try:
                response: AgentResponse = self.call_model(
                    self.agent_system_prompt(),
                    self.agent_prompt(node, feedback, problem),
                    AgentResponse,
                    label="agent",
                    node=node,
                    live=live,
                )
                if response.mode == "manager":
                    if node.depth >= self.config["max_depth"]:
                        raise ValueError('You are at the maximum depth and must use mode="worker".')
                    if not response.subtasks:
                        raise ValueError("Manager mode requires at least one subtask.")
                    self.spawn_children(node, response, live)
                    break
                problem = self.finish_worker(node, response, live)
                if not problem:
                    return
            except OutputTruncated as e:
                # Overflowing the budget is evidence the task is too big for one pass
                if node.depth < self.config["max_depth"]:
                    problem = (
                        f"Your previous response was cut off after about {e.words:,} words: this task is too "
                        'large for one pass. You MUST choose mode="manager" and split it into subtasks that each fit.'
                    )
                else:
                    problem = (
                        f"Your previous response was cut off after about {e.words:,} words, and you are at the "
                        "maximum depth so you cannot split. Cover the essentials concisely so it fits your Output Budget."
                    )
                self.warn(live, f"Node {node.id} attempt {attempt}: {e}")
            except Exception as e:
                problem = f"Your previous attempt was rejected: {e}"
                self.warn(live, f"Node {node.id} attempt {attempt}: {e}")
        else:
            node.status = "FAILED"
            node.last_feedback = problem
            self.save_checkpoint()
            self.log_outcome(node)
            live.update(self.render_tree())
            return

        self.run_manager(node, live)

    def finish_worker(self, node: TaskNode, response: AgentResponse, live: Live) -> str:
        """Writes and verifies a worker's deliverable. Returns the problem to fix, or "" once DONE."""
        if not response.content.strip():
            raise ValueError('mode="worker" requires the complete deliverable in `content`.')

        file_path = node.file_path or response.file_path
        target = None
        if file_path:
            target = resolve_within(self.workspace, file_path)
            if target is None:
                raise ValueError(f"file_path {file_path!r} is outside the workspace; use a relative path.")
        node.file_path = file_path
        node.verify_command = node.verify_command or response.verify_command
        node.output = response.content
        node.summary = response.summary

        if target is not None:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(node.output, encoding="utf-8")

        node.verify_status, node.verify_output = "", ""
        if node.verify_command:
            node.status = "TESTING"
            live.update(self.render_tree())
            passed, output = self.run_verify_command(node.verify_command)
            node.verify_status = "passed" if passed else "failed"
            node.verify_output = output
            if not passed:
                return f"Verify command `{node.verify_command}` failed:\n{output}"

        node.status = "DONE"
        node.last_feedback = ""
        git_commit(self.workspace, f"Complete node {node.id}: {node.title}")
        self.save_checkpoint()
        self.log_outcome(node)
        live.update(self.render_tree())
        return ""

    def spawn_children(self, node: TaskNode, response: AgentResponse, live: Live) -> None:
        subtasks = response.subtasks[: self.config["max_children"]]
        child_weight = node.scope_weight / len(subtasks)
        node.shared_invariants = response.shared_invariants
        node.output = ""
        created_ids: List[str] = []
        for idx, spec in enumerate(subtasks, start=1):
            cid = f"{node.id}.{idx}"
            created_ids.append(cid)
            if spec.file_path and resolve_within(self.workspace, spec.file_path) is None:
                self.warn(live, f"Node {cid}: dropping file_path {spec.file_path!r} (outside the workspace).")
                spec.file_path = None
            self.nodes[cid] = TaskNode(
                id=cid,
                title=spec.title,
                description=spec.description,
                test_criteria=spec.test_criteria,
                parent_id=node.id,
                depth=node.depth + 1,
                scope_weight=child_weight,
                file_path=spec.file_path,
                read_files=spec.read_files,
                verify_command=spec.verify_command,
            )

        # Resolve sibling DAG dependencies
        for idx, spec in enumerate(subtasks):
            cid = created_ids[idx]
            for dep_idx in spec.depends_on:
                if 0 <= dep_idx < len(created_ids) and dep_idx != idx:
                    self.nodes[cid].depends_on.append(created_ids[dep_idx])

        node.children_ids = created_ids
        self.save_checkpoint()
        live.update(self.render_tree())

    def run_children(self, node: TaskNode, live: Live) -> None:
        """Runs unfinished children, respecting the sibling depends_on DAG."""
        remaining = [cid for cid in node.children_ids if self.nodes[cid].status != "DONE"]
        while remaining:
            progressed = False
            for cid in list(remaining):
                child = self.nodes[cid]
                if all(self.nodes[d].status == "DONE" for d in child.depends_on):
                    self.run_node(cid, live)
                    remaining.remove(cid)
                    progressed = True
            if not progressed:
                # Dependencies failed or form a cycle; run the next child anyway
                self.run_node(remaining.pop(0), live)

    # --- Reports Flowing Up: the manager judges and aggregates ---

    def child_report(self, child: TaskNode) -> str:
        if child.verify_status == "passed":
            verification = f"passed (`{child.verify_command}`)"
        elif child.verify_status == "failed":
            verification = f"FAILED (`{child.verify_command}`):\n{child.verify_output[-800:]}"
        else:
            verification = "none"
        lines = [
            f"--- {child.id}. {child.title} [{child.status}] ---",
            f"Assigned Scope: {child.description}",
            f"Test Criteria: {child.test_criteria}",
            f"Word Count (wc -w): {word_count(child.output)}",
            f"Delivered To: {f'file {child.file_path}' if child.file_path else 'your aggregated output'}",
            f"Verification: {verification}",
            f"Summary: {child.summary or '(none)'}",
        ]
        if child.status == "FAILED" and child.last_feedback:
            lines.append(f"Failure: {child.last_feedback[-800:]}")
        return "\n".join(lines)

    def manager_review_prompt(self, node: TaskNode, upstream_feedback: str) -> str:
        sections = [
            f"Root Goal: {self.root_prompt}",
            f"Current Task: {node.title}\nDescription: {node.description}\nTest Criteria: {node.test_criteria}",
        ]
        if upstream_feedback:
            sections.append(f"Feedback From Your Own Manager (your branch was sent back):\n{upstream_feedback}")
        reports = "\n\n".join(self.child_report(self.nodes[cid]) for cid in node.children_ids)
        sections.append(f"Subordinate Reports (use these exact node_id values for revisions):\n{reports}")
        sections.append(f"Combined Word Count of Aggregated Content (wc -w): {word_count(self.aggregate(node))}")
        return "\n\n".join(sections)

    def aggregate(self, node: TaskNode) -> str:
        """Joins the children's content in order. Children that wrote their own file deliver it there."""
        parts = [
            self.nodes[cid].output.strip()
            for cid in node.children_ids
            if not self.nodes[cid].file_path and self.nodes[cid].output.strip()
        ]
        return "\n\n".join(parts)

    def run_manager(self, node: TaskNode, live: Live, upstream_feedback: str = "") -> None:
        node.status = "MANAGING"
        live.update(self.render_tree())
        self.run_children(node, live)

        rounds = self.config["max_review_rounds"]
        summary = " ".join(self.nodes[cid].summary for cid in node.children_ids if self.nodes[cid].summary)
        approved = rounds <= 0
        critique = ""
        for round_idx in range(rounds):
            node.status = "REVIEWING"
            live.update(self.render_tree())
            try:
                review: ManagerReviewResponse = self.call_model(
                    self.config["prompts"]["manager_review"],
                    self.manager_review_prompt(node, upstream_feedback),
                    ManagerReviewResponse,
                    label="review",
                    node=node,
                    live=live,
                )
            except Exception as e:
                self.warn(live, f"Review failed for node {node.id}; accepting subordinates' work as is. ({e})")
                approved = True
                break

            summary = review.summary or summary
            if review.approved:
                approved = True
                break

            critique = "; ".join(f"{r.node_id}: {r.feedback}" for r in review.revisions) or review.reasoning
            targets = [r for r in review.revisions if r.node_id in node.children_ids]
            if not targets:
                self.warn(live, f"Node {node.id} rejected its subordinates' work without actionable revisions.")
                break
            if round_idx == rounds - 1:
                self.warn(live, f"Node {node.id} not approved after {rounds} review round(s).")
                break

            node.status = "MANAGING"
            live.update(self.render_tree())
            for revision in targets:
                self.run_node(revision.node_id, live, feedback=revision.feedback)

        node.output = self.aggregate(node)
        node.summary = summary
        if node.file_path and node.output:
            target = resolve_within(self.workspace, node.file_path)
            if target is not None:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(node.output, encoding="utf-8")

        if approved:
            node.status = "DONE"
            node.last_feedback = ""
            git_commit(self.workspace, f"Complete node {node.id}: {node.title}")
        else:
            node.status = "FAILED"
            node.last_feedback = f"Subordinates' work not approved: {critique}"
        self.save_checkpoint()
        self.log_outcome(node)
        live.update(self.render_tree())

    def run_verify_command(self, command: str) -> tuple[bool, str]:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.workspace) + os.pathsep + env.get("PYTHONPATH", "")
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=self.workspace,
                capture_output=True,
                text=True,
                env=env,
                timeout=60,
            )
        except subprocess.TimeoutExpired:
            return False, f"Verify command timed out after 60s: {command}"
        passed = result.returncode == 0
        output = (result.stdout + "\n" + result.stderr).strip()
        return passed, output[-2000:]

    def run(self) -> None:
        console = Console()
        if self.log.path is not None:
            console.print(f"[dim]Streaming model calls to {self.log.path}  (follow with: tail -f {self.log.path})[/dim]")
        try:
            with Live(self.render_tree(), console=console, refresh_per_second=6) as live:
                self.run_node(self.root_id, live)

        except KeyboardInterrupt:
            self.save_checkpoint()
            self.log.line(f"\n##### [{timestamp()}] interrupted")
            self.log.close()
            console.print(
                f"\n[yellow]Interrupted. State saved to {self.checkpoint_path}; "
                f"continue with: tasktree --resume -w {self.workspace}[/yellow]"
            )
            sys.exit(130)

        self.log.line(f"\n##### [{timestamp()}] finished: root {self.nodes[self.root_id].status}")
        self.log.close()

        failed_nodes = [n for n in self.nodes.values() if n.status == "FAILED"]
        for fn in failed_nodes:
            console.print(f"\n[bold red]Node {fn.id} ({fn.title}) failed:[/bold red] {fn.last_feedback}")
        if failed_nodes:
            console.print(f"\n[yellow]{len(failed_nodes)} node(s) failed; rerun them with: tasktree --resume -w {self.workspace}[/yellow]")

        root = self.nodes[self.root_id]
        if root.summary:
            console.print(f"\n[bold]Summary:[/bold] {root.summary}")
        if self.output_file and root.output:
            out_path = Path(self.output_file)
            out_path.write_text(root.output, encoding="utf-8")
            console.print(f"\n[bold green]Saved final output to {out_path}[/bold green]")

# ---------------------------------------------------------------------------
# CLI & Config Loading
# ---------------------------------------------------------------------------
def get_default_rc() -> str:
    return files("tasktree").joinpath("default_tasktreerc").read_text(encoding="utf-8")

def load_config(rc_path: Path) -> Dict[str, Any]:
    parser = configparser.ConfigParser(interpolation=None)
    parser.read_string(get_default_rc())
    if rc_path.exists():
        parser.read(rc_path)

    return {
        "base_url": parser["model"]["base_url"],
        "api_key": parser["model"]["api_key"],
        "model": parser["model"]["model"],
        "temperature": parser["model"].getfloat("temperature"),
        "max_tokens": parser["model"].getint("max_tokens"),
        "enable_thinking": parser["model"].getboolean("enable_thinking"),
        "context_window": parser["model"].getint("context_window"),
        "max_depth": parser["harness"].getint("max_depth"),
        "max_children": parser["harness"].getint("max_children"),
        "max_retries": parser["harness"].getint("max_retries"),
        "max_review_rounds": parser["harness"].getint("max_review_rounds"),
        "workspace": parser["harness"]["workspace"],
        "log_file": parser["harness"]["log_file"].strip(),
        "prompts": {
            "agent": parser["prompts"]["agent"].strip(),
            "manager_review": parser["prompts"]["manager_review"].strip(),
        },
    }

def find_unknown_rc_keys(rc_path: Path) -> List[str]:
    """Lists sections and keys in the user's rc that tasktree does not recognize (and so ignores)."""
    if not rc_path.exists():
        return []
    defaults = configparser.ConfigParser(interpolation=None)
    defaults.read_string(get_default_rc())
    user = configparser.ConfigParser(interpolation=None)
    user.read(rc_path)
    unknown: List[str] = []
    for section in user.sections():
        if not defaults.has_section(section):
            unknown.append(f"[{section}]")
            continue
        unknown += [f"[{section}] {key}" for key in user[section] if not defaults.has_option(section, key)]
    return unknown

def main() -> None:
    argp = argparse.ArgumentParser(description="Recursive local-context task tree harness.")
    argp.add_argument("prompt", nargs="?", help="Root task to plan and execute.")
    argp.add_argument("-o", "--output", help="Optional file path to save the root node's final output.")
    argp.add_argument("-w", "--workspace", help="Override workspace directory.")
    argp.add_argument("-c", "--context", help="Optional read-only directory (e.g. an existing repo) to analyze.")
    argp.add_argument("--rc", default=str(Path.home() / ".tasktreerc"), help="Path to .tasktreerc file.")
    argp.add_argument("--init-config", action="store_true", help="Write or overwrite .tasktreerc with default settings and prompts.")
    argp.add_argument("--resume", action="store_true", help="Resume the run checkpointed in the workspace, re-running any node not DONE.")
    argp.add_argument("--log", help="Where to stream model calls (default: .tasktree.log in the workspace; '' disables).")
    args = argp.parse_args()

    rc_path = Path(args.rc)
    if args.init_config:
        rc_path.write_text(get_default_rc().strip() + "\n", encoding="utf-8")
        print(f"Wrote default configuration to {rc_path}")
        return

    if args.resume and args.prompt:
        argp.error("--resume takes the prompt from the checkpoint; do not pass one")
    if not args.prompt and not args.resume:
        argp.error("the following arguments are required: prompt")

    for key in find_unknown_rc_keys(rc_path):
        print(f"warning: {rc_path}: unrecognized setting {key} is ignored", file=sys.stderr)

    config = load_config(rc_path)
    if args.workspace:
        config["workspace"] = args.workspace
    if args.log is not None:
        # Relative to where the command runs, unlike the rc setting, which is relative to the workspace
        config["log_file"] = str(Path(args.log).resolve()) if args.log else ""

    try:
        harness = TaskTreeHarness(
            config, args.prompt, output_file=args.output, context_dir=args.context, resume=args.resume
        )
    except (FileNotFoundError, ValueError) as e:
        argp.error(str(e))
    harness.run()

if __name__ == "__main__":
    main()
