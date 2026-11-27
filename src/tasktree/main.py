#!/usr/bin/env python3
"""
tasktree: A strictly local, contract-first recursive reasoning harness for LLMs.
"""

import argparse
import ast
import configparser
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any

from importlib.resources import files

from openai import OpenAI
from pydantic import BaseModel, Field
from rich.console import Console
from rich.live import Live
from rich.text import Text


# ---------------------------------------------------------------------------
# Structured Output Schemas (reasoning first so Qwen can think before acting)
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


class PlanResponse(BaseModel):
    reasoning: str
    is_leaf: bool
    shared_invariants: str = ""
    subtasks: List[SubtaskSpec] = Field(default_factory=list)

class ExecutionResponse(BaseModel):
    reasoning: str = Field(description="Step-by-step reasoning to solve the task.")
    output: str = Field(description="The final prose, analysis, script, or source code for this node.")
    summary: str = Field(description="Tight, 2-4 sentence summary of what was built or concluded, for the parent.")
    assumptions: List[str] = Field(
        default_factory=list,
        description="Short list of key assumptions or caveats made so the parent can check for conflicts."
    )

class JudgeResponse(BaseModel):
    reasoning: str = Field(description="Strict evaluation of the output against the test criteria.")
    passed: bool = Field(description="True only if all test criteria are satisfied.")
    feedback: str = Field(description="Specific, actionable feedback if passed is False.")

class PlanReviewResponse(BaseModel):
    approved: bool
    critique: str
    flagged_node_ids: List[str] = Field(default_factory=list)

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
    output: str = ""
    summary: str = ""
    assumptions: List[str] = field(default_factory=list)
    last_feedback: str = ""
    retries: int = 0
    live_tokens: int = 0
    shared_invariants: str = ""

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

def init_git_workspace(workspace: Path) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    if not (workspace / ".git").exists():
        subprocess.run(["git", "init"], cwd=workspace, capture_output=True)
        subprocess.run(["git", "config", "user.name", "tasktree"], cwd=workspace, capture_output=True)
        subprocess.run(["git", "config", "user.email", "tasktree@local"], cwd=workspace, capture_output=True)


def git_commit(workspace: Path, message: str) -> None:
    subprocess.run(["git", "add", "."], cwd=workspace, capture_output=True)
    subprocess.run(["git", "commit", "-m", message], cwd=workspace, capture_output=True)


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
        self.planned = False
        self.context_dir = Path(context_dir).resolve() if context_dir else None

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

    def save_checkpoint(self) -> None:
        data = {
            "root_prompt": self.root_prompt,
            "planned": self.planned,
            "context_dir": str(self.context_dir) if self.context_dir else None,
            "nodes": {nid: asdict(node) for nid, node in self.nodes.items()},
        }
        self.checkpoint_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load_checkpoint(self) -> None:
        """Restores a saved run. DONE nodes are kept; everything else is reset to PENDING and re-run."""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint found at {self.checkpoint_path}")
        data = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
        if "nodes" not in data:
            raise ValueError(f"{self.checkpoint_path} predates resume support and cannot be resumed")
        self.root_prompt = data["root_prompt"]
        self.planned = data["planned"]
        if self.context_dir is None and data.get("context_dir"):
            self.context_dir = Path(data["context_dir"])
        self.nodes = {nid: TaskNode(**fields) for nid, fields in data["nodes"].items()}
        if not self.planned:
            # Planning was interrupted mid-tree; start the plan over from the root
            root = self.nodes[self.root_id]
            self.prune_subtree(self.root_id)
            self.nodes = {self.root_id: root}
        for node in self.nodes.values():
            node.live_tokens = 0
            if node.status != "DONE":
                node.status = "PENDING"

    def warn(self, live: Optional[Live], message: str) -> None:
        if live is not None:
            live.console.print(f"[yellow]{message}[/yellow]")

    def call_model(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: type[BaseModel],
        temp: Optional[float] = None,
        node: Optional[TaskNode] = None,
        live: Optional[Live] = None,
    ) -> BaseModel:
        """Stateless single-turn call with streaming token counter and schema validation."""
        schema_json = json.dumps(schema.model_json_schema(), indent=2)
        full_system = (
            f"{system_prompt}\n\n"
            f"You MUST respond with valid JSON strictly matching this schema:\n{schema_json}"
        )
        temperature = self.config["temperature"] if temp is None else temp

        last_err = None
        for _ in range(3):
            if node is not None:
                node.live_tokens = 0

            stream = self.client.chat.completions.create(
                model=self.config["model"],
                messages=[
                    {"role": "system", "content": full_system},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=self.config["max_tokens"],
                response_format={"type": "json_object"},
                stream=True,
            )

            chunks: List[str] = []
            for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    chunks.append(delta)
                    if node is not None:
                        node.live_tokens += 1
                        if live is not None and node.live_tokens % 5 == 0:
                            live.update(self.render_tree())

            content = "".join(chunks)
            if node is not None:
                node.live_tokens = 0
                if live is not None:
                    live.update(self.render_tree())

            try:
                parsed = extract_json_payload(content)
                return schema.model_validate(parsed)
            except Exception as e:
                last_err = e
                user_prompt += f"\n\nYour previous JSON was invalid ({e}). Return ONLY valid JSON matching the schema."
        raise RuntimeError(f"Model failed to return valid structured output: {last_err}")

    # --- Live Indented Number Renderer ---

    def render_tree(self) -> Text:
        status_styles = {
            "PENDING": "[dim]PENDING[/dim]",
            "PLANNING": "[cyan]PLANNING[/cyan]",
            "DOING": "[bold yellow]DOING[/bold yellow]",
            "TESTING": "[magenta]TESTING[/magenta]",
            "RETRYING": "[bold orange1]RETRYING[/bold orange1]",
            "DONE": "[bold green]DONE[/bold green]",
            "FAILED": "[bold red]FAILED[/bold red]",
            "REVIEWING": "[cyan]REVIEWING[/cyan]"
        }

        lines: List[str] = []

        def add_lines(node_id: str) -> None:
            node = self.nodes[node_id]
            tag = status_styles.get(node.status, node.status)

            # Show live token counter on active states, or child progress on pending parents
            if node.status in ("PLANNING", "DOING", "TESTING", "RETRYING") and node.live_tokens > 0:
                tag = f"{tag} [dim]({node.live_tokens} tok)[/dim]"

            elif node.status == "PENDING" and node.children_ids:
                done_count = sum(1 for cid in node.children_ids if self.nodes[cid].status == "DONE")

                if done_count > 0:
                    tag = f"{tag} [dim]({done_count}/{len(node.children_ids)})[/dim]"

            file_hint = f" [dim]({node.file_path})[/dim]" if node.file_path else ""

            lines.append(f"[bold]{node.id}.[/bold] {node.title}{file_hint}  {tag}")

            for cid in node.children_ids:
                add_lines(cid)

        add_lines(self.root_id)

        return Text.from_markup("\n".join(lines))

    def inherited_invariants(self, node: TaskNode) -> str:
        """Collects shared_invariants set by every ancestor, root first."""
        chain: List[str] = []
        curr = node.parent_id
        while curr:
            parent = self.nodes[curr]
            if parent.shared_invariants:
                chain.append(f"From {parent.id} ({parent.title}): {parent.shared_invariants}")
            curr = parent.parent_id
        return "\n".join(reversed(chain)) if chain else "None."

    # --- Phase 1: Top-Down Contract-First Planning ---
    def plan_node(self, node_id: str, live: Live, revision_feedback: str = "") -> None:
        node = self.nodes[node_id]
        if node.depth >= self.config["max_depth"]:
            return

        node.status = "PLANNING"
        live.update(self.render_tree())

        parent_context = "None (this is the root task)."
        if node.parent_id:
            p = self.nodes[node.parent_id]
            parent_context = f"Parent Task: {p.title}\nParent Test Criteria: {p.test_criteria}"

        system_prompt = self.config["prompts"]["planner"].format(
            max_children=self.config["max_children"]
        )

        context_map = ""
        if node.depth == 0 and self.context_dir:
            context_map = f"\n\nRead-Only Context Directory Files:\n{get_directory_tree(self.context_dir)}"

        feedback_block = (
            f"\nRevision Feedback from Plan Reviewer (you MUST address this):\n{revision_feedback}\n"
            if revision_feedback
            else ""
        )

        user_prompt = (
            f"Root Goal (for scale only): {self.root_prompt}\n"
            f"Current Depth: {node.depth} | Estimated Share of Total Scope: {node.scope_weight:.1f}%\n\n"
            f"{parent_context}\n\n"
            f"Current Task: {node.title}\n"
            f"Description: {node.description}\n"
            f"Current Test Criteria: {node.test_criteria}\n"
            f"Shared Invariants (from ancestors, must be honored):\n{self.inherited_invariants(node)}"
            f"{context_map}"
            f"{feedback_block}"
        )

        try:
            plan: Optional[PlanResponse] = self.call_model(
                system_prompt, user_prompt, PlanResponse, node=node, live=live
            )
        except Exception as e:
            self.warn(live, f"Planning failed for node {node.id}; treating it as a leaf. ({e})")
            plan = None

        if plan is not None and not plan.is_leaf and plan.subtasks:
            node.shared_invariants = plan.shared_invariants
            subtasks = plan.subtasks[: self.config["max_children"]]
            child_weight = node.scope_weight / len(subtasks)
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
                    verify_command=spec.verify_command
                )

            # Resolve sibling DAG dependencies
            for idx, spec in enumerate(subtasks):
                cid = created_ids[idx]
                for dep_idx in spec.depends_on:
                    if 0 <= dep_idx < len(created_ids) and dep_idx != idx:
                        self.nodes[cid].depends_on.append(created_ids[dep_idx])

            node.children_ids = created_ids

        node.status = "PENDING"
        self.save_checkpoint()
        live.update(self.render_tree())

        for cid in node.children_ids:
            self.plan_node(cid, live)

    # --- Phase 2: Bottom-Up Execution & Verification ---
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

    def judge_output(self, node: TaskNode, live: Live) -> tuple[bool, str]:
        """Isolated judge call seeing only the contract and the artifact."""
        word_count = len(node.output.split())
        system_prompt = self.config["prompts"]["judge"]

        user_prompt = (
            f"Task: {node.title}\n"
            f"Test Criteria: {node.test_criteria}\n"
            f"Actual Word Count: {word_count}\n\n"
            f"Produced Output:\n{node.output}\n\n"
            f"Assumptions Made: {node.assumptions}"
        )
        verdict: JudgeResponse = self.call_model(
            system_prompt, user_prompt, JudgeResponse, temp=0.0, node=node, live=live
        )
        return verdict.passed, verdict.feedback

    def gather_dependency_stubs(self, node: TaskNode) -> str:
        """Pulls lightweight AST stubs (for code) or summaries (for prose) from completed dependencies."""
        if not node.depends_on:
            return "None."
        blocks = []
        for dep_id in node.depends_on:
            dep = self.nodes[dep_id]
            if dep.file_path and dep.file_path.endswith(".py"):
                stub = extract_python_stub(self.workspace / dep.file_path)
                blocks.append(f"--- Interface Stub for {dep.file_path} ({dep.title}) ---\n{stub}")
            else:
                blocks.append(f"--- Summary of Dependency '{dep.title}' ---\n{dep.summary}")
        return "\n\n".join(blocks)

    def execute_node(self, node_id: str, live: Live) -> None:
        node = self.nodes[node_id]
        if node.status == "DONE":
            return  # Already completed in a resumed run

        # First ensure all children are done (respecting sibling depends_on DAG)
        if node.children_ids:
            remaining = list(node.children_ids)
            while remaining:
                progressed = False
                for cid in list(remaining):
                    child = self.nodes[cid]
                    if all(self.nodes[d].status == "DONE" for d in child.depends_on):
                        self.execute_node(cid, live)
                        remaining.remove(cid)
                        progressed = True
                if not progressed:
                    cid = remaining.pop(0)
                    self.execute_node(cid, live)

        parent_context = "None (this is the root task)."
        if node.parent_id:
            p = self.nodes[node.parent_id]
            parent_context = f"Parent Task: {p.title}\nParent Test Criteria: {p.test_criteria}"

        children_context = ""
        is_code_integration = False
        if node.children_ids:
            parts = []
            for cid in node.children_ids:
                c = self.nodes[cid]
                if c.file_path and c.file_path.endswith(".py"):
                    is_code_integration = True
                    stub = extract_python_stub(self.workspace / c.file_path)
                    parts.append(
                        f"Child {c.id} ({c.title}) -> {c.file_path}\n"
                        f"Summary: {c.summary}\nAssumptions: {c.assumptions}\nInterface:\n{stub}"
                    )
                else:
                    parts.append(
                        f"Child {c.id} ({c.title})\nSummary: {c.summary}\n"
                        f"Assumptions: {c.assumptions}\nContent:\n{c.output}"
                    )
            children_context = "\n\n".join(parts)

        dep_context = self.gather_dependency_stubs(node)
        read_context = read_context_files(self.context_dir, node.read_files)
        invariants = self.inherited_invariants(node)

        for attempt in range(1, self.config["max_retries"] + 1):
            node.status = "DOING" if attempt == 1 else "RETRYING"
            node.retries = attempt - 1
            live.update(self.render_tree())

            integration_guidance = (
                self.config["prompts"]["code_integration"]
                if is_code_integration or not node.children_ids
                else self.config["prompts"]["prose_integration"]
            )

            system_prompt = self.config["prompts"]["executor"].format(
                integration_guidance=integration_guidance
            )

            user_prompt = (
                f"Root Goal (for scale only): {self.root_prompt}\n"
                f"{parent_context}\n\n"
                f"Current Task: {node.title}\n"
                f"Description: {node.description}\n"
                f"Test Criteria: {node.test_criteria}\n"
                f"Target File: {node.file_path or 'None (prose/integration)'}\n"
                f"Verify Command: {node.verify_command or 'None'}\n\n"
                f"Shared Invariants (from ancestors, must be honored):\n{invariants}\n\n"
                f"Sibling Dependencies:\n{dep_context}\n\n"
                f"Reference Files to Inspect:\n{read_context}\n"
            )
            if children_context:
                user_prompt += f"\nImmediate Children to Integrate:\n{children_context}\n"
            if node.last_feedback:
                user_prompt += f"\nPrevious Attempt Failed With:\n{node.last_feedback}\nFix this completely."

            try:
                result: ExecutionResponse = self.call_model(
                    system_prompt, user_prompt, ExecutionResponse, node=node, live=live
                )
                node.output = result.output
                node.summary = result.summary
                node.assumptions = result.assumptions

                # Write artifacts to disk if applicable
                if node.file_path:
                    target = resolve_within(self.workspace, node.file_path)
                    if target is None:
                        raise ValueError(f"file_path {node.file_path!r} is outside the workspace")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(node.output, encoding="utf-8")

                # Verify node
                node.status = "TESTING"
                live.update(self.render_tree())

                if node.verify_command:
                    passed, feedback = self.run_verify_command(node.verify_command)
                else:
                    passed, feedback = self.judge_output(node, live)
            except Exception as e:
                self.warn(live, f"Node {node.id} attempt {attempt} errored: {e}")
                passed, feedback = False, f"The previous attempt errored before it could be verified: {e}"

            if passed:
                node.status = "DONE"
                node.last_feedback = ""
                git_commit(self.workspace, f"Complete node {node.id}: {node.title}")
                self.save_checkpoint()
                live.update(self.render_tree())
                return
            else:
                node.last_feedback = feedback

        node.status = "FAILED"
        self.save_checkpoint()
        live.update(self.render_tree())

    def run(self) -> None:
        console = Console()
        try:
            with Live(self.render_tree(), console=console, refresh_per_second=6) as live:
                if not self.planned:
                    self.plan_node(self.root_id, live)
                    self.review_and_revise_plan(live)
                    self.planned = True
                    self.save_checkpoint()
                self.execute_node(self.root_id, live)

        except KeyboardInterrupt:
            self.save_checkpoint()
            console.print(
                f"\n[yellow]Interrupted. State saved to {self.checkpoint_path}; "
                f"continue with: tasktree --resume -w {self.workspace}[/yellow]"
            )
            sys.exit(130)

        failed_nodes = [n for n in self.nodes.values() if n.status == "FAILED"]
        for fn in failed_nodes:
            console.print(f"\n[bold red]Node {fn.id} ({fn.title}) failed:[/bold red] {fn.last_feedback}")
        if failed_nodes:
            console.print(f"\n[yellow]{len(failed_nodes)} node(s) failed; rerun them with: tasktree --resume -w {self.workspace}[/yellow]")

        root = self.nodes[self.root_id]
        if self.output_file and root.output:
            out_path = Path(self.output_file)
            out_path.write_text(root.output, encoding="utf-8")
            console.print(f"\n[bold green]Saved final output to {out_path}[/bold green]")

    def serialize_tree_outline(self, node_id: Optional[str] = None, level: int = 0) -> str:
        nid = node_id or self.root_id
        node = self.nodes[nid]
        pad = "  " * level
        lines = [
            f"{pad}- {node.id}. {node.title}",
            f"{pad}  Description: {node.description}",
            f"{pad}  Test Criteria: {node.test_criteria}",
        ]
        if node.shared_invariants:
            lines.append(f"{pad}  Shared Invariants: {node.shared_invariants}")
        for cid in node.children_ids:
            lines.append(self.serialize_tree_outline(cid, level + 1))
        return "\n".join(lines)

    def prune_subtree(self, node_id: str) -> None:
        node = self.nodes[node_id]
        for cid in list(node.children_ids):
            self.prune_subtree(cid)
            self.nodes.pop(cid, None)
        node.children_ids = []
        node.shared_invariants = ""

    def review_and_revise_plan(self, live: Live) -> None:
        max_revs = self.config.get("max_plan_revisions", 0)
        if max_revs <= 0:
            return

        root = self.nodes[self.root_id]
        try:
            self._review_rounds(root, max_revs, live)
        finally:
            root.status = "PENDING"
            live.update(self.render_tree())

    def _review_rounds(self, root: TaskNode, max_revs: int, live: Live) -> None:
        # One more review than revisions, so the final revision is also checked
        for round_idx in range(max_revs + 1):
            root.status = "REVIEWING"
            live.update(self.render_tree())

            system_prompt = self.config["prompts"]["plan_reviewer"]
            user_prompt = (
                f"Root Goal: {self.root_prompt}\n\n"
                f"Proposed Task Tree:\n{self.serialize_tree_outline()}\n"
            )
            try:
                review: PlanReviewResponse = self.call_model(
                    system_prompt, user_prompt, PlanReviewResponse
                )
            except Exception as e:
                self.warn(live, f"Plan review failed; executing the current plan. ({e})")
                return

            if review.approved or not review.flagged_node_ids:
                return

            if round_idx == max_revs:
                self.warn(live, f"Plan still not approved after {max_revs} revision(s): {review.critique}")
                return

            valid_flagged = [nid for nid in review.flagged_node_ids if nid in self.nodes]
            if not valid_flagged:
                return

            # Only re-plan the highest flagged ancestors so we don't re-plan nodes already wiped by a parent
            flagged_set = set(valid_flagged)
            top_level_flagged = []
            for nid in valid_flagged:
                curr = self.nodes[nid].parent_id
                has_flagged_ancestor = False
                while curr:
                    if curr in flagged_set:
                        has_flagged_ancestor = True
                        break
                    curr = self.nodes[curr].parent_id
                if not has_flagged_ancestor:
                    top_level_flagged.append(nid)

            for nid in top_level_flagged:
                self.prune_subtree(nid)
                self.plan_node(nid, live, revision_feedback=review.critique)

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
        "max_depth": parser["harness"].getint("max_depth"),
        "max_children": parser["harness"].getint("max_children"),
        "max_retries": parser["harness"].getint("max_retries"),
        "workspace": parser["harness"]["workspace"],
        "max_plan_revisions": parser["harness"].getint("max_plan_revisions"),
        "prompts": {
            "planner": parser["prompts"]["planner"].strip(),
            "executor": parser["prompts"]["executor"].strip(),
            "code_integration": parser["prompts"]["code_integration"].strip(),
            "prose_integration": parser["prompts"]["prose_integration"].strip(),
            "judge": parser["prompts"]["judge"].strip(),
            "plan_reviewer": parser["prompts"]["plan_reviewer"].strip(),
        },
    }

def main() -> None:
    argp = argparse.ArgumentParser(description="Recursive local-context task tree harness.")
    argp.add_argument("prompt", nargs="?", help="Root task to plan and execute.")
    argp.add_argument("-o", "--output", help="Optional file path to save the root node's final output.")
    argp.add_argument("-w", "--workspace", help="Override workspace directory.")
    argp.add_argument("-c", "--context", help="Optional read-only directory (e.g. an existing repo) to analyze.")
    argp.add_argument("--rc", default=str(Path.home() / ".tasktreerc"), help="Path to .tasktreerc file.")
    argp.add_argument("--init-config", action="store_true", help="Write or overwrite .tasktreerc with default settings and prompts.")
    argp.add_argument("--resume", action="store_true", help="Resume the run checkpointed in the workspace, re-running any node not DONE.")
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

    config = load_config(rc_path)
    if args.workspace:
        config["workspace"] = args.workspace

    try:
        harness = TaskTreeHarness(
            config, args.prompt, output_file=args.output, context_dir=args.context, resume=args.resume
        )
    except (FileNotFoundError, ValueError) as e:
        argp.error(str(e))
    harness.run()

if __name__ == "__main__":
    main()

