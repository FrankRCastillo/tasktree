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

from openai import OpenAI
from pydantic import BaseModel, Field
from rich.console import Console
from rich.live import Live
from rich.tree import Tree


# ---------------------------------------------------------------------------
# Structured Output Schemas (reasoning first so Qwen can think before acting)
# ---------------------------------------------------------------------------

class SubtaskSpec(BaseModel):
    reasoning: str = Field(description="Why this subtask is needed and how it fits the parent.")
    title: str = Field(description="Short, clear name for the subtask.")
    description: str = Field(description="Detailed instructions for what this subtask must produce.")
    test_criteria: str = Field(description="Concrete pass/fail criteria aligned with the parent's test.")
    depends_on_indices: List[int] = Field(
        default_factory=list,
        description="0-based indices of sibling subtasks in this same list that must finish before this one."
    )
    file_path: Optional[str] = Field(
        default=None,
        description="Relative path to write code/artifact if this is a file-producing task (e.g. 'models/base.py')."
    )
    test_file_path: Optional[str] = Field(
        default=None,
        description="Relative path for a pytest file if this is a Python code task (e.g. 'tests/test_base.py')."
    )


class PlanResponse(BaseModel):
    reasoning: str = Field(description="Evaluate whether this task fits in a single small context window or needs splitting.")
    is_leaf: bool = Field(description="True if this task is small enough to execute directly; False if it must be split.")
    subtasks: List[SubtaskSpec] = Field(default_factory=list, description="Ordered list of child subtasks if is_leaf is False.")


class ExecutionResponse(BaseModel):
    reasoning: str = Field(description="Step-by-step reasoning to solve the task.")
    output: str = Field(description="The final prose or complete source code for this node.")
    test_code: Optional[str] = Field(
        default=None,
        description="Complete pytest test code if this node has a test_file_path."
    )
    summary: str = Field(description="Tight, 2-4 sentence summary of what was built or concluded, for the parent.")
    assumptions: List[str] = Field(
        default_factory=list,
        description="Short list of key assumptions or caveats made so the parent can check for conflicts."
    )


class JudgeResponse(BaseModel):
    reasoning: str = Field(description="Strict evaluation of the output against the test criteria.")
    passed: bool = Field(description="True only if all test criteria are satisfied.")
    feedback: str = Field(description="Specific, actionable feedback if passed is False.")


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
    status: str = "PENDING"  # PENDING, PLANNING, DOING, TESTING, RETRYING, DONE, FAILED
    depends_on: List[str] = field(default_factory=list)
    children_ids: List[str] = field(default_factory=list)
    file_path: Optional[str] = None
    test_file_path: Optional[str] = None
    output: str = ""
    summary: str = ""
    assumptions: List[str] = field(default_factory=list)
    last_feedback: str = ""
    retries: int = 0


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
    def __init__(self, config: Dict[str, Any], root_prompt: str, output_file: Optional[str] = None):
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
        self.nodes[self.root_id] = TaskNode(
            id=self.root_id,
            title=root_prompt,
            description=root_prompt,
            test_criteria=f"Fully and accurately satisfies the user's root request: {root_prompt}",
            depth=0,
        )
        init_git_workspace(self.workspace)

    def save_checkpoint(self) -> None:
        data = {nid: asdict(node) for nid, node in self.nodes.items()}
        self.checkpoint_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def call_model(self, system_prompt: str, user_prompt: str, schema: type[BaseModel], temp: Optional[float] = None) -> BaseModel:
        """Stateless single-turn call with schema guidance and retry on malformed JSON."""
        schema_json = json.dumps(schema.model_json_schema(), indent=2)
        full_system = (
            f"{system_prompt}\n\n"
            f"You MUST respond with valid JSON strictly matching this schema:\n{schema_json}"
        )
        temperature = self.config["temperature"] if temp is None else temp

        last_err = None
        for _ in range(3):
            resp = self.client.chat.completions.create(
                model=self.config["model"],
                messages=[
                    {"role": "system", "content": full_system},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=self.config["max_tokens"],
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content or ""
            try:
                parsed = extract_json_payload(content)
                return schema.model_validate(parsed)
            except Exception as e:
                last_err = e
                user_prompt += f"\n\nYour previous JSON was invalid ({e}). Return ONLY valid JSON matching the schema."
        raise RuntimeError(f"Model failed to return valid structured output: {last_err}")

    # --- Live Rich Tree Renderer ---

    def render_tree(self) -> Tree:
        status_styles = {
            "PENDING": "[dim]PENDING[/dim]",
            "PLANNING": "[cyan]PLANNING[/cyan]",
            "DOING": "[bold yellow]DOING[/bold yellow]",
            "TESTING": "[magenta]TESTING[/magenta]",
            "RETRYING": "[bold orange1]RETRYING[/bold orange1]",
            "DONE": "[bold green]DONE[/bold green]",
            "FAILED": "[bold red]FAILED[/bold red]",
        }

        def add_branch(node_id: str, parent_branch: Optional[Tree] = None) -> Tree:
            node = self.nodes[node_id]
            tag = status_styles.get(node.status, node.status)
            file_hint = f" [dim]({node.file_path})[/dim]" if node.file_path else ""
            label = f"[bold]{node.id}.[/bold] {node.title}{file_hint}  {tag}"
            branch = Tree(label) if parent_branch is None else parent_branch.add(label)
            for cid in node.children_ids:
                add_branch(cid, branch)
            return branch

        return add_branch(self.root_id)

    # --- Phase 1: Top-Down Contract-First Planning ---

    def plan_node(self, node_id: str, live: Live) -> None:
        node = self.nodes[node_id]
        if node.depth >= self.config["max_depth"]:
            return

        node.status = "PLANNING"
        live.update(self.render_tree())

        parent_context = "None (this is the root task)."
        if node.parent_id:
            p = self.nodes[node.parent_id]
            parent_context = f"Parent Task: {p.title}\nParent Test Criteria: {p.test_criteria}"

        system_prompt = (
            "You are a disciplined recursive task planner. Decide if the current task is small enough "
            "to solve cleanly in one focused pass (is_leaf=True) or should be decomposed into at most "
            f"{self.config['max_children']} subtasks (is_leaf=False). Every child's test_criteria must "
            "explicitly align with the parent's test_criteria so that passing all children guarantees "
            "the parent can be integrated. If this is a software project, assign relative file_path and "
            "test_file_path to leaf tasks, and use depends_on_indices when a derived module depends on a base module."
        )
        user_prompt = (
            f"{parent_context}\n\n"
            f"Current Task: {node.title}\n"
            f"Description: {node.description}\n"
            f"Current Test Criteria: {node.test_criteria}\n"
            f"Current Depth: {node.depth} / {self.config['max_depth']}"
        )

        plan: PlanResponse = self.call_model(system_prompt, user_prompt, PlanResponse)

        if not plan.is_leaf and plan.subtasks:
            subtasks = plan.subtasks[: self.config["max_children"]]
            created_ids: List[str] = []
            for idx, spec in enumerate(subtasks, start=1):
                cid = f"{node.id}.{idx}"
                created_ids.append(cid)
                self.nodes[cid] = TaskNode(
                    id=cid,
                    title=spec.title,
                    description=spec.description,
                    test_criteria=spec.test_criteria,
                    parent_id=node.id,
                    depth=node.depth + 1,
                    file_path=spec.file_path,
                    test_file_path=spec.test_file_path,
                )

            # Resolve sibling DAG dependencies
            for idx, spec in enumerate(subtasks):
                cid = created_ids[idx]
                for dep_idx in spec.depends_on_indices:
                    if 0 <= dep_idx < len(created_ids) and dep_idx != idx:
                        self.nodes[cid].depends_on.append(created_ids[dep_idx])

            node.children_ids = created_ids

        node.status = "PENDING"
        self.save_checkpoint()
        live.update(self.render_tree())

        for cid in node.children_ids:
            self.plan_node(cid, live)

    # --- Phase 2: Bottom-Up Execution & Verification ---

    def run_pytest(self, test_file: Path) -> tuple[bool, str]:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.workspace) + os.pathsep + env.get("PYTHONPATH", "")
        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(test_file)],
            cwd=self.workspace,
            capture_output=True,
            text=True,
            env=env,
            timeout=60,
        )
        passed = result.returncode == 0
        output = (result.stdout + "\n" + result.stderr).strip()
        return passed, output[-2000:]

    def judge_output(self, node: TaskNode) -> tuple[bool, str]:
        """Isolated judge call seeing only the contract and the artifact."""
        system_prompt = (
            "You are a strict, independent verifier. Evaluate whether the provided output completely "
            "satisfies the test criteria. Do not be lenient."
        )
        user_prompt = (
            f"Task: {node.title}\n"
            f"Test Criteria: {node.test_criteria}\n\n"
            f"Produced Output:\n{node.output}\n\n"
            f"Assumptions Made: {node.assumptions}"
        )
        verdict: JudgeResponse = self.call_model(system_prompt, user_prompt, JudgeResponse, temp=0.0)
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
                    # Break any accidental cycle by running the first remaining child
                    cid = remaining.pop(0)
                    self.execute_node(cid, live)

        # Now execute this node (either leaf work or parent integration of immediate children)
        parent_context = "None (this is the root task)."
        if node.parent_id:
            p = self.nodes[node.parent_id]
            parent_context = f"Parent Task: {p.title}\nParent Test Criteria: {p.test_criteria}"

        children_context = ""
        if node.children_ids:
            parts = []
            for cid in node.children_ids:
                c = self.nodes[cid]
                if c.file_path and c.file_path.endswith(".py"):
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

        for attempt in range(1, self.config["max_retries"] + 1):
            node.status = "DOING" if attempt == 1 else "RETRYING"
            node.retries = attempt - 1
            live.update(self.render_tree())

            system_prompt = (
                "You are executing a single node in a hierarchical task tree. Focus strictly on the "
                "current task, honor the parent's contract, and satisfy the test criteria. If children "
                "outputs are provided, integrate them cleanly into a cohesive whole and check that their "
                "assumptions do not conflict. If file_path is specified, output complete, runnable code."
            )
            user_prompt = (
                f"{parent_context}\n\n"
                f"Current Task: {node.title}\n"
                f"Description: {node.description}\n"
                f"Test Criteria: {node.test_criteria}\n"
                f"Target File: {node.file_path or 'None (prose/integration)'}\n"
                f"Test File: {node.test_file_path or 'None'}\n\n"
                f"Sibling Dependencies:\n{dep_context}\n"
            )
            if children_context:
                user_prompt += f"\nImmediate Children to Integrate:\n{children_context}\n"
            if node.last_feedback:
                user_prompt += f"\nPrevious Attempt Failed With:\n{node.last_feedback}\nFix this completely."

            result: ExecutionResponse = self.call_model(system_prompt, user_prompt, ExecutionResponse)
            node.output = result.output
            node.summary = result.summary
            node.assumptions = result.assumptions

            # Write artifacts to disk if applicable
            if node.file_path:
                target = self.workspace / node.file_path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(node.output, encoding="utf-8")
            if node.test_file_path and result.test_code:
                test_target = self.workspace / node.test_file_path
                test_target.parent.mkdir(parents=True, exist_ok=True)
                test_target.write_text(result.test_code, encoding="utf-8")

            # Verify node
            node.status = "TESTING"
            live.update(self.render_tree())

            if node.test_file_path and (self.workspace / node.test_file_path).exists():
                passed, feedback = self.run_pytest(self.workspace / node.test_file_path)
            else:
                passed, feedback = self.judge_output(node)

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
        with Live(self.render_tree(), console=console, refresh_per_second=6) as live:
            self.plan_node(self.root_id, live)
            self.execute_node(self.root_id, live)

        root = self.nodes[self.root_id]
        if self.output_file and root.output:
            out_path = Path(self.output_file)
            out_path.write_text(root.output, encoding="utf-8")
            console.print(f"\n[bold green]Saved final output to {out_path}[/bold green]")


# ---------------------------------------------------------------------------
# CLI & Config Loading
# ---------------------------------------------------------------------------

def load_config(rc_path: Path) -> Dict[str, Any]:
    defaults = {
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
        "model": "qwen2.5:14b",
        "temperature": 0.2,
        "max_tokens": 2048,
        "max_depth": 3,
        "max_children": 4,
        "max_retries": 3,
        "workspace": "./tasktree_workspace",
    }
    if rc_path.exists():
        parser = configparser.ConfigParser()
        parser.read(rc_path)
        if "model" in parser:
            defaults["base_url"] = parser["model"].get("base_url", defaults["base_url"])
            defaults["api_key"] = parser["model"].get("api_key", defaults["api_key"])
            defaults["model"] = parser["model"].get("model", defaults["model"])
            defaults["temperature"] = parser["model"].getfloat("temperature", defaults["temperature"])
            defaults["max_tokens"] = parser["model"].getint("max_tokens", defaults["max_tokens"])
        if "harness" in parser:
            defaults["max_depth"] = parser["harness"].getint("max_depth", defaults["max_depth"])
            defaults["max_children"] = parser["harness"].getint("max_children", defaults["max_children"])
            defaults["max_retries"] = parser["harness"].getint("max_retries", defaults["max_retries"])
            defaults["workspace"] = parser["harness"].get("workspace", defaults["workspace"])
    return defaults


def main() -> None:
    argp = argparse.ArgumentParser(description="Recursive local-context task tree harness.")
    argp.add_argument("prompt", help="Root task to plan and execute.")
    argp.add_argument("-o", "--output", help="Optional file path to save the root node's final output.")
    argp.add_argument("-w", "--workspace", help="Override workspace directory.")
    argp.add_argument("--rc", default=str(Path.home() / ".tasktreerc"), help="Path to .tasktreerc file.")
    args = argp.parse_args()

    config = load_config(Path(args.rc))
    if args.workspace:
        config["workspace"] = args.workspace

    harness = TaskTreeHarness(config, args.prompt, output_file=args.output)
    harness.run()


if __name__ == "__main__":
    main()
