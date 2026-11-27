import json
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from tasktree import main as tt


# ---------------------------------------------------------------------------
# Fake OpenAI client: routes each call to a script by which prompt it carries
# ---------------------------------------------------------------------------

class FakeClient:
    def __init__(self, script):
        self.script = script
        self.calls = []  # (role, task title, user prompt)
        self.chat = NS(completions=NS(create=self.create))

    def create(self, messages, **kwargs):
        system, user = messages[0]["content"], messages[1]["content"]
        match = re.search(r"Current Task: (.*)", user)
        task = match.group(1) if match else None
        if "task planner" in system:
            role = "plan"
        elif "tree architect" in system:
            role = "review"
        elif "independent verifier" in system:
            role = "judge"
        else:
            role = "exec"
        self.calls.append((role, task, user))
        return [NS(choices=[NS(delta=NS(content=self.script(role, task, user)))])]

    def prompts(self, role, task):
        return [u for r, t, u in self.calls if r == role and t == task]

    def tasks(self, role):
        return [t for r, t, _ in self.calls if r == role]


def leaf_plan():
    return json.dumps({"reasoning": "r", "is_leaf": True})


def exec_ok(output="text"):
    return json.dumps({"reasoning": "r", "output": output, "summary": "s", "assumptions": []})


def judge_pass():
    return json.dumps({"reasoning": "r", "passed": True, "feedback": ""})


def approve():
    return json.dumps({"approved": True, "critique": "ok", "flagged_node_ids": []})


def split_plan(*subtasks, invariants=""):
    specs = [{"title": t, "description": t, "test_criteria": t, **extra} for t, extra in subtasks]
    return json.dumps({"reasoning": "r", "is_leaf": False, "shared_invariants": invariants, "subtasks": specs})


def default_script(root_plan):
    def script(role, task, user):
        if role == "plan":
            return root_plan if task == "Root" else leaf_plan()
        if role == "review":
            return approve()
        if role == "judge":
            return judge_pass()
        return exec_ok()
    return script


@pytest.fixture
def make_harness(tmp_path):
    def make(script, prompt="Root", resume=False, workspace=None, **overrides):
        config = tt.load_config(tmp_path / "missing-rc")
        config.update(workspace=str(workspace or tmp_path / "ws"), max_retries=2, **overrides)
        harness = tt.TaskTreeHarness(config, None if resume else prompt, resume=resume)
        harness.client = FakeClient(script)
        return harness
    return make


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

def test_depends_on_is_zero_based(make_harness):
    h = make_harness(default_script(split_plan(
        ("A", {}), ("B", {"depends_on": [0]}), ("C", {"depends_on": [0, 1]}),
    )))
    h.run()
    assert h.nodes["1.2"].depends_on == ["1.1"]
    assert h.nodes["1.3"].depends_on == ["1.1", "1.2"]
    assert h.client.tasks("exec") == ["A", "B", "C", "Root"]


def test_schema_documents_zero_based_depends_on():
    assert "0-based" in json.dumps(tt.PlanResponse.model_json_schema())


def test_shared_invariants_reach_child_planner_and_executor(make_harness):
    h = make_harness(default_script(split_plan(("A", {}), invariants="Use British spelling")))
    h.run()
    assert h.nodes["1"].shared_invariants == "Use British spelling"
    assert "Use British spelling" in h.client.prompts("plan", "A")[0]
    assert "Use British spelling" in h.client.prompts("exec", "A")[0]


def test_planner_failure_makes_node_a_leaf(make_harness):
    def script(role, task, user):
        if role == "plan":
            return "garbage"
        return judge_pass() if role == "judge" else exec_ok()
    h = make_harness(script, max_plan_revisions=0)
    h.run()
    assert h.nodes["1"].children_ids == []
    assert h.nodes["1"].status == "DONE"


# ---------------------------------------------------------------------------
# Path containment
# ---------------------------------------------------------------------------

def test_escaping_file_paths_are_dropped(make_harness, tmp_path):
    outside = tmp_path / "abs_evil.txt"
    h = make_harness(default_script(split_plan(
        ("A", {"file_path": "../evil.txt"}),
        ("B", {"file_path": str(outside)}),
        ("C", {"file_path": "ok/c.py"}),
    )))
    h.run()
    assert h.nodes["1.1"].file_path is None
    assert h.nodes["1.2"].file_path is None
    assert not (tmp_path / "evil.txt").exists()
    assert not outside.exists()
    assert (h.workspace / "ok/c.py").read_text() == "text"


def test_resolve_within_rejects_sibling_prefix(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    assert tt.resolve_within(root, "../repo-secrets/key") is None
    assert tt.resolve_within(root, "/etc/passwd") is None
    assert tt.resolve_within(root, ".") is None
    assert tt.resolve_within(root, "a/b.txt") == (root / "a/b.txt").resolve()


def test_read_context_files_skips_paths_outside_context(tmp_path):
    ctx = tmp_path / "repo"
    ctx.mkdir()
    (tmp_path / "repo-secrets").mkdir()
    (tmp_path / "repo-secrets/key").write_text("SECRET")
    (ctx / "f.txt").write_text("fine")
    out = tt.read_context_files(ctx, ["../repo-secrets/key", "f.txt"])
    assert "SECRET" not in out
    assert "fine" in out


# ---------------------------------------------------------------------------
# Execution failures
# ---------------------------------------------------------------------------

def test_bad_model_output_fails_node_without_stopping_run(make_harness, capsys):
    base = default_script(split_plan(("A", {}), ("B", {}), ("C", {})))

    def script(role, task, user):
        if role == "exec" and task == "B":
            return "not json"
        return base(role, task, user)

    h = make_harness(script)
    h.run()
    assert h.nodes["1.2"].status == "FAILED"
    assert "errored" in h.nodes["1.2"].last_feedback
    assert [h.nodes[i].status for i in ("1.1", "1.3", "1")] == ["DONE"] * 3
    # Reported even without -o
    assert "Node 1.2 (B) failed" in capsys.readouterr().out


def test_verify_timeout_is_a_failure_not_a_crash(make_harness, monkeypatch):
    h = make_harness(default_script(leaf_plan()))

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("cmd", 60)

    monkeypatch.setattr(tt.subprocess, "run", timeout)
    passed, feedback = h.run_verify_command("sleep 999")
    assert passed is False
    assert "timed out" in feedback


# ---------------------------------------------------------------------------
# Plan review
# ---------------------------------------------------------------------------

def test_final_revision_is_reviewed_and_root_leaves_reviewing(make_harness):
    base = default_script(split_plan(("A", {}), ("B", {})))
    reviews = []

    def script(role, task, user):
        if role == "review":
            reviews.append(user)
            return json.dumps({"approved": False, "critique": "fix B", "flagged_node_ids": ["1.2"]})
        return base(role, task, user)

    h = make_harness(script, max_plan_revisions=1)
    h.plan_node(h.root_id, live=NS(update=lambda *_: None, console=NS(print=lambda *_: None)))
    h.review_and_revise_plan(live=NS(update=lambda *_: None, console=NS(print=lambda *_: None)))
    assert len(reviews) == 2  # one revision, then a review of that revision
    assert h.client.tasks("plan").count("B") == 2
    assert h.nodes["1"].status == "PENDING"


# ---------------------------------------------------------------------------
# Checkpoint / resume
# ---------------------------------------------------------------------------

def test_resume_reruns_only_unfinished_nodes(make_harness):
    base = default_script(split_plan(("A", {}), ("B", {})))

    def flaky(role, task, user):
        if role == "exec" and task == "B":
            return "not json"
        return base(role, task, user)

    first = make_harness(flaky)
    first.run()
    assert first.nodes["1.2"].status == "FAILED"

    resumed = make_harness(base, resume=True, workspace=first.workspace)
    assert resumed.root_prompt == "Root"
    resumed.run()
    assert set(resumed.client.tasks("exec")) == {"B", "Root"}
    assert not resumed.client.tasks("plan")
    assert not resumed.client.tasks("review")
    assert all(n.status == "DONE" for n in resumed.nodes.values())


def test_resume_replans_if_planning_was_interrupted(make_harness):
    h = make_harness(default_script(split_plan(("A", {}))))
    h.nodes["1.9"] = tt.TaskNode(id="1.9", title="stale", description="", test_criteria="", parent_id="1")
    h.nodes["1"].children_ids = ["1.9"]
    h.save_checkpoint()  # planned is still False

    resumed = make_harness(default_script(split_plan(("A", {}))), resume=True, workspace=h.workspace)
    assert list(resumed.nodes) == ["1"]
    resumed.run()
    assert "stale" not in [n.title for n in resumed.nodes.values()]
    assert resumed.client.tasks("plan")[0] == "Root"


def test_resume_rejects_old_checkpoint_format(make_harness, tmp_path):
    ws = tmp_path / "old"
    ws.mkdir()
    (ws / ".tasktree_state.json").write_text(json.dumps({"1": {}}))
    with pytest.raises(ValueError, match="predates resume support"):
        make_harness(default_script(leaf_plan()), resume=True, workspace=ws)


def test_resume_without_checkpoint_errors(make_harness, tmp_path):
    with pytest.raises(FileNotFoundError):
        make_harness(default_script(leaf_plan()), resume=True, workspace=tmp_path / "empty")
