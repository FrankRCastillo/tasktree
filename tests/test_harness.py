import io
import json
import re
import subprocess
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
        self.request_kwargs = []
        self.chat = NS(completions=NS(create=self.create))

    def create(self, messages, **kwargs):
        self.request_kwargs.append(kwargs)
        system, user = messages[0]["content"], messages[1]["content"]
        match = re.search(r"Current Task: (.*)", user)
        task = match.group(1) if match else None
        role = "review" if "reviewing the completion reports" in system else "agent"
        self.calls.append((role, task, user))
        out = self.script(role, task, user)
        if isinstance(out, list):
            return out  # a prebuilt stream, e.g. from truncated()
        return [NS(choices=[NS(delta=NS(content=out), finish_reason="stop")])]

    def prompts(self, role, task):
        return [u for r, t, u in self.calls if r == role and t == task]

    def tasks(self, role):
        return [t for r, t, _ in self.calls if r == role]


def worker(content="text", summary="s", **extra):
    return json.dumps({"reasoning": "r", "mode": "worker", "content": content, "summary": summary, **extra})


def manager(*subtasks, invariants=""):
    specs = [{"title": t, "description": f"scope of {t}", "test_criteria": t, **extra} for t, extra in subtasks]
    return json.dumps({"reasoning": "r", "mode": "manager", "shared_invariants": invariants, "subtasks": specs})


def approve(summary="branch done"):
    return json.dumps({"reasoning": "r", "approved": True, "summary": summary, "revisions": []})


def reject(*revisions):
    return json.dumps({
        "reasoning": "r", "approved": False, "summary": "not yet",
        "revisions": [{"node_id": nid, "feedback": fb} for nid, fb in revisions],
    })


def truncated(response_tokens=0, thinking_tokens=0):
    """A stream that hits its token limit, one chunk per token like llama.cpp."""
    stream = [NS(choices=[NS(delta=NS(content=None, reasoning_content="hmm "), finish_reason=None)])] * thinking_tokens
    stream += [NS(choices=[NS(delta=NS(content="word "), finish_reason=None)])] * response_tokens
    return stream + [NS(choices=[NS(delta=NS(content=None), finish_reason="length")])]


def tree(plans, review=approve):
    """plans maps a task title to the agent's response; anything unlisted works as a leaf."""
    def script(role, task, user):
        if role == "review":
            return review()
        return plans[task]() if task in plans else worker(content=f"{task} content", summary=f"{task} summary")
    return script


@pytest.fixture
def make_harness(tmp_path, monkeypatch):
    monkeypatch.setattr(tt, "detect_context_window", lambda base_url: None)  # no network in tests

    def make(script, prompt="Root", resume=False, workspace=None, **overrides):
        config = tt.load_config(tmp_path / "missing-rc")
        config.update(workspace=str(workspace or tmp_path / "ws"), max_retries=2, **overrides)
        harness = tt.TaskTreeHarness(config, None if resume else prompt, resume=resume)
        harness.client = FakeClient(script)
        return harness
    return make


# ---------------------------------------------------------------------------
# Worker / manager modes and aggregation
# ---------------------------------------------------------------------------

def test_root_worker_writes_output(make_harness, tmp_path):
    h = make_harness(tree({"Root": lambda: worker(content="the essay")}))
    h.output_file = str(tmp_path / "out.md")
    h.run()
    assert h.nodes["1"].status == "DONE"
    assert (tmp_path / "out.md").read_text() == "the essay"
    assert h.client.tasks("review") == []


def test_manager_aggregates_children_in_order_without_rewriting(make_harness):
    h = make_harness(tree({"Root": lambda: manager(("A", {}), ("B", {}), ("C", {}))}))
    h.run()
    assert h.nodes["1"].output == "A content\n\nB content\n\nC content"
    assert h.nodes["1"].summary == "branch done"
    # The root decides once, then only reviews: it never re-emits its children's work
    assert h.client.tasks("agent") == ["Root", "A", "B", "C"]
    assert h.client.tasks("review") == ["Root"]


def test_children_with_files_deliver_there_not_into_aggregate(make_harness):
    h = make_harness(tree({"Root": lambda: manager(("Intro", {}), ("Code", {"file_path": "app.py"}))}))
    h.run()
    assert h.nodes["1"].output == "Intro content"
    assert (h.workspace / "app.py").read_text() == "Code content"


def test_manager_with_file_path_writes_its_aggregate(make_harness):
    h = make_harness(tree({
        "Root": lambda: manager(("Report", {"file_path": "report.md"})),
        "Report": lambda: manager(("S1", {}), ("S2", {})),
    }))
    h.run()
    assert (h.workspace / "report.md").read_text() == "S1 content\n\nS2 content"


def test_max_depth_forces_worker(make_harness):
    responses = iter([manager(("A", {})), worker(content="done anyway")])
    h = make_harness(tree({"Root": lambda: next(responses)}), max_depth=0)
    h.run()
    assert h.nodes["1"].children_ids == []
    assert h.nodes["1"].output == "done anyway"
    first, second = h.client.prompts("agent", "Root")
    assert 'you MUST choose mode="worker"' in first
    assert "maximum depth" in second.split("Problem With Your Previous Attempt")[1]


def test_reasoning_stream_counts_as_progress_but_not_content(make_harness):
    h = make_harness(tree({}))
    node = h.nodes["1"]
    seen = []
    h.render_tree = lambda: seen.append(node.live_tokens)
    stream = [NS(choices=[NS(delta=NS(content=None, reasoning_content="hmm {"))])] * 5
    stream += [NS(choices=[NS(delta=NS(content=worker(content="final")))])]
    h.client.chat.completions.create = lambda **kwargs: stream
    response = h.call_model("sys", "user", tt.AgentResponse, label="agent", node=node, live=NS(update=lambda *_: None))
    assert response.content == "final"
    assert 5 in seen  # the counter advanced while the model was only thinking


def test_enable_thinking_true_leaves_request_unchanged(make_harness):
    h = make_harness(tree({}))
    assert h.config["enable_thinking"] is True  # the default
    h.run()
    assert all("extra_body" not in kwargs for kwargs in h.client.request_kwargs)


def test_enable_thinking_false_asks_template_to_skip_reasoning(make_harness):
    h = make_harness(tree({"Root": lambda: manager(("A", {}))}), enable_thinking=False)
    h.run()
    assert len(h.client.request_kwargs) == 3  # agent calls and the review alike
    for kwargs in h.client.request_kwargs:
        assert kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


def test_agent_prompt_placeholder_tolerates_literal_braces(make_harness):
    h = make_harness(tree({}))
    h.config["prompts"]["agent"] = 'Up to {max_children} subtasks. Example: {"mode": "worker"}'
    assert h.agent_system_prompt() == 'Up to 4 subtasks. Example: {"mode": "worker"}'


# ---------------------------------------------------------------------------
# Context flowing down: intent, invariants, peers, the preceding section
# ---------------------------------------------------------------------------

def test_worker_sees_intent_invariants_and_peers(make_harness):
    h = make_harness(tree({"Root": lambda: manager(("A", {}), ("B", {}), invariants="Use British spelling")}))
    h.run()
    prompt_b = h.client.prompts("agent", "B")[0]
    assert "Intent: Root" in prompt_b                       # chain of command
    assert "Use British spelling" in prompt_b               # shared invariants
    assert "1.2. B [WORKING]  <-- you" in prompt_b          # knows its own place
    assert "1.1. A [DONE]" in prompt_b and "Report: A summary" in prompt_b  # finished peer's report
    assert "scope of A" in prompt_b
    assert "End of the Preceding Section" in prompt_b and "A content" in prompt_b


def test_preceding_section_crosses_branches(make_harness):
    h = make_harness(tree({
        "Root": lambda: manager(("Part1", {}), ("Part2", {})),
        "Part2": lambda: manager(("Part2a", {})),
    }))
    h.run()
    prompt = h.client.prompts("agent", "Part2a")[0]
    assert "End of the Preceding Section" in prompt
    assert "Part1 content" in prompt
    assert "Intent: scope of Part2" in prompt


def test_depends_on_is_zero_based_and_shares_code_stubs(make_harness):
    h = make_harness(tree({
        "Root": lambda: manager(("Lib", {"file_path": "lib.py"}), ("App", {"depends_on": [0]})),
        "Lib": lambda: worker(content="def helper(x: int) -> int:\n    return x\n"),
    }))
    h.run()
    assert h.nodes["1.2"].depends_on == ["1.1"]
    assert "def helper(x: int) -> int: ..." in h.client.prompts("agent", "App")[0]
    assert "0-based" in json.dumps(tt.AgentResponse.model_json_schema())


# ---------------------------------------------------------------------------
# Reports flowing up: the manager judges and sends revisions
# ---------------------------------------------------------------------------

def test_review_sees_reports_not_full_text(make_harness):
    h = make_harness(tree({
        "Root": lambda: manager(("A", {})),
        "A": lambda: worker(content="one two three", summary="three words"),
    }))
    h.run()
    review = h.client.prompts("review", "Root")[0]
    assert "Word Count (wc -w): 3" in review
    assert "Summary: three words" in review
    assert "one two three" not in review


def test_revision_reruns_only_the_flagged_child_with_feedback(make_harness):
    reviews = iter([reject(("1.2", "expand B to 500 words")), approve("all good")])
    h = make_harness(tree({"Root": lambda: manager(("A", {}), ("B", {}))}, review=lambda: next(reviews)))
    h.run()
    assert h.client.tasks("agent") == ["Root", "A", "B", "B"]
    revised = h.client.prompts("agent", "B")[1]
    assert "expand B to 500 words" in revised
    assert "Your Previous Submission:\nB content" in revised
    assert h.nodes["1"].status == "DONE" and h.nodes["1"].summary == "all good"


def test_revision_to_a_manager_child_reaches_its_review(make_harness):
    root_reviews = iter([reject(("1.1", "section too thin")), approve()])

    def script(role, task, user):
        if role == "review":
            if task == "Root":
                return next(root_reviews)
            if "section too thin" in user:
                return reject(("1.1.1", "add examples"))
            return approve()
        if task == "Root":
            return manager(("Branch", {}))
        if task == "Branch":
            return manager(("Leaf", {}))
        return worker(content=f"{task} content")

    h = make_harness(script, max_review_rounds=2)
    h.run()
    assert "add examples" in h.client.prompts("agent", "Leaf")[1]
    assert h.nodes["1"].status == "DONE"


def test_manager_fails_after_review_rounds_exhausted(make_harness, capsys):
    h = make_harness(
        tree({"Root": lambda: manager(("A", {}))}, review=lambda: reject(("1.1", "still wrong"))),
        max_review_rounds=3,
    )
    h.run()
    assert len(h.client.tasks("review")) == 3
    assert h.client.tasks("agent").count("A") == 3  # first run + 2 revisions
    assert h.nodes["1"].status == "FAILED"
    assert "still wrong" in h.nodes["1"].last_feedback
    assert "Node 1 (Root) failed" in capsys.readouterr().out


def test_reviews_disabled_approves_directly(make_harness):
    h = make_harness(tree({"Root": lambda: manager(("A", {}), ("B", {}))}), max_review_rounds=0)
    h.run()
    assert h.client.tasks("review") == []
    assert h.nodes["1"].status == "DONE"
    assert h.nodes["1"].summary == "A summary B summary"


# ---------------------------------------------------------------------------
# Verification and failures
# ---------------------------------------------------------------------------

def test_failed_verify_is_retried_then_reported(make_harness):
    h = make_harness(tree({
        "Root": lambda: manager(("Script", {"file_path": "s.sh", "verify_command": "exit 1"})),
    }))
    h.run()
    script_node = h.nodes["1.1"]
    assert script_node.status == "FAILED" and script_node.verify_status == "failed"
    assert "Verify command `exit 1` failed" in h.client.prompts("agent", "Script")[1]
    assert "Verification: FAILED" in h.client.prompts("review", "Root")[0]


def test_bad_model_output_fails_node_without_stopping_run(make_harness):
    h = make_harness(tree({"Root": lambda: manager(("A", {}), ("B", {})), "A": lambda: "not json"}))
    h.run()
    assert h.nodes["1.1"].status == "FAILED"
    assert "rejected" in h.nodes["1.1"].last_feedback
    assert h.nodes["1.2"].status == "DONE"
    assert "[FAILED]" in h.client.prompts("review", "Root")[0]


def test_verify_timeout_is_a_failure_not_a_crash(make_harness, monkeypatch):
    h = make_harness(tree({}))

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("cmd", 60)

    monkeypatch.setattr(tt.subprocess, "run", timeout)
    passed, feedback = h.run_verify_command("sleep 999")
    assert passed is False
    assert "timed out" in feedback


# ---------------------------------------------------------------------------
# Path containment
# ---------------------------------------------------------------------------

def test_escaping_subtask_file_paths_are_dropped(make_harness, tmp_path):
    outside = tmp_path / "abs_evil.txt"
    h = make_harness(tree({"Root": lambda: manager(
        ("A", {"file_path": "../evil.txt"}), ("B", {"file_path": str(outside)}),
    )}))
    h.run()
    assert h.nodes["1.1"].file_path is None and h.nodes["1.2"].file_path is None
    assert not (tmp_path / "evil.txt").exists() and not outside.exists()


def test_escaping_worker_file_path_is_rejected_and_retried(make_harness, tmp_path):
    responses = iter([worker(file_path="../evil.txt"), worker(file_path="ok.txt")])
    h = make_harness(tree({"Root": lambda: next(responses)}))
    h.run()
    assert not (tmp_path / "evil.txt").exists()
    assert (h.workspace / "ok.txt").read_text() == "text"


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
# Checkpoint / resume
# ---------------------------------------------------------------------------

def test_resume_reruns_unfinished_nodes_and_keeps_manager_decisions(make_harness):
    plans = {"Root": lambda: manager(("A", {}), ("B", {})), "B": lambda: "not json"}
    first = make_harness(tree(plans))
    first.run()
    assert first.nodes["1.2"].status == "FAILED"

    resumed = make_harness(tree({}), resume=True, workspace=first.workspace)
    assert resumed.root_prompt == "Root"
    assert resumed.nodes["1"].status == "PENDING"  # its aggregate included B's failure
    resumed.run()
    assert resumed.client.tasks("agent") == ["B"]  # A is kept, Root does not re-split
    assert all(n.status == "DONE" for n in resumed.nodes.values())
    assert resumed.nodes["1"].output == "A content\n\nB content"


def test_resume_ignores_fields_from_older_checkpoints(make_harness):
    h = make_harness(tree({}))
    h.save_checkpoint()
    data = json.loads(h.checkpoint_path.read_text())
    data["nodes"]["1"]["assumptions"] = ["from an older version"]
    h.checkpoint_path.write_text(json.dumps(data))
    resumed = make_harness(tree({}), resume=True, workspace=h.workspace)
    assert resumed.nodes["1"].title == "Root"


def test_resume_rejects_old_checkpoint_format(make_harness, tmp_path):
    ws = tmp_path / "old"
    ws.mkdir()
    (ws / ".tasktree_state.json").write_text(json.dumps({"1": {}}))
    with pytest.raises(ValueError, match="predates resume support"):
        make_harness(tree({}), resume=True, workspace=ws)


def test_resume_without_checkpoint_errors(make_harness, tmp_path):
    with pytest.raises(FileNotFoundError):
        make_harness(tree({}), resume=True, workspace=tmp_path / "empty")


# ---------------------------------------------------------------------------
# Output budget and truncation
# ---------------------------------------------------------------------------

def budget_words(prompt):
    return int(re.search(r"Output Budget: .*?about ([\d,]+) words", prompt).group(1).replace(",", ""))


def expected_words(tokens, thinking):
    tokens -= tt.RESPONSE_OVERHEAD_TOKENS
    if thinking:
        tokens = int(tokens * (1 - tt.THINKING_RESERVE))
    return int(tokens * tt.WORDS_PER_TOKEN)


def test_output_budget_follows_max_tokens_and_thinking(make_harness):
    on = make_harness(tree({}), max_tokens=8192)
    on.run()
    assert budget_words(on.client.prompts("agent", "Root")[0]) == expected_words(8192, thinking=True)

    off = make_harness(tree({}), max_tokens=8192, enable_thinking=False)
    off.run()
    assert budget_words(off.client.prompts("agent", "Root")[0]) == expected_words(8192, thinking=False)


def test_output_budget_shrinks_to_fit_context_window(make_harness):
    h = make_harness(tree({}), max_tokens=32768, context_window=6000)
    h.run()
    prompt = h.client.prompts("agent", "Root")[0]
    assert 0 < budget_words(prompt) < expected_words(6000, thinking=True)
    assert h.client.request_kwargs[0]["max_tokens"] < 6000


def test_truncated_worker_is_told_to_split(make_harness):
    responses = iter([truncated(response_tokens=40), manager(("A", {}), ("B", {}))])
    h = make_harness(tree({"Root": lambda: next(responses)}))
    h.run()
    retry = h.client.prompts("agent", "Root")[1]
    assert "cut off after about 30 words" in retry
    assert 'You MUST choose mode="manager"' in retry
    assert h.nodes["1"].children_ids == ["1.1", "1.2"]
    assert h.nodes["1"].status == "DONE"


def test_truncated_worker_at_max_depth_is_told_to_condense(make_harness):
    responses = iter([truncated(response_tokens=40), worker(content="short")])
    h = make_harness(tree({"Root": lambda: next(responses)}), max_depth=0)
    h.run()
    retry = h.client.prompts("agent", "Root")[1]
    assert "cannot split" in retry and "Cover the essentials" in retry
    assert h.nodes["1"].output == "short"


def test_thinking_overflow_retries_the_call_with_thinking_off(make_harness):
    responses = iter([truncated(response_tokens=2, thinking_tokens=50), worker(content="answer")])
    h = make_harness(tree({"Root": lambda: next(responses)}))
    h.run()
    first, second = h.client.request_kwargs
    assert "extra_body" not in first
    assert second["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    # Same call retried, not a failed attempt: the prompt is unchanged and no split is demanded
    prompt_1, prompt_2 = h.client.prompts("agent", "Root")
    assert prompt_1 == prompt_2
    assert h.nodes["1"].output == "answer"
    assert "retrying with thinking off" in (h.workspace / ".tasktree.log").read_text()


def test_truncated_review_accepts_the_work(make_harness):
    h = make_harness(tree({"Root": lambda: manager(("A", {}))}, review=lambda: truncated(response_tokens=10)))
    h.run()
    assert h.nodes["1"].status == "DONE"


# ---------------------------------------------------------------------------
# Run log
# ---------------------------------------------------------------------------

def test_log_streams_prompt_thinking_and_response(make_harness):
    def script(role, task, user):
        return [
            NS(choices=[NS(delta=NS(content=None, reasoning_content="Let me "), finish_reason=None)]),
            NS(choices=[NS(delta=NS(content=None, reasoning_content="think."), finish_reason=None)]),
            NS(choices=[NS(delta=NS(content=worker(content="the essay")), finish_reason="stop")]),
        ]

    h = make_harness(script)
    h.run()
    log = (h.workspace / ".tasktree.log").read_text()
    assert "tasktree started: Root" in log
    assert "node 1 | agent | thinking on" in log
    assert "--- prompt ---\nRoot Goal: Root" in log
    assert "--- thinking ---\nLet me think." in log
    assert "--- response ---\n" in log and "the essay" in log
    assert "finish: stop | ~2 thinking + ~1 response tokens" in log
    assert "node 1 DONE: Root" in log
    # Kept out of the workspace's per-node commits
    tracked = subprocess.run(["git", "ls-files"], cwd=h.workspace, capture_output=True, text=True).stdout
    assert ".tasktree.log" not in tracked
    assert ".tasktree.log" in (h.workspace / ".git/info/exclude").read_text().splitlines()


def test_log_can_be_disabled(make_harness):
    h = make_harness(tree({}), log_file="")
    h.run()
    assert not (h.workspace / ".tasktree.log").exists()


def test_log_appends_across_resume(make_harness):
    first = make_harness(tree({"Root": lambda: "not json"}))
    first.run()
    resumed = make_harness(tree({}), resume=True, workspace=first.workspace)
    resumed.run()
    log = (first.workspace / ".tasktree.log").read_text()
    assert log.index("tasktree started") < log.index("tasktree resumed")


def test_detect_context_window_reads_llama_cpp_props(monkeypatch):
    requested = []

    def fake_urlopen(url, timeout):
        requested.append(url)
        return io.BytesIO(json.dumps({"default_generation_settings": {"n_ctx": 65536}}).encode())

    monkeypatch.setattr(tt.urllib.request, "urlopen", fake_urlopen)
    assert tt.detect_context_window("http://localhost:8092/v1") == 65536
    assert requested == ["http://localhost:8092/props"]

    def unreachable(url, timeout):
        raise OSError("connection refused")

    monkeypatch.setattr(tt.urllib.request, "urlopen", unreachable)
    assert tt.detect_context_window("http://localhost:11434/v1") is None


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def test_unknown_rc_keys_are_reported(tmp_path):
    rc = tmp_path / "rc"
    rc.write_text(
        "[model]\nenable_thinking = false\n"
        "[harness]\nmax_plan_revisions = 2\nmax_review_rounds = 3\n"
        "[prompts]\nplanner = old\nagent = new\n"
        "[extras]\nx = 1\n"
    )
    assert tt.find_unknown_rc_keys(rc) == ["[harness] max_plan_revisions", "[prompts] planner", "[extras]"]
    config = tt.load_config(rc)
    assert config["max_review_rounds"] == 3
    assert config["enable_thinking"] is False
    assert config["prompts"]["agent"] == "new"
