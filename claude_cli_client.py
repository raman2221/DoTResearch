"""Cloud-model client that sends DoT's "GPT" calls through the Claude Code CLI.

It mirrors the slice of the OpenAI client that askLLM uses, so the pipeline runs on a
Claude subscription without a paid OpenAI key.
"""
import json
import os
import shutil
import subprocess
import tempfile
from types import SimpleNamespace

# Keeps the allocator's strong-vs-weak ranking: GPT-4-class -> sonnet, small GPTs -> haiku.
GPT_TO_CLAUDE = {
    'gpt-4-turbo': 'sonnet',
    'gpt-4': 'sonnet',
    'gpt-4o': 'sonnet',
    'gpt-4o-mini': 'haiku',
    'gpt-3.5-turbo': 'haiku',
}

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
CLI_TIMEOUT_SECONDS = 300


class ClaudeCliClient:
    def __init__(self, executable=None):
        self.executable = executable or shutil.which("claude")
        if self.executable is None:
            raise FileNotFoundError("The `claude` CLI is not on PATH. Install Claude Code or set DOT_CLOUD_BACKEND=openai.")
        # An empty working dir keeps project CLAUDE.md files out of the model's context.
        self.workdir = tempfile.mkdtemp(prefix="dot-claude-")
        # Same call shape as OpenAI's client.chat.completions.create, so askLLM needs no Claude branch.
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, model, messages, **_unsupported):
        # The CLI has no temperature/max_tokens flags; DoT uses temperature=1, which is Claude's default.
        system_prompt, conversation = split_messages(messages)
        command = self.build_command(GPT_TO_CLAUDE[model], system_prompt)
        return to_openai_response(self.run(command, conversation))

    def build_command(self, claude_model, system_prompt):
        return [
            self.executable, "-p",
            "--output-format", "json",
            "--model", claude_model,
            "--system-prompt", system_prompt,
            "--tools", "",
            "--strict-mcp-config",
            "--no-session-persistence",
        ]

    def run(self, command, prompt):
        # Claude Code refuses to start nested inside another session unless this marker is cleared.
        env = {key: value for key, value in os.environ.items() if key != "CLAUDECODE"}
        completed = subprocess.run(
            command, input=prompt, capture_output=True, text=True, encoding="utf-8",
            cwd=self.workdir, env=env, timeout=CLI_TIMEOUT_SECONDS,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"claude CLI failed ({completed.returncode}): {completed.stderr.strip()}")
        output = json.loads(completed.stdout)
        if output.get("is_error"):
            raise RuntimeError(f"claude CLI returned an error: {output.get('result')}")
        return output


def split_messages(messages):
    system_parts = [m['content'] for m in messages if m['role'] == 'system']
    turns = [m for m in messages if m['role'] != 'system']
    system_prompt = "\n\n".join(system_parts) or DEFAULT_SYSTEM_PROMPT
    if len(turns) == 1:
        return system_prompt, turns[0]['content']
    # The CLI takes one prompt, so earlier turns are replayed as a labelled transcript.
    transcript = "\n\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in turns)
    return system_prompt, transcript


def to_openai_response(cli_output):
    usage = cli_output["usage"]
    prompt_tokens = (usage["input_tokens"] + usage.get("cache_creation_input_tokens", 0)
                     + usage.get("cache_read_input_tokens", 0))
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=cli_output["result"]))],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=usage["output_tokens"]),
    )
