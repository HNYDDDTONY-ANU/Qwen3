# Utilities and an incremental Markdown streamer (rich) for terminal rendering
# To be used by chat scripts

import os
import re
import textwrap
import time

from transformers import TextStreamer

# Optional rich imports
try:
    from rich.console import Console, Group
    from rich.markdown import Markdown
    from rich.live import Live
    from rich.text import Text
    RICH_AVAILABLE = True
except Exception:
    RICH_AVAILABLE = False

# Configuration defaults read from env variables (can be overridden by importing module)
STREAMING_MARKDOWN = True
STREAMING_RENDER_EVERY = int(os.environ.get("STREAMING_RENDER_EVERY", "3"))
STREAMING_MIN_INTERVAL_MS = int(os.environ.get("STREAMING_MIN_INTERVAL_MS", "100"))


def _markdown_renderable(text: str) -> bool:
    """Heuristic check to decide whether to apply Markdown rendering.
    Use even number of code fences as a quick sanity check.
    """
    fences = text.count("```")
    return (fences % 2) == 0


def normalize_markdown(text: str, final: bool = False) -> str:
    """Normalize Markdown text to render better in terminals.

    - Dedent common indentation
    - If final=True, convert indented code blocks to fenced code blocks and remove
      leading spaces before table rows.
    """
    if not text:
        return text

    normalized = textwrap.dedent(text)

    if not final:
        return normalized

    # Convert indented code blocks to fenced code blocks
    lines = normalized.splitlines()
    out_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("    ") or line.startswith("\t"):
            code_block = []
            while i < len(lines) and (lines[i].startswith("    ") or lines[i].startswith("\t") or lines[i] == ""):
                if lines[i] == "":
                    code_block.append("")
                else:
                    code_block.append(lines[i].lstrip(" \t"))
                i += 1
            out_lines.append("```")
            out_lines.extend(code_block)
            out_lines.append("```")
        else:
            out_lines.append(line)
            i += 1

    normalized = "\n".join(out_lines)

    # Remove leading spaces before pipe characters in table rows
    normalized = re.sub(r'(?m)^[ \t]+\|', lambda m: m.group(0).lstrip(), normalized)

    return normalized


class RichMarkdownStreamer(TextStreamer):
    """A Transformer TextStreamer that updates a rich.Live area incrementally.

    Parameters:
    - tokenizer: transformers tokenizer used for decoding
    - live: rich.Live instance used for incremental updates
    - console: rich.Console to use (optional)
    - prefill_len: number of tokens in the prompt (skip them)
    - render_every: render every N tokens during incremental updates
    - min_interval_ms: minimum time between two renders (debounce)
    - skip_prompt: whether to skip prefill tokens
    - label: an optional label to show (e.g., 'Qwen3-0.6B：')
    - Other args are passed to TextStreamer
    """

    def __init__(self, tokenizer, live: Live, console: Console = None, prefill_len: int = 0, render_every: int = 8, min_interval_ms: int = 100, skip_prompt: bool = True, label: str = '', *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        self.tokenizer = tokenizer
        self.console = console or Console()
        self.live = live
        self.buffer_ids = []
        self._since_render = 0
        self.prefill_len = int(prefill_len or 0)
        self._total_seen = 0
        self.skip_prompt = bool(skip_prompt)
        self.label = str(label or '')
        self.render_every = int(render_every or 8)
        self.min_interval_ms = int(min_interval_ms or 100)
        self._last_render_time = 0.0

    def put(self, token_ids, scores=None, **kwargs):
        try:
            if hasattr(token_ids, "cpu"):
                ids = token_ids.cpu().numpy().tolist()
            else:
                ids = list(token_ids)
        except Exception:
            try:
                ids = [int(x) for x in token_ids]
            except Exception:
                ids = []

        flat = []
        for item in ids:
            if isinstance(item, list):
                flat.extend(item)
            else:
                flat.append(item)

        if not flat:
            return

        original_len = len(flat)
        if self.skip_prompt and self.prefill_len:
            to_skip = max(0, self.prefill_len - self._total_seen)
            if to_skip >= len(flat):
                self._total_seen += original_len
                return
            flat = flat[to_skip:]
        self._total_seen += original_len

        self.buffer_ids.extend(flat)
        self._since_render += len(flat)
        now = time.time() * 1000.0

        if (self._since_render >= self.render_every or any(x == self.tokenizer.eos_token_id for x in flat)) and (now - self._last_render_time) >= self.min_interval_ms:
            self._render()
            self._since_render = 0
            self._last_render_time = now

    def _render(self):
        text = self.tokenizer.decode(self.buffer_ids, skip_special_tokens=True)
        text = normalize_markdown(text, final=False)
        try:
            if self.label:
                if _markdown_renderable(text):
                    self.live.update(Group(Text(self.label), Markdown(text)))
                else:
                    self.live.update(Group(Text(self.label), Text(text)))
            else:
                if _markdown_renderable(text):
                    self.live.update(Markdown(text))
                else:
                    self.live.update(Text(text))
        except Exception:
            self.live.update(Text(text))

    def end(self):
        text = self.tokenizer.decode(self.buffer_ids, skip_special_tokens=True)
        text = normalize_markdown(text, final=True)
        try:
            if self.label:
                if _markdown_renderable(text):
                    self.live.update(Group(Text(self.label), Markdown(text)))
                else:
                    self.live.update(Group(Text(self.label), Text(text)))
            else:
                if _markdown_renderable(text):
                    self.live.update(Markdown(text))
                else:
                    self.live.update(Text(text))
        except Exception:
            self.live.update(Text(text))
        try:
            super().end()
        except Exception:
            pass


def create_streamer(tokenizer, inputs=None, label='', render_every=None, min_interval_ms=None, skip_prompt=True, console=None, live=None, **kwargs):
    """Convenience helper to create a RichMarkdownStreamer with sensible defaults.

    - tokenizer: required
    - inputs: optional model inputs; if present, `prefill_len` is taken from `inputs['input_ids'].shape[-1]`
    """
    prefill_len = 0
    if inputs is not None and "input_ids" in inputs:
        try:
            prefill_len = int(inputs["input_ids"].shape[-1])
        except Exception:
            prefill_len = 0

    render_every = int(render_every or STREAMING_RENDER_EVERY)
    min_interval_ms = int(min_interval_ms or STREAMING_MIN_INTERVAL_MS)
    console = console or (Console() if RICH_AVAILABLE else None)
    return RichMarkdownStreamer(tokenizer, live=live, console=console, prefill_len=prefill_len, render_every=render_every, min_interval_ms=min_interval_ms, skip_prompt=skip_prompt, label=label, **kwargs)


__all__ = [
    "RICH_AVAILABLE",
    "STREAMING_MARKDOWN",
    "STREAMING_RENDER_EVERY",
    "STREAMING_MIN_INTERVAL_MS",
    "_markdown_renderable",
    "normalize_markdown",
    "RichMarkdownStreamer",
    "create_streamer",
]
