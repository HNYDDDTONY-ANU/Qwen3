# Utilities and an incremental Markdown streamer (rich) for terminal rendering
# To be used by chat scripts

import os
import re
import textwrap
import time

from transformers import TextStreamer

# Attempt to import Rich library components for enhanced terminal rendering
# If not available, set RICH_AVAILABLE to False for fallback behavior
try:
    from rich.console import Console, Group
    from rich.markdown import Markdown
    from rich.live import Live
    from rich.text import Text
    RICH_AVAILABLE = True
except Exception:
    RICH_AVAILABLE = False

# Default configuration values, can be overridden via environment variables
STREAMING_MARKDOWN = True
STREAMING_RENDER_EVERY = int(os.environ.get("STREAMING_RENDER_EVERY", "1"))
STREAMING_MIN_INTERVAL_MS = int(os.environ.get("STREAMING_MIN_INTERVAL_MS", "100"))


def _markdown_renderable(text: str) -> bool:
    """Heuristic check to decide whether to apply Markdown rendering.
    Check if code fences are properly balanced.
    """
    # Find all potential fence sequences (``` or ~~~)
    fence_positions = []
    for fence_type in ['```', '~~~']:
        start = 0
        while True:
            pos = text.find(fence_type, start)
            if pos == -1:
                break
            fence_positions.append(pos)
            start = pos + len(fence_type)
    
    # If no fences found, it's renderable
    if not fence_positions:
        return True
    
    # Sort all fence positions
    fence_positions.sort()
    
    # Track fence state
    fence_open = False
    for pos in fence_positions:
        fence_open = not fence_open
    
    # Only renderable if no unclosed fences remain
    return not fence_open


def normalize_markdown(text: str, final: bool = False) -> str:
    """Normalize Markdown text to render better in terminals.

    - Dedent common indentation
    - If final=True, convert indented code blocks to fenced code blocks and remove
      leading spaces before table rows.
    """
    if not text:
        return text

    # Remove common leading whitespace from all lines
    normalized = textwrap.dedent(text)

    # Replace regular spaces after emphasis markers (bold/italic) with
    # non-breaking spaces to avoid wrapping immediately after the emphasis.
    # We must not perform this replacement inside fenced code blocks, so
    # split the text on triple-backtick fences and only modify the segments
    # that are outside code fences.
    def _protect_emphasis_spaces(s: str) -> str:
        # Patterns: **bold**␣, __bold__␣, *italic*␣, _italic_␣
        patterns = [r"(\*\*[^\*]+\*\*)( )",
                    r"(__[^_]+__)( )",
                    r"(\*[^\*]+\*)( )",
                    r"(_[^_]+_)( )"]
        for p in patterns:
            s = re.sub(p, lambda m: m.group(1) + "\u00A0", s)
        return s

    # If not final, apply protection to reduce undesirable wraps during incremental render
    if not final:
        parts = normalized.split('```')
        for i in range(len(parts)):
            if i % 2 == 0:
                parts[i] = _protect_emphasis_spaces(parts[i])
        return '```'.join(parts)

    # Process lines to convert indented code blocks to fenced blocks
    lines = normalized.splitlines()
    out_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # Check for indented code block (4 spaces or tab)
        if line.startswith("    ") or line.startswith("\t"):
            code_block = []
            # Collect all lines of the code block
            while i < len(lines) and (lines[i].startswith("    ") or lines[i].startswith("\t") or lines[i] == ""):
                if lines[i] == "":
                    code_block.append("")
                else:
                    # Remove leading indentation
                    code_block.append(lines[i].lstrip(" \t"))
                i += 1
            # Add fenced code block markers
            out_lines.append("```")
            out_lines.extend(code_block)
            out_lines.append("```")
        else:
            out_lines.append(line)
            i += 1

    normalized = "\n".join(out_lines)

    # Clean up table rows by removing leading spaces before pipes
    normalized = re.sub(r'(?m)^[ \t]+\|', lambda m: m.group(0).lstrip(), normalized)

    # After final normalization, also protect spaces after emphasis markers
    # outside of fenced code blocks so the renderer doesn't wrap immediately
    # after bold/italic markers.
    parts = normalized.split('```')
    for i in range(len(parts)):
        if i % 2 == 0:
            parts[i] = re.sub(r"(\*\*[^\*]+\*\*)( )", lambda m: m.group(1) + "\u00A0", parts[i])
            parts[i] = re.sub(r"(__[^_]+__)( )", lambda m: m.group(1) + "\u00A0", parts[i])
            parts[i] = re.sub(r"(\*[^\*]+\*)( )", lambda m: m.group(1) + "\u00A0", parts[i])
            parts[i] = re.sub(r"(_[^_]+_)( )", lambda m: m.group(1) + "\u00A0", parts[i])
    return '```'.join(parts)


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

    def __init__(self, tokenizer, live: Live, console: Console = None, prefill_len: int = 0, render_every: int = 8, min_interval_ms: int = 100, skip_prompt: bool = True, label: str = '', messages_ref=None, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        self.tokenizer = tokenizer
        # Use provided console or create one with soft_wrap disabled
        self.console = console or Console(soft_wrap=False)
        self.live = live
        # Buffer to accumulate token IDs for decoding
        self.buffer_ids = []
        # Counter for tokens since last render
        self._since_render = 0
        # Length of prefill tokens to skip
        self.prefill_len = int(prefill_len or 0)
        # Total tokens seen so far
        self._total_seen = 0
        # Whether to skip prompt tokens
        self.skip_prompt = bool(skip_prompt)
        # Label to display before output
        self.label = str(label or '')
        # Render frequency in tokens
        self.render_every = int(render_every or 8)
        # Minimum interval between renders in milliseconds
        self.min_interval_ms = int(min_interval_ms or 100)
        # Timestamp of last render
        self._last_render_time = 0.0
        # Reference to messages list for dynamic context updates
        self.messages_ref = messages_ref

    def put(self, token_ids, scores=None, **kwargs):
        # Convert token_ids to a flat list of integers
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

        # Flatten nested lists
        flat = []
        for item in ids:
            if isinstance(item, list):
                flat.extend(item)
            else:
                flat.append(item)

        if not flat:
            return

        original_len = len(flat)
        # Skip prefill tokens if configured
        if self.skip_prompt and self.prefill_len:
            to_skip = max(0, self.prefill_len - self._total_seen)
            if to_skip >= len(flat):
                self._total_seen += original_len
                return
            flat = flat[to_skip:]
        self._total_seen += original_len

        # Add new tokens to buffer
        self.buffer_ids.extend(flat)
        self._since_render += len(flat)
        now = time.time() * 1000.0

        # Check if we should render: enough tokens or EOS token reached, and debounce time passed
        if (self._since_render >= self.render_every or any(x == self.tokenizer.eos_token_id for x in flat)) and (now - self._last_render_time) >= self.min_interval_ms:
            self._render()
            self._since_render = 0
            self._last_render_time = now

    def _render(self):
        # Decode accumulated tokens to text
        text = self.tokenizer.decode(self.buffer_ids, skip_special_tokens=True)
        # Normalize markdown for incremental rendering
        text = normalize_markdown(text, final=False)
        
        # Dynamically update messages reference if available
        if self.messages_ref is not None:
            try:
                # Update the last assistant message with current streaming content
                if self.messages_ref and self.messages_ref[-1]["role"] == "assistant":
                    self.messages_ref[-1]["content"] = text
            except (IndexError, KeyError, TypeError):
                # Silently handle any errors in dynamic updating
                pass
        
        try:
            # Determine rendering mode based on label and markdown detectability
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
            # Fallback to plain text if rendering fails
            self.live.update(Text(text))

    def end(self):
        # Final decode of all accumulated tokens
        text = self.tokenizer.decode(self.buffer_ids, skip_special_tokens=True)
        # Normalize markdown for final rendering
        text = normalize_markdown(text, final=True)
        
        # Ensure final complete response is saved to messages
        if self.messages_ref is not None:
            try:
                if self.messages_ref and self.messages_ref[-1]["role"] == "assistant":
                    self.messages_ref[-1]["content"] = text
            except (IndexError, KeyError, TypeError):
                # Fallback: if dynamic update fails, append new message
                try:
                    self.messages_ref.append({"role": "assistant", "content": text})
                except Exception:
                    pass
        
        try:
            # Render final output
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
            # Fallback to plain text
            self.live.update(Text(text))
        # Do not call super().end() to avoid extra newline


def create_streamer(tokenizer, inputs=None, label='', render_every=None, min_interval_ms=None, skip_prompt=True, console=None, live=None, messages_ref=None, **kwargs):
    """Convenience helper to create a RichMarkdownStreamer with sensible defaults.

    - tokenizer: required
    - inputs: optional model inputs; if present, `prefill_len` is taken from `inputs['input_ids'].shape[-1]`
    - messages_ref: optional reference to messages list for dynamic context updates
    """
    # Calculate prefill length from inputs if provided
    prefill_len = 0
    if inputs is not None and "input_ids" in inputs:
        try:
            prefill_len = int(inputs["input_ids"].shape[-1])
        except Exception:
            prefill_len = 0

    # Use provided values or defaults
    render_every = int(render_every or STREAMING_RENDER_EVERY)
    min_interval_ms = int(min_interval_ms or STREAMING_MIN_INTERVAL_MS)
    # Create a Console with soft_wrap disabled by default to avoid Rich auto-wrapping
    console = console or (Console(soft_wrap=False) if RICH_AVAILABLE else None)
    # Instantiate and return the streamer
    return RichMarkdownStreamer(tokenizer, live=live, console=console, prefill_len=prefill_len, render_every=render_every, min_interval_ms=min_interval_ms, skip_prompt=skip_prompt, label=label, messages_ref=messages_ref, **kwargs)


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
