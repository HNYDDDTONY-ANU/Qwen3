# Interactive chat with the model
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import torch

# Streaming markdown code is moved to a separate module for clarity
from streaming_markdown import RICH_AVAILABLE, create_streamer, STREAMING_MARKDOWN, STREAMING_RENDER_EVERY, STREAMING_MIN_INTERVAL_MS
import argparse

if RICH_AVAILABLE:
    # Local imports for Live/Console to create a Live context when needed
    from rich.console import Console
    from rich.live import Live
else:
    Console = None
    Live = None

parser = argparse.ArgumentParser(description="Interactive chat with Qwen3-0.6B with optional streaming Markdown rendering")
parser.add_argument('--no-markdown', action='store_true', help='Disable streaming markdown rendering (fall back to plain streaming)')
parser.add_argument('--render-every', type=int, default=STREAMING_RENDER_EVERY, help='Render every N tokens for incremental markdown render (smaller = more frequent)')
parser.add_argument('--min-interval-ms', type=int, default=STREAMING_MIN_INTERVAL_MS, help='Minimum ms between two incremental renders (debounce)')
parser.add_argument('--label', type=str, default='Qwen3-0.6B：', help='Model label to display before response')
parser.add_argument('--max-new-tokens', type=int, default=300, help='Maximum new tokens to generate (max_new_tokens)')
parser.add_argument('--context-length', type=int, default=None, help='Model context length in tokens (defaults to tokenizer.model_max_length or 32768)')
args = parser.parse_args()

# CLI-overridden values
STREAMING_MARKDOWN = False if args.no_markdown else STREAMING_MARKDOWN
STREAMING_RENDER_EVERY = int(args.render_every)
STREAMING_MIN_INTERVAL_MS = int(args.min_interval_ms)
MODEL_LABEL = args.label
MAX_NEW_TOKENS = args.max_new_tokens

tokenizer = AutoTokenizer.from_pretrained("./models/Qwen3-0.6B")
model = AutoModelForCausalLM.from_pretrained("./models/Qwen3-0.6B")

# Determine model context length from tokenizer or CLI arg
MODEL_CONTEXT_LENGTH = int(args.context_length) if args.context_length else (getattr(tokenizer, 'model_max_length', None) or 32768)

# Set to True for thinking mode, False for non-thinking mode
thinking_mode = False  # Change to False to disable thinking

# Configuration: enable streaming markdown rendering (incremental)
# - True: attempt incremental markdown rendering in terminal using rich
# - False: keep existing streaming (plain text) behavior
STREAMING_MARKDOWN = True

# NOTE: streaming parameters are provided by the imported module `streaming_markdown`

# Streamer and normalization code were moved to streaming_markdown.py

messages = []
print("开始与模型聊天！输入 'exit' 退出。输入 'new' 开始新对话。")
if STREAMING_MARKDOWN and not RICH_AVAILABLE:
    print("警告：未检测到 rich 库；将回退到普通流式输出。若想启用 Markdown 流式渲染，请运行：pip install rich")

while True:
    user_input = input("你：")
    if user_input.lower() == 'exit':
        break
    elif user_input.lower() == 'new':
        messages = []
        print("开始新对话！")
        continue
    messages.append({"role": "user", "content": user_input})
    
    # Build tokenized inputs but enforce context length by trimming message history if needed
    def ensure_within_context(messages_local):
        buffer = 16
        while True:
            tmp_inputs = tokenizer.apply_chat_template(
                messages_local,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                enable_thinking=thinking_mode,
                thinking_budget=1024 if thinking_mode else 0,
            )
            input_len = tmp_inputs["input_ids"].shape[-1]
            if input_len + MAX_NEW_TOKENS <= MODEL_CONTEXT_LENGTH - buffer:
                return tmp_inputs
            if len(messages_local) == 0:
                return tmp_inputs
            # Drop the oldest message and retry
            messages_local.pop(0)

    inputs = ensure_within_context(messages)
    inputs = inputs.to(model.device)
    # _inputs built and safe for generation
    
    # END ensure_within_context
    
    # Replace original unbounded tokenization call
    
    
    # Create streamer for streaming output
    if STREAMING_MARKDOWN and RICH_AVAILABLE:
        console = Console()
        # live area will update markdown/text incrementally
        with Live(console=console, refresh_per_second=6) as live:
            # NOTE: create streamer via helper
            prefill_len = inputs["input_ids"].shape[-1] if "input_ids" in inputs else 0
            streamer = create_streamer(tokenizer, inputs=inputs, label=MODEL_LABEL, render_every=STREAMING_RENDER_EVERY, min_interval_ms=STREAMING_MIN_INTERVAL_MS, skip_prompt=True, console=console, live=live, skip_special_tokens=True)
            outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=True, temperature=0.7, eos_token_id=[151645, 151643], streamer=streamer)
    else:
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        print(MODEL_LABEL, end="")
        outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=True, temperature=0.7, eos_token_id=[151645, 151643], streamer=streamer)
    
    # Decode the full response for history
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    
    print()  # New line after streaming
    messages.append({"role": "assistant", "content": response})