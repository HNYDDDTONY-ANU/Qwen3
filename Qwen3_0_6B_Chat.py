# Interactive chat with the model
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import torch

# Import streaming markdown utilities from a separate module for better organization
from streaming_markdown import RICH_AVAILABLE, create_streamer, STREAMING_MARKDOWN, STREAMING_RENDER_EVERY, STREAMING_MIN_INTERVAL_MS
import argparse

# Conditionally import Rich library components if available for enhanced terminal rendering
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
parser.add_argument('--max-new-tokens', type=int, default=1000, help='Maximum new tokens to generate (max_new_tokens)')
parser.add_argument('--context-length', type=int, default=None, help='Model context length in tokens (defaults to tokenizer.model_max_length or 32768)')
parser.add_argument('--device', type=str, default=None, help="Device to run the model on, e.g. 'cuda:0' or 'cpu'. If not set, auto-selects 'cuda:0' when available.")
args = parser.parse_args()

# Override default streaming configurations with CLI arguments
STREAMING_MARKDOWN = False if args.no_markdown else STREAMING_MARKDOWN
STREAMING_RENDER_EVERY = int(args.render_every)
STREAMING_MIN_INTERVAL_MS = int(args.min_interval_ms)
MODEL_LABEL = args.label
MAX_NEW_TOKENS = args.max_new_tokens

# Determine device: prefer user-provided, otherwise use CUDA if available
if args.device:
    device_str = args.device
else:
    device_str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_str)
print(f"Using device: {device}")

# Load the tokenizer and model from the local Qwen3-0.6B directory
tokenizer = AutoTokenizer.from_pretrained("./models/Qwen3-0.6B")
model = AutoModelForCausalLM.from_pretrained("./models/Qwen3-0.6B")
# Move model to target device
try:
    model.to(device)
except Exception:
    pass

# Ensure any remaining parameters/buffers are moved to the requested device
for name, param in model.named_parameters(recurse=True):
    if param.device != device:
        try:
            param.data = param.data.to(device)
        except Exception:
            pass
for name, buf in model.named_buffers(recurse=True):
    if buf.device != device:
        try:
            buf.data = buf.data.to(device)
        except Exception:
            pass


# Determine model context length from CLI argument or tokenizer default
MODEL_CONTEXT_LENGTH = int(args.context_length) if args.context_length else (getattr(tokenizer, 'model_max_length', None) or 32768)

# Enable or disable thinking mode for the model (affects token generation behavior)
thinking_mode = False  # Change to False to disable thinking

# Global configuration for streaming markdown rendering
# - True: Use Rich library for incremental markdown rendering in terminal
# - False: Fall back to plain text streaming
# STREAMING_MARKDOWN = True

# Initialize message history for conversation context
messages = []
print("开始与Qwen3-0.6B聊天！输入 'exit' 退出。输入 'new' 开始新对话。")
if STREAMING_MARKDOWN and not RICH_AVAILABLE:
    print("警告：未检测到 rich 库；将回退到普通流式输出。若想启用 Markdown 流式渲染，请运行：pip install rich")

# Main chat loop: continuously prompt for user input and generate responses
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break
    elif user_input.lower() == 'new':
        messages = []
        print("开始新对话！")
        continue
    messages.append({"role": "user", "content": user_input})
    # 立即添加空的助手回复占位符，流式输出时动态更新
    messages.append({"role": "assistant", "content": ""})
    
    # Function to build tokenized inputs while ensuring they fit within context length
    # Trims message history if necessary to prevent exceeding model limits
    def ensure_within_context(messages_local):
        buffer = 16  # Safety buffer to account for generation overhead
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
    # Move inputs to the same device as the model
    inputs = inputs.to(device)

    # Prepare streamer for real-time output rendering
    # Add a blank line to visually separate user input from model response
    print() 

    if STREAMING_MARKDOWN and RICH_AVAILABLE:
        # Create a Console with soft_wrap disabled to avoid auto line-wrapping
        console = Console(soft_wrap=False)
        # Use Rich Live display for incremental markdown rendering
        with Live(console=console, refresh_per_second=6) as live:
            # Create custom streamer with specified parameters
            prefill_len = inputs["input_ids"].shape[-1] if "input_ids" in inputs else 0
            streamer = create_streamer(
                tokenizer, 
                inputs=inputs, 
                label=MODEL_LABEL, 
                render_every=STREAMING_RENDER_EVERY, 
                min_interval_ms=STREAMING_MIN_INTERVAL_MS, 
                skip_prompt=True, 
                console=console, 
                live=live, 
                skip_special_tokens=True,
                messages_ref=messages)  # 传递messages引用
            
            # Generate response with sampling and specified parameters
            outputs = model.generate(
                **inputs, 
                max_new_tokens=MAX_NEW_TOKENS, 
                do_sample=True, 
                temperature=0.7, 
                eos_token_id=[151645, 151643], 
                streamer=streamer)
    else:
        # Fallback to plain text streaming if Rich is not available
        streamer = TextStreamer(
            tokenizer, 
            skip_prompt=True, 
            skip_special_tokens=True)
        
        print(MODEL_LABEL, end="")

        # Generate response with the same parameters
        outputs = model.generate(
            **inputs, 
            max_new_tokens=MAX_NEW_TOKENS, 
            do_sample=True, 
            temperature=0.7, 
            eos_token_id=[151645, 151643], 
            streamer=streamer)
        
        # 手动更新非流式输出的messages
        try:
            if messages and messages[-1]["role"] == "assistant":
                messages[-1]["content"] = response
        except (IndexError, KeyError, TypeError):
            # 备用方案：直接添加
            pass
    
    # Extract and decode the generated response from model outputs (excluding input tokens)
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    
    print()  # Ensure proper line spacing after streaming output
    # messages现在由streamer动态管理，无需再次添加