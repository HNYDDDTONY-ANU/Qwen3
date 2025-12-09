# Interactive chat with the model
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

tokenizer = AutoTokenizer.from_pretrained("./models/Qwen3-0.6B")
model = AutoModelForCausalLM.from_pretrained("./models/Qwen3-0.6B")

# Set to True for thinking mode, False for non-thinking mode
thinking_mode = False  # Change to False to disable thinking

messages = []
print("Start chatting with the model! Type 'exit' to quit.")

while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break
    messages.append({"role": "user", "content": user_input})
    
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=thinking_mode,  # Enable or disable thinking mode
        thinking_budget=1024 if thinking_mode else 0,  # Set thinking budget
    ).to(model.device)
    
    # Create streamer for streaming output
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    print("Qwen3-0.6B: ", end="", flush=True)
    outputs = model.generate(**inputs, max_new_tokens=300, do_sample=True, temperature=0.7, eos_token_id=[151645, 151643], streamer=streamer)
    
    # Decode the full response for history
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    
    print()  # New line after streaming
    messages.append({"role": "assistant", "content": response})