from transformers import AutoModelForCausalLM, AutoTokenizer

# Step 1: Specify the model name or path
model_name = "deepseek-ai/DeepSeek-Coder-V2-Instruct"  # Replace with the path to your model or Hugging Face repo name

# Step 2: Load the model and tokenizer
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",           # Automatically selects the appropriate device (GPU/CPU)
    torch_dtype="auto",          # Uses the optimal data type for your hardware
    load_in_8bit=True            # Optional: Load in 8-bit precision to save memory
)

# Step 3: Define the function to ask a question
def ask_question(question):
    # Tokenize the input question
    inputs = tokenizer(question, return_tensors="pt").to("cuda")  # Move inputs to GPU if available
    # Generate the answer
    outputs = model.generate(inputs["input_ids"], max_length=200)  # Adjust max_length as needed
    # Decode and return the answer
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# Step 4: Ask a specific question
question = "What is the capital of France?"  # Replace with your own question
print(f"Question: {question}")
answer = ask_question(question)
print(f"Answer: {answer}")
