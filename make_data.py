from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Step 1: Specify the model name or path
model_name = "meta-llama/Meta-Llama-3-70B"  # Replace with the path to your model or Hugging Face repo name

# Step 2: Load the model and tokenizer
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto",
    quantization_config=quantization_config,
    trust_remote_code=True
)

# Step 3: Define the function to ask a question
def ask_question(question):
    inputs = tokenizer(question, return_tensors="pt").to("cuda")
    outputs = model.generate(inputs["input_ids"], max_length=200)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# Step 4: Ask a specific question
question = "how do you reverse an array in python?"
print(f"Question: {question}")
answer = ask_question(question)
print(f"Answer: {answer}")
