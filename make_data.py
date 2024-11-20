import os
import json

# Path to the folder containing the JSONL files
folder_path = "batch_inputs"

from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-Coder-32B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)



for file_name in os.listdir(folder_path):
    if file_name.endswith(".jsonl") and file_name.startswith("batch_"):
        file_path = os.path.join(folder_path, file_name)
        
        with open(file_path, "r") as file:
            for line in file:
                data = json.loads(line)
                prompt = data.get("prompt", "")
                
                example_code = prompt.split(" but is not the same.")[0].split(": ", 1)[1].strip()
                
                print(prompt)

                prompt = "write a quick sort algorithm."
                messages = [
                    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

                generated_ids = model.generate(**model_inputs, max_new_tokens=512)

                response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                print(response)
                break
