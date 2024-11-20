import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import json
import torch
import csv

# Path to the folder containing the JSONL files
folder_path = "batch_inputs"

from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer


print(torch.cuda.is_available())
print(torch.version.cuda)


model_name = "Qwen/Qwen2.5-Coder-14B-Instruct-GPTQ-Int4"
# model_name = "Qwen/Qwen2.5-Coder-32B-Instruct" #use this for better model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

import pandas as pd

df = pd.read_csv('train.csv')

def clean_code_block(code):
    code = code.strip()
    if code.startswith("```python"):
        code = code[9:]
    if code.endswith("```"):
        code = code[:-3]
    return code.strip()

# Create output.csv with headers if it doesn't exist
if not os.path.exists('output.csv'):
    with open('output.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['original', 'clone'])

for output in df['output']:
    prompt = "make some code that functions the same as the following code:" + output + " but is not the same. just give one example and only return the code."
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

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        streamer=TextStreamer(tokenizer),
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    response = clean_code_block(response)
    # Write response to CSV
    with open('output.csv', 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([output, response])
