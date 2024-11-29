import os
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM

# 警告レベルを抑制
warnings.filterwarnings("ignore", category=UserWarning)

# Set environment to use GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("nvidia/Mistral-NeMo-Minitron-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained("nvidia/Mistral-NeMo-Minitron-8B-Instruct").to("cuda")

# Define the directory containing the captions.txt files
base_dir = r"C:\Users\user\Desktop\git\dataset\outputs\captions"

def summarize_text(text):
    # Send the user message with a strict instruction to provide only the summary
    messages = [
        {"role": "user", "content": f"Summarize the following text strictly, providing a detailed summary of the actions mentioned. Do not include any additional commentary, explanation, or interpretation: {text}"}
    ]


    # Tokenize the chat messages
    tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")

    # Generate the summary with max_new_tokens set to 100
    outputs = model.generate(tokenized_chat, max_new_tokens=100, stop_strings=["<extra_id_1>"], tokenizer=tokenizer)

    # Decode the output and extract only the generated part
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_part = full_output.split("<extra_id_1>")[2].strip()

    # Remove the "Assistant" string
    generated_part = generated_part.replace("Assistant", "").strip()

    return generated_part

def find_and_summarize_captions(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file == "captions.txt":
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                    summary = summarize_text(text)
                    print(f"{summary}\n")

# Find and summarize all captions.txt files in the base directory
find_and_summarize_captions(base_dir)