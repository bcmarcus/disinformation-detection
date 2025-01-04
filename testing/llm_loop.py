import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline 

torch.random.manual_seed(0) 
model = AutoModelForCausalLM.from_pretrained( 
    "google/gemma-2-2b-it",  
    device_map="cuda",  
    trust_remote_code=True,
    torch_dtype=torch.bfloat16, 
    max_length=4096
) 

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it") 

pipe = pipeline( 
    "text-generation", 
    model=model, 
    tokenizer=tokenizer, 
) 

generation_args = { 
    "max_new_tokens": 4096, 
    "return_full_text": False, 
    "temperature": 0.0, 
    "do_sample": False, 
}

messages = []

template = """<bos><start_of_turn>user
"""

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    # messages.append({"role": "user", "content": user_input})
    template = template + user_input + "<end_of_turn>\n<start_of_turn>model\n"

    output = pipe(
        template,
        **generation_args
    )
    print(output[0]['generated_text'])

    template = template + output[0]['generated_text'] + "<end_of_turn>\n<bos><start_of_turn>user\n"
