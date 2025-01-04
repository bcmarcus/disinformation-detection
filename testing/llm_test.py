import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline 

torch.random.manual_seed(0) 
model = AutoModelForCausalLM.from_pretrained( 
    "google/gemma-2-2b-it",  
    device_map="cuda",  
    trust_remote_code=True,
    torch_dtype=torch.bfloat16, 
    max_length=200
) 

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it") 

# messages = [
#     {"role": "system", "content": "You are a helpful AI assistant."}, 
#     {"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"}, 
#     {"role": "assistant", "content": "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey."}, 
#     {"role": "user", "content": "What about solving an 2x + 3 = 7 equation?"}, 
# ] 

pipe = pipeline( 
    "text-generation", 
    model=model, 
    tokenizer=tokenizer, 
) 

generation_args = { 
    "max_new_tokens": 500, 
    "return_full_text": False, 
    "temperature": 0.0, 
    "do_sample": False, 
} 

template = """<bos><start_of_turn>user
    What is 5x4?<end_of_turn>
    <start_of_turn>model
"""

template = """<bos><start_of_turn>user
hi<end_of_turn>
<start_of_turn>model
"""

print (template)
output = pipe(template, **generation_args) 
print(output[0]['generated_text']) 