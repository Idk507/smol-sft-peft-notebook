# set up

# Install Pytorch & other libraries
!pip install "torch==2.1.2" tensorboard
 
# Install Hugging Face libraries
!pip install  --upgrade \
  "transformers==4.38.2" \
  "datasets==2.16.1" \
  "accelerate==0.26.1" \
  "evaluate==0.4.1" \
  "bitsandbytes==0.42.0" \
  "trl==0.7.11" \
  "peft==0.8.2"

import torch; assert torch.cuda.get_device_capability()[0] >= 8, 'Hardware not supported for Flash Attention'
# install flash-attn
!pip install ninja packaging
!MAX_JOBS=4 pip install flash-attn --no-build-isolation --upgrade

from huggingface_hub import login
 
login(
  token="", # ADD YOUR TOKEN HERE
  add_to_git_credential=True
)


#Create and prepare the dataset

from datasets import load_dataset
 
# Load Dolly Dataset.
dataset = load_dataset("philschmid/dolly-15k-oai-style", split="train")
 
print(dataset[3]["messages"])

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
 
# Hugging Face model id
model_id = "google/gemma-7b"
tokenizer_id = "philschmid/gemma-tokenizer-chatml"
 
# BitsAndBytesConfig int-4 config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)
 
# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config
)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
tokenizer.padding_side = 'right' # to prevent warnings


from peft import LoraConfig
 
# LoRA config based on QLoRA paper & Sebastian Raschka experiment
peft_config = LoraConfig(
        lora_alpha=8,
        lora_dropout=0.05,
        r=6,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
)

from transformers import TrainingArguments
 
args = TrainingArguments(
    output_dir="gemma-7b-dolly-chatml", # directory to save and repository id
    num_train_epochs=3,                     # number of training epochs
    per_device_train_batch_size=2,          # batch size per device during training
    gradient_accumulation_steps=2,          # number of steps before performing a backward/update pass
    gradient_checkpointing=True,            # use gradient checkpointing to save memory
    optim="adamw_torch_fused",              # use fused adamw optimizer
    logging_steps=10,                       # log every 10 steps
    save_strategy="epoch",                  # save checkpoint every epoch
    bf16=True,                              # use bfloat16 precision
    tf32=True,                              # use tf32 precision
    learning_rate=2e-4,                     # learning rate, based on QLoRA paper
    max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
    warmup_ratio=0.03,                      # warmup ratio based on QLoRA paper
    lr_scheduler_type="constant",           # use constant learning rate scheduler
    push_to_hub=False,                       # push model to hub
    report_to="tensorboard",                # report metrics to tensorboard
)


from trl import SFTTrainer
 
max_seq_length = 1512 # max sequence length for model and packing of the dataset
 
trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    peft_config=peft_config,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    packing=True,
    dataset_kwargs={
        "add_special_tokens": False, # We template with special tokens
        "append_concat_token": False, # No need to add additional separator token
    }
)


# start training, the model will be automatically saved to the hub and the output directory
trainer.train()
 
# save model
trainer.save_model()

# Test Model and run Inference

# free the memory again
del model
del trainer
torch.cuda.empty_cache()

import torch
from peft import AutoPeftModelForCausalLM
from transformers import  AutoTokenizer, pipeline
 
peft_model_id = "gemma-7b-dolly-chatml"
 
# Load Model with PEFT adapter
tokenizer = AutoTokenizer.from_pretrained(peft_model_id)
model = AutoPeftModelForCausalLM.from_pretrained(peft_model_id, device_map="auto", torch_dtype=torch.float16)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
# get token id for end of conversation
eos_token = tokenizer("<|im_end|>",add_special_tokens=False)["input_ids"][0]


prompts = [
    "What is the capital of Germany? Explain why thats the case and if it was different in the past?",
    "Write a Python function to calculate the factorial of a number.",
    "A rectangular garden has a length of 25 feet and a width of 15 feet. If you want to build a fence around the entire garden, how many feet of fencing will you need?",
    "What is the difference between a fruit and a vegetable? Give examples of each.",
]
 
def test_inference(prompt):
    prompt = pipe.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=1024, do_sample=True, temperature=0.7, top_k=50, top_p=0.95, eos_token_id=eos_token)
    return outputs[0]['generated_text'][len(prompt):].strip()
 
 
for prompt in prompts:
    print(f"    prompt:\n{prompt}")
    print(f"    response:\n{test_inference(prompt)}")
    print("-"*50)

