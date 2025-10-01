pip install trl transformers datasets accelerate peft

accelerate launch -m trl.scripts.sft ^
  --model_name_or_path "HuggingFaceTB/SmolLM3-3B-Base" ^
  --dataset_name "HuggingFaceTB/smoltalk2" ^
  --dataset_config "SFT" ^
  --output_dir "./smollm3-finetuned" ^
  --learning_rate 5e-5 ^
  --per_device_train_batch_size 2 ^
  --gradient_accumulation_steps 8 ^
  --max_steps 1000 ^
  --warmup_steps 100 ^
  --logging_steps 10 ^
  --save_steps 200 ^
  --bf16 True ^
  --max_seq_length 2048 ^
  --push_to_hub ^
  --hub_model_id "Dhanushkumar/smollm3-finetuned-smoltalk"
