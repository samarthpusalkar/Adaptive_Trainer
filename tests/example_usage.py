

from adaptive_trainer import train_adaptively

# import os
# os.environ['HF_TOKEN'] = '<HF_TOKEN>'
# os.environ['WANDB_API_KEY'] = '<WANDB_API_KEY>'

## example using the nvidia/HelpSteer (CC-BY-4.0) dataset
# Credits to the dataset authors at nvidia:
# @misc{wang2023helpsteer,
#       title={HelpSteer: Multi-attribute Helpfulness Dataset for SteerLM}, 
#       author={Zhilin Wang and Yi Dong and Jiaqi Zeng and Virginia Adams and Makesh Narsimhan Sreedhar and Daniel Egert and Olivier Delalleau and Jane Polak Scowcroft and Neel Kant and Aidan Swope and Oleksii Kuchaiev},
#       year={2023},
#       eprint={2311.09528},
#       archivePrefix={arXiv},
#       primaryClass={cs.CL}
# }

system_prompts = {
    'attention': """""",
    'ideas': """You are a helpful ai assistant, you need to help answer the user's query.""",
    'attention': """"""
}


data_processing_function = lambda sample, context_mode: (sample['prompt'] + (f"""\nhelpfulness: {sample['helpfulness']}\ncorrectness: {sample['correctness']}\ncoherence: {sample['coherence']}\ncomplexity: {sample['complexity']}\nverbosity: {sample['verbosity']}""" if context_mode else ""), sample['response'])

# Configure datasets (remains the same)
datasets_config = {
    'ideas': ["nvidia/HelpSteer:|:15000"],
    'attention': [],
    'both': [],
    'misc': [],
    'data_processing_function':data_processing_function
    # 'data_processing_function_attention':None,
    # 'data_processing_function_ideas':data_processing_function
    # 'data_processing_function_nvidia/HelpSteer':data_processing_function
}


datasets_kwargs={"nvidia/HelpSteer": {'context_mode':'ideas', 'train_split':'train', 'val_split':'validation'}}
dataset_specific_system_prompts={
    "nvidia/HelpSteer": """In requested parameter values of the following parameters will be provided that have the following meaning:
    helpfulness: higher value means highly helpful
    correctness: higher value means the response should be correct, value of 0 means the response is completely inaccurate
    coherence: if the language of the response is expected to follow the requested instructions properly the value is high it can be very low if the response is inaccurate
    complexity: when the response required to fulfill user's query is expected to be highly complex when the value is high
    verbosity: the value is high when the response is expected to be very verbose and low when user needs consice answer
    """
}

# Configure training parameters
training_config = {
    'max_length_token': 3072,
    'batch_size': 3,
    'gradient_accumulation_steps': 5,
    'learning_rate': 2e-5,
    'num_epochs': 2,
    'run_name': 'Test_Run_llama3.2',
    'wandb_project': 'Testing_Library_NvidiaSteer',
    'attn_impl': 'sdpa',
    'fp16': True,
    'optimizer': 'paged_adamw_8bit'
}

adaptive_loss_config = {
    'top_k': 6,
    'adaptive_log_steps': 100
}

# Run training
model_path = train_adaptively(
    model_name="meta-llama/Llama-3.2-1B-Instruct",
    datasets_config=datasets_config,
    datasets_kwargs =datasets_kwargs,
    dataset_specific_system_prompts=dataset_specific_system_prompts,
    output_dir="llama3.2_1B_nvidia_helpSteer",
    huggingface_repo="YourHuggingFaceID/llama3.2_1B_nvidia_helpSteer",
    system_prompts=system_prompts,
    training_config=training_config,
    adaptive_loss_config=adaptive_loss_config
)

print(f"Training complete. Model saved to {model_path}")
