"""
SyntasticAI Configuration

This module contains the default configuration settings for the SyntasticAI model.
You can customize these settings based on your hardware capabilities and requirements.
"""

class Config:
    """Configuration settings for SyntasticAI model training and inference."""
    
    # Model settings
    model_name = "codellama/CodeLlama-7b-hf"  # Base model
    use_8bit_quantization = True  # Enable for lower VRAM usage
    use_4bit_quantization = False  # Even lower VRAM, more quality tradeoff
    use_lora = True  # Parameter-efficient fine-tuning
    
    # LoRA configuration
    lora_r = 16  # LoRA attention dimension
    lora_alpha = 32  # LoRA alpha parameter
    lora_dropout = 0.05  # Dropout probability for LoRA layers
    lora_target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "down_proj", "up_proj"
    ]
    
    # Dataset configuration
    dataset_names = [
        "codeparrot/github-code",   # General code
        "bigcode/the-stack",        # More diverse code
        "codeparrot/codeparrot-clean-valid"  # High-quality code
    ]
    dataset_splits = {
        "codeparrot/github-code": "train[:5%]",  # Increase percentage as resources allow
        "bigcode/the-stack": "train[:1%]",
        "codeparrot/codeparrot-clean-valid": "train[:10%]"
    }
    dataset_languages = ["python", "javascript", "java", "go", "rust"]  # Languages to include
    
    # Training configuration
    sequence_length = 2048  # Maximum sequence length
    train_batch_size = 4  # Per device batch size for training
    eval_batch_size = 8  # Per device batch size for evaluation
    learning_rate = 5e-5  # Learning rate
    weight_decay = 0.01  # Weight decay for AdamW optimizer
    num_train_epochs = 3  # Number of training epochs
    warmup_steps = 1000  # Warmup steps for learning rate scheduler
    gradient_accumulation_steps = 8  # Number of updates steps to accumulate before backward pass
    fp16 = True  # Use mixed precision training
    bf16 = False  # Use bfloat16 precision (if available)
    gradient_checkpointing = True  # Use gradient checkpointing to save memory
    
    # Output configuration
    output_dir = "./syntasticai-model"  # Directory to save model
    eval_steps = 500  # Evaluate every N steps
    save_steps = 1000  # Save model every N steps
    logging_steps = 100  # Log every N steps
    
    # Experiment tracking
    use_wandb = False  # Set to True to use Weights & Biases for tracking
    project_name = "syntasticai"  # W&B project name
    run_name = None  # Will be auto-generated if None
    
    # Evaluation
    use_code_eval = True  # Evaluate on code quality metrics
    evaluate_on_humaneval = True  # Evaluate on HumanEval benchmark
    
    # Inference settings
    default_temperature = 0.7  # Temperature for sampling
    default_top_p = 0.95  # Top-p sampling parameter
    default_top_k = 50  # Top-k sampling parameter
    default_num_return_sequences = 1  # Number of sequences to return
    default_max_new_tokens = 512  # Maximum number of tokens to generate
    
    # Multi-GPU training
    use_deepspeed = False  # Use DeepSpeed for distributed training
    deepspeed_config = "./configs/ds_config.json"  # DeepSpeed config file
    
    # Advanced settings
    seed = 42  # Random seed for reproducibility
    push_to_hub = False  # Push model to Hugging Face Hub
    hub_model_id = None  # Hugging Face Hub model ID
    hub_token = None  # Hugging Face Hub token
    
    def __init__(self, **kwargs):
        """Initialize config with optional overrides."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")
