import os
import torch
import wandb
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, 
    DataCollatorForLanguageModeling, get_linear_schedule_with_warmup,
    CodeLlamaTokenizer, LlamaForCausalLM, LlamaConfig
)
from torch.optim import AdamW
from torch.utils.data import DataLoader
from datetime import datetime
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from accelerate import Accelerator
import evaluate

# Configuration settings
class Config:
    model_name = "codellama/CodeLlama-7b-hf"  # Using CodeLlama as a strong code-specialized base
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
    sequence_length = 2048  # CodeLlama supports longer contexts
    train_batch_size = 4
    eval_batch_size = 8
    learning_rate = 5e-5
    weight_decay = 0.01
    num_train_epochs = 3
    warmup_steps = 1000
    gradient_accumulation_steps = 8
    fp16 = True  # Use mixed precision training
    output_dir = "./codegen-model"
    eval_steps = 500
    save_steps = 1000
    logging_steps = 100
    use_lora = True  # Parameter-efficient fine-tuning
    lora_r = 16
    lora_alpha = 32
    lora_dropout = 0.05
    use_wandb = False  # Set to True to use Weights & Biases for tracking
    project_name = "code-gen-llm"
    run_name = f"codegen-{datetime.now().strftime('%Y%m%d-%H%M')}"
    # Evaluation metrics
    use_code_eval = True
    evaluate_on_humaneval = True


def setup_logging(config):
    """Configure logging including optional W&B integration"""
    if config.use_wandb:
        wandb.init(
            project=config.project_name,
            name=config.run_name,
            config=vars(config)
        )
    
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(f"{config.output_dir}/logs", exist_ok=True)

    return get_logger(f"{config.output_dir}/logs/training.log")


def get_logger(log_file=None):
    """Setup logging to both console and optional file"""
    def log(msg, level="INFO"):
        timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
        formatted_msg = f"{timestamp} [{level}] {msg}"
        print(formatted_msg)
        if log_file:
            with open(log_file, "a") as f:
                f.write(formatted_msg + "\n")
    return log


def filter_by_language(example, languages):
    """Filter dataset entries by programming language"""
    if "language" in example:
        return example["language"] in languages
    elif "lang" in example:
        return example["lang"] in languages
    # For datasets with code but no language field - assume it's acceptable
    return True


def prepare_datasets(config, tokenizer, logger):
    """Load and prepare multiple datasets for training and evaluation"""
    logger("Loading and preparing datasets...")
    
    all_train_datasets = []
    all_eval_datasets = []
    
    for dataset_name in config.dataset_names:
        logger(f"Processing dataset: {dataset_name}")
        split = config.dataset_splits.get(dataset_name, "train[:1%]")
        
        try:
            # Load dataset with appropriate split
            dataset = load_dataset(dataset_name, split=split)
            
            # Filter by programming languages if applicable
            if any(field in dataset.column_names for field in ["language", "lang"]):
                logger(f"Filtering {dataset_name} by languages: {config.dataset_languages}")
                dataset = dataset.filter(
                    lambda x: filter_by_language(x, config.dataset_languages)
                )
            
            # Standardize column names - ensure we have a "code" column
            if "code" not in dataset.column_names:
                if "content" in dataset.column_names:
                    dataset = dataset.rename_column("content", "code")
                elif "source_code" in dataset.column_names:
                    dataset = dataset.rename_column("source_code", "code")
            
            # Split into train/eval (90/10)
            dataset = dataset.train_test_split(test_size=0.1)
            train_dataset, eval_dataset = dataset["train"], dataset["test"]
            
            logger(f"Added {len(train_dataset)} training samples and {len(eval_dataset)} evaluation samples from {dataset_name}")
            
            all_train_datasets.append(train_dataset)
            all_eval_datasets.append(eval_dataset)
            
        except Exception as e:
            logger(f"Error loading dataset {dataset_name}: {str(e)}", level="ERROR")
    
    # Combine all datasets
    if all_train_datasets:
        train_dataset = concatenate_datasets(all_train_datasets)
        eval_dataset = concatenate_datasets(all_eval_datasets)
        
        logger(f"Final combined dataset: {len(train_dataset)} training samples, {len(eval_dataset)} evaluation samples")
        
        # Tokenize datasets
        train_tokenized = tokenize_dataset(train_dataset, tokenizer, config.sequence_length, logger)
        eval_tokenized = tokenize_dataset(eval_dataset, tokenizer, config.sequence_length, logger)
        
        return train_tokenized, eval_tokenized
    else:
        raise ValueError("No datasets were successfully loaded.")


def tokenize_dataset(dataset, tokenizer, max_length, logger):
    """Tokenize dataset with appropriate settings for causal language modeling"""
    logger(f"Tokenizing dataset with {len(dataset)} samples...")
    
    def tokenize_function(examples):
        # Format code examples with appropriate beginning and end tokens
        texts = examples["code"]
        tokenized = tokenizer(
            texts, 
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt"
        )
        # For causal language modeling, we need the labels to be the same as the inputs
        tokenized["labels"] = tokenized["input_ids"].clone()
        return tokenized
    
    # Use batched processing for efficiency
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=os.cpu_count(),
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset"
    )
    
    logger(f"Tokenization complete. Dataset size: {len(tokenized_dataset)}")
    return tokenized_dataset


def compute_metrics(eval_pred, tokenizer):
    """Compute evaluation metrics for code generation"""
    predictions, labels = eval_pred
    
    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # Replace -100 in labels with pad token id as we can't decode -100
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Initialize metrics
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")
    
    # Calculate metrics
    rouge_output = rouge.compute(predictions=decoded_preds, references=decoded_labels)
    bleu_output = bleu.compute(predictions=decoded_preds, references=decoded_labels)
    
    # Combine metrics
    results = {
        "rouge1": rouge_output["rouge1"],
        "rouge2": rouge_output["rouge2"],
        "rougeL": rouge_output["rougeL"],
        "bleu": bleu_output["bleu"]
    }
    
    return results


def load_model_and_tokenizer(config, logger):
    """Load appropriate model and tokenizer based on configuration"""
    logger(f"Loading base model: {config.model_name}")
    
    try:
        # Load tokenizer first
        tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
        
        # Ensure the tokenizer has padding token for batch processing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger("Set padding token to EOS token")
        
        # Load model with appropriate quantization if needed
        if config.use_lora:
            # For LoRA, we can use 8-bit quantization to reduce memory usage
            model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                torch_dtype=torch.float16 if config.fp16 else torch.float32,
                load_in_8bit=True,
                trust_remote_code=True,
                device_map="auto"
            )
            
            # Prepare model for LoRA fine-tuning
            model = prepare_model_for_kbit_training(model)
            
            # Add LoRA adapters
            logger("Adding LoRA adapters to model")
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
                bias="none"
            )
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
            
        else:
            # Regular loading without quantization for full fine-tuning
            model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                torch_dtype=torch.float16 if config.fp16 else torch.float32,
                trust_remote_code=True
            )
        
        logger(f"Model loaded: {config.model_name}")
        return model, tokenizer
    
    except Exception as e:
        logger(f"Error loading model: {str(e)}", level="ERROR")
        raise


def setup_training(config, model, tokenizer, train_dataset, eval_dataset, logger):
    """Configure training arguments and trainer"""
    logger("Setting up training configuration")
    
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        num_train_epochs=config.num_train_epochs,
        warmup_steps=config.warmup_steps,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        fp16=config.fp16,
        evaluation_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        logging_dir=f"{config.output_dir}/logs",
        logging_steps=config.logging_steps,
        report_to="wandb" if config.use_wandb else "none",
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    
    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False  # We're doing causal language modeling, not masked
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, tokenizer),
    )
    
    return trainer


def evaluate_on_humaneval(model, tokenizer, config, logger):
    """Evaluate model on HumanEval benchmark if possible"""
    if not config.evaluate_on_humaneval:
        return {}
    
    try:
        from human_eval.data import write_jsonl, read_problems
        from human_eval.evaluation import evaluate_functional_correctness
        import tempfile
        
        logger("Evaluating model on HumanEval benchmark...")
        
        problems = read_problems()
        
        # Generate completions for each problem
        completions = []
        
        for problem_id, problem in tqdm(problems.items(), desc="Generating solutions"):
            prompt = problem["prompt"]
            
            # Encode the prompt
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            # Generate completion
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_new_tokens=512,
                    temperature=0.2,
                    top_p=0.95,
                    do_sample=True,
                    num_return_sequences=1
                )
            
            # Decode and extract only the newly generated text
            generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            completions.append({
                "task_id": problem_id,
                "completion": generated_text
            })
        
        # Write completions to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as temp_file:
            write_jsonl(temp_file.name, completions)
            completion_file = temp_file.name
        
        # Evaluate
        results = evaluate_functional_correctness(completion_file)
        os.unlink(completion_file)  # Clean up
        
        logger(f"HumanEval results: pass@1={results['pass@1']:.4f}")
        return results
        
    except ImportError:
        logger("human_eval package not installed. Skipping HumanEval evaluation.", level="WARNING")
        return {}
    except Exception as e:
        logger(f"Error during HumanEval evaluation: {str(e)}", level="ERROR")
        return {}


def save_model(model, tokenizer, config, logger):
    """Save the trained model and tokenizer"""
    final_output_dir = f"{config.output_dir}/final"
    os.makedirs(final_output_dir, exist_ok=True)
    
    logger(f"Saving model and tokenizer to {final_output_dir}")
    
    try:
        # If using LoRA, save adapter weights
        if config.use_lora:
            model.save_pretrained(final_output_dir)
        else:
            # Full model saving
            model.save_pretrained(
                final_output_dir, 
                save_function=torch.save,
                max_shard_size="10GB"  # Shard large models
            )
        
        # Save tokenizer
        tokenizer.save_pretrained(final_output_dir)
        logger("Model and tokenizer saved successfully")
        
        # Save configuration
        with open(f"{final_output_dir}/training_config.txt", "w") as f:
            for key, value in vars(config).items():
                f.write(f"{key}: {value}\n")
    
    except Exception as e:
        logger(f"Error saving model: {str(e)}", level="ERROR")


def evaluate_model(model, tokenizer, config, logger):
    """Perform comprehensive model evaluation beyond training metrics"""
    logger("Performing comprehensive model evaluation...")
    
    evaluation_results = {}
    
    # Evaluate on HumanEval if configured
    if config.evaluate_on_humaneval:
        humaneval_results = evaluate_on_humaneval(model, tokenizer, config, logger)
        evaluation_results.update(humaneval_results)
    
    # Add additional evaluation methods here
    # ...
    
    # Log all evaluation results
    logger(f"Evaluation results: {evaluation_results}")
    
    if config.use_wandb:
        wandb.log(evaluation_results)
    
    return evaluation_results


def train_model():
    """Main training function"""
    # Initialize configuration
    config = Config()
    
    # Setup logging
    logger = setup_logging(config)
    logger(f"Starting code generation model training - {config.run_name}")
    
    try:
        # Check for GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger(f"Using device: {device}")
        
        if device.type == "cpu" and not config.use_lora:
            logger("Warning: Training on CPU without LoRA will be very slow", level="WARNING")
            logger("Consider enabling use_lora in config for parameter-efficient training")
        
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(config, logger)
        
        # Prepare datasets
        train_dataset, eval_dataset = prepare_datasets(config, tokenizer, logger)
        
        # Setup training
        trainer = setup_training(
            config, model, tokenizer, train_dataset, eval_dataset, logger
        )
        
        # Train model
        logger("Starting training...")
        trainer.train()
        
        # Save model
        save_model(model, tokenizer, config, logger)
        
        # Evaluate model
        evaluation_results = evaluate_model(model, tokenizer, config, logger)
        
        logger(f"Training complete. Model saved to '{config.output_dir}/final'")
        
        # Clean up
        if config.use_wandb:
            wandb.finish()
        
        return evaluation_results
        
    except Exception as e:
        logger(f"Fatal error: {str(e)}", level="ERROR")
        import traceback
        logger(traceback.format_exc(), level="ERROR")
        
        if config.use_wandb:
            wandb.finish()


def generate_code(prompt, model_path="./codegen-model/final", max_length=512):
    """Generate code using the trained model"""
    try:
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        
        # Move model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # Prepare input
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Generate
        outputs = model.generate(
            inputs.input_ids,
            max_length=max_length,
            temperature=0.7,
            top_p=0.95,
            top_k=50,
            do_sample=True,
            num_return_sequences=1
        )
        
        # Decode and return
        generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_code
    
    except Exception as e:
        print(f"Error generating code: {str(e)}")
        return None


if __name__ == "__main__":
    try:
        # Check CUDA availability
        if torch.cuda.is_available():
            print(f"CUDA is available. Found {torch.cuda.device_count()} device(s).")
            for i in range(torch.cuda.device_count()):
                print(f"Device {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("CUDA is not available. Training will use CPU only.")
        
        # Run training
        train_model()
        
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        print(traceback.format_exc())
