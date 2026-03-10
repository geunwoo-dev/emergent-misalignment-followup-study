import os
import subprocess

def evaluate_all_checkpoints(output_dir, trait="evil"):
    """Evaluate all checkpoints in a training output directory."""
    if not os.path.exists(output_dir):
        print(f"Output directory {output_dir} does not exist.")
        return
    
    # Find all checkpoint directories
    checkpoints = sorted([
        d for d in os.listdir(output_dir) 
        if d.startswith('checkpoint-')
    ], key=lambda x: int(x.split('-')[1]))
    
    if not checkpoints:
        print("No checkpoints found.")
        return
    
    print(f"Found {len(checkpoints)} checkpoints: {checkpoints}")
    
    # Create evaluation directory
    eval_dir = os.path.join(output_dir, 'evaluations')
    os.makedirs(eval_dir, exist_ok=True)
    
    # Evaluate each checkpoint
    for ckpt in checkpoints:
        step = ckpt.split('-')[1]
        ckpt_path = os.path.join(output_dir, ckpt)
        eval_output = os.path.join(eval_dir, f'{trait}_step_{step}.csv')
        
        # Skip if already evaluated
        if os.path.exists(eval_output):
            print(f"✓ Step {step} already evaluated")
            continue
            
        print(f"🔍 Evaluating checkpoint at step {step}...")
        
        cmd = [
            "python", "-m", "eval.eval_persona",
            "--model", ckpt_path,
            "--trait", trait,
            "--output_path", eval_output,
            "--judge_model", "gpt-4.1-mini-2025-04-14",
            "--version", "eval",
            "--n_per_question", "5"  # Reduce samples for faster evaluation
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"✅ Completed evaluation for step {step}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error evaluating step {step}: {e}")
            print(f"Command: {' '.join(cmd)}")

# Usage
output_dir = "./ckpt/Qwen2.5-7B-Instruct/qwen-evil_misaligned_2"
evaluate_all_checkpoints(output_dir, "evil")
