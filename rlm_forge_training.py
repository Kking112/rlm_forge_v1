import torch
import json
import re
import random
from typing import Optional
from dataclasses import dataclass
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

print(f"GPU: {torch.cuda.get_device_name(0)}")
# print(f"VRAM: {torch.cuda.get_device_properties(0). / 1e9:.1f} GB")
print(f"PyTorch: {torch.__version__}")

from rlm_forge.server.environment import RLMForgeEnvironment
from rlm_forge.models import RLMForgeAction

env = RLMForgeEnvironment()

# Run a quick episode
obs = env.reset(seed=1)
print(f"Task: {obs.task_description[:200]}...")
print(f"Available tools: {obs.available_functions}")

# Take a step — list files
obs2 = env.step(RLMForgeAction(code="print(list_dir())"))
print(f"\nStep 1 stdout: {obs2.stdout[:200]}")

# Finalize and get reward
obs3 = env.step(RLMForgeAction(code="FINAL()"))
print(f"\nBaseline reward (no implementation): {obs3.reward:.4f}")
print(f"Test results: {obs3.test_results}")

env.cleanup()


# Model config — adjust based on available VRAM
# MODEL_ID = "Qwen/Qwen2.5-Coder-32B-Instruct"  # 32B for H100
MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"  # Fallback for smaller GPUs
HF_TOKEN = ''
MAX_STEPS_PER_EPISODE = 6  # Max REPL interactions per episode
NUM_EPISODES_PER_PROMPT = 2  # GRPO group size (completions per prompt)
NUM_TRAINING_PROMPTS = 8  # 16 # Total unique prompts (episodes) for training
GRPO_EPOCHS = 2  # Training epochs over collected data
BATCH_SIZE = 2
GRAD_ACCUM = 4

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

# 4-bit quantization for 32B model on H100
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True,token=HF_TOKEN)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token



model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    # attn_implementation="flash_attention_2",
    token=HF_TOKEN
)

# LoRA config for efficient training
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

SYSTEM_PROMPT = """You are an expert Python developer. You are given a repository where a source file has been replaced with a broken stub. Your task is to explore the repository, understand the expected behavior from the tests, and rewrite the source file so all tests pass.

You interact via a Python REPL. Available functions:
- read_file(path) — Read a file from the repo
- list_dir(path='.') — List directory contents
- search(pattern, path='.') — Grep for a pattern
- write_file(path, content) — Write/create a file
- run_tests(test_path=None) — Run pytest on a test file
- FINAL() — Signal that your implementation is complete

Strategy:
1. Read the failing test file to understand expected behavior
2. Read other source files for context (imports, dependencies)
3. Write the implementation
4. Run tests to verify
5. Fix any failures
6. Call FINAL() when done

Output ONLY valid Python code. No markdown, no explanations — just code to execute."""


def build_prompt(task_description: str, failing_tests: list[str]) -> list[dict]:
    """Build the chat prompt for the initial observation."""
    user_msg = f"{task_description}\n\nFailing tests:\n" + "\n".join(failing_tests)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]


def extract_code_from_response(response: str) -> str:
    """Extract executable Python code from model response."""
    # Try to find code blocks first
    code_blocks = re.findall(r"```(?:python)?\n(.*?)```", response, re.DOTALL)
    if code_blocks:
        return "\n".join(code_blocks)
    # Otherwise treat the whole response as code
    lines = response.strip().split("\n")
    code_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and any(c in stripped for c in "=()[]{}:"):
            code_lines.append(line)
        elif stripped.startswith("#") or stripped.startswith("import") or stripped.startswith("from"):
            code_lines.append(line)
        elif not stripped:
            code_lines.append(line)
        else:
            code_lines.append(f"# {line}")
    return "\n".join(code_lines)


print("Prompt builder ready.")

@dataclass
class Trajectory:
    """A full multi-step episode trajectory for GRPO training."""
    prompt_text: str        # Tokenized prompt (system + task)
    completion_text: str    # All model outputs concatenated
    reward: float           # Final episode reward
    steps: int              # Number of steps taken
    seed: int               # Environment seed (for reproducibility)
    tests_passed: int
    tests_total: int


def run_episode(
    model,
    tokenizer,
    env: RLMForgeEnvironment,
    seed: int,
    max_steps: int = MAX_STEPS_PER_EPISODE,
    temperature: float = 0.7,
    max_new_tokens: int = 2048,
) -> Trajectory:
    """Run a single episode: generate code actions, execute them, collect trajectory."""
    obs = env.reset(seed=seed)

    messages = build_prompt(obs.task_description, obs.failing_tests or [])
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    all_completions = []  # All model outputs for this episode

    for step_i in range(max_steps):
        # Build the full conversation so far for the model
        if step_i > 0:
            # Add the observation as assistant feedback
            messages.append({"role": "user", "content": f"REPL output:\n{obs.stdout}\n{obs.stderr}"})

        # Generate next action
        full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=8192).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode only the new tokens
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        all_completions.append(response)

        # Add to conversation history
        messages.append({"role": "assistant", "content": response})

        # Extract and execute code
        code = extract_code_from_response(response)

        # Check if model wants to finalize
        if "FINAL()" in code:
            obs = env.step(RLMForgeAction(code=code))
            break
        else:
            obs = env.step(RLMForgeAction(code=code))

        if obs.done:
            break

    # If we exhausted steps without FINAL, force finalize
    if not obs.done:
        obs = env.step(RLMForgeAction(code="FINAL()"))

    # Build the full completion text (all model outputs joined)
    completion_text = "\n<|step|>\n".join(all_completions)

    reward = obs.reward or 0.0
    test_results = obs.test_results or {}

    return Trajectory(
        prompt_text=prompt_text,
        completion_text=completion_text,
        reward=reward,
        steps=step_i + 1,
        seed=seed,
        tests_passed=test_results.get("tests_passed", 0),
        tests_total=test_results.get("tests_total", 0),
    )


print("Episode runner ready.")


def collect_trajectories(
    model,
    tokenizer,
    num_prompts: int = NUM_TRAINING_PROMPTS,
    episodes_per_prompt: int = NUM_EPISODES_PER_PROMPT,
    temperature: float = 0.7,
) -> list[list[Trajectory]]:
    """Collect GRPO groups: multiple trajectories per unique prompt/seed."""
    env = RLMForgeEnvironment()
    all_groups = []

    for prompt_idx in range(num_prompts):
        seed = prompt_idx * 100  # Deterministic seeds
        group = []

        for ep_idx in range(episodes_per_prompt):
            print(f"  Prompt {prompt_idx+1}/{num_prompts}, Episode {ep_idx+1}/{episodes_per_prompt}...", end=" ")
            traj = run_episode(
                model, tokenizer, env,
                seed=seed,  # Same seed = same task for GRPO group
                temperature=temperature + 0.1 * ep_idx,  # Vary temperature for diversity
            )
            group.append(traj)
            print(f"reward={traj.reward:.3f}, steps={traj.steps}, "
                  f"tests={traj.tests_passed}/{traj.tests_total}")

        all_groups.append(group)

    env.cleanup()
    return all_groups

# GRPO Training configuration
grpo_config = GRPOConfig(
    output_dir="./rlm_forge_grpo_output",
    num_train_epochs=GRPO_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=1e-5,
    warmup_ratio=0.1,
    max_completion_length=4096,
    # max_prompt_length=4096,
    num_generations=NUM_EPISODES_PER_PROMPT,  # GRPO group size
    logging_steps=1,
    save_strategy="epoch",
    bf16=True,
    gradient_checkpointing=True,
    # GRPO-specific
    beta=0.1,  # KL penalty coefficient
    report_to="none",
)

# Collect pre-training baseline
print("=" * 60)
print("COLLECTING BASELINE TRAJECTORIES")
print("=" * 60)
baseline_groups = collect_trajectories(model, tokenizer)

# Summary stats
all_rewards = [t.reward for g in baseline_groups for t in g]
print(f"\nBaseline: mean_reward={sum(all_rewards)/len(all_rewards):.4f}, "
      f"min={min(all_rewards):.4f}, max={max(all_rewards):.4f}")





def trajectories_to_dataset(groups: list[list[Trajectory]]) -> Dataset:
    """Convert trajectory groups into a HuggingFace Dataset for GRPO training."""
    records = []
    for group in groups:
        prompt = group[0].prompt_text
        for traj in group:
            records.append({
                "prompt": prompt,
                "completion": traj.completion_text,
                "reward": traj.reward,
            })
    return Dataset.from_list(records)


def build_reward_fn(groups: list[list[Trajectory]]):
    """Build a reward function from pre-collected trajectories."""
    reward_map = {}
    for group in groups:
        for traj in group:
            key = traj.completion_text[:200]
            reward_map[key] = traj.reward

    def reward_fn(completions: list[str], **kwargs) -> list[float]:
        rewards = []
        for c in completions:
            key = c[:200]
            rewards.append(reward_map.get(key, 0.0))
        return rewards

    return reward_fn


# Build dataset from baseline trajectories
train_dataset = trajectories_to_dataset(baseline_groups)
print(f"Training dataset: {len(train_dataset)} examples")
print(f"Sample prompt length: {len(train_dataset[0]['prompt'])} chars")
print(f"Sample completion length: {len(train_dataset[0]['completion'])} chars")
print(f"Sample reward: {train_dataset[0]['reward']:.4f}")




# Build reward function from collected trajectories
reward_fn = build_reward_fn(baseline_groups)

# Prepare prompts dataset (unique prompts only, GRPO generates completions)
prompt_dataset = Dataset.from_list([
    {"prompt": group[0].prompt_text}
    for group in baseline_groups
])

# Initialize GRPO trainer
trainer = GRPOTrainer(
    model=model,
    args=grpo_config,
    train_dataset=prompt_dataset,
    reward_funcs=reward_fn,
    processing_class=tokenizer,
)

print("GRPO Trainer initialized. Starting training...")
trainer.train()
print("Training complete!")


# Collect post-training trajectories with the same seeds
print("=" * 60)
print("COLLECTING POST-TRAINING TRAJECTORIES")
print("=" * 60)
post_groups = collect_trajectories(model, tokenizer, temperature=0.5)

post_rewards = [t.reward for g in post_groups for t in g]
baseline_rewards = [t.reward for g in baseline_groups for t in g]

print(f"\n{'='*60}")
print(f"RESULTS COMPARISON")
print(f"{'='*60}")
print(f"Baseline: mean={sum(baseline_rewards)/len(baseline_rewards):.4f}, "
      f"max={max(baseline_rewards):.4f}")
print(f"Trained:  mean={sum(post_rewards)/len(post_rewards):.4f}, "
      f"max={max(post_rewards):.4f}")
print(f"Improvement: {(sum(post_rewards)/len(post_rewards) - sum(baseline_rewards)/len(baseline_rewards)):.4f}")

# Per-task comparison
print(f"\nPer-task breakdown:")
for i, (bg, pg) in enumerate(zip(baseline_groups, post_groups)):
    b_mean = sum(t.reward for t in bg) / len(bg)
    p_mean = sum(t.reward for t in pg) / len(pg)
    delta = p_mean - b_mean
    arrow = "\u2191" if delta > 0 else "\u2193" if delta < 0 else "\u2192"
    print(f"  Task {i}: baseline={b_mean:.3f} \u2192 trained={p_mean:.3f} ({arrow} {abs(delta):.3f})")

import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 1. Reward distribution: baseline vs trained
ax1 = axes[0]
ax1.hist(baseline_rewards, bins=20, alpha=0.6, label="Baseline", color="steelblue")
ax1.hist(post_rewards, bins=20, alpha=0.6, label="After GRPO", color="coral")
ax1.set_xlabel("Episode Reward")
ax1.set_ylabel("Count")
ax1.set_title("Reward Distribution")
ax1.legend()
ax1.axvline(np.mean(baseline_rewards), color="steelblue", linestyle="--", alpha=0.8)
ax1.axvline(np.mean(post_rewards), color="coral", linestyle="--", alpha=0.8)

# 2. Per-task mean reward comparison
ax2 = axes[1]
task_ids = list(range(len(baseline_groups)))
b_means = [np.mean([t.reward for t in g]) for g in baseline_groups]
p_means = [np.mean([t.reward for t in g]) for g in post_groups]
x = np.arange(len(task_ids))
width = 0.35
ax2.bar(x - width/2, b_means, width, label="Baseline", color="steelblue", alpha=0.8)
ax2.bar(x + width/2, p_means, width, label="After GRPO", color="coral", alpha=0.8)
ax2.set_xlabel("Task ID")
ax2.set_ylabel("Mean Reward")
ax2.set_title("Per-Task Reward Improvement")
ax2.legend()
ax2.set_xticks(x)

# 3. Test pass rate improvement
ax3 = axes[2]
b_pass_rates = [np.mean([t.tests_passed / max(t.tests_total, 1) for t in g]) for g in baseline_groups]
p_pass_rates = [np.mean([t.tests_passed / max(t.tests_total, 1) for t in g]) for g in post_groups]
ax3.bar(x - width/2, b_pass_rates, width, label="Baseline", color="steelblue", alpha=0.8)
ax3.bar(x + width/2, p_pass_rates, width, label="After GRPO", color="coral", alpha=0.8)
ax3.set_xlabel("Task ID")
ax3.set_ylabel("Test Pass Rate")
ax3.set_title("Test Pass Rate Improvement")
ax3.legend()
ax3.set_xticks(x)

plt.tight_layout()
plt.savefig("rlm_forge_results.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"\nOverall test pass rate:")
print(f"  Baseline: {np.mean(b_pass_rates):.1%}")
print(f"  Trained:  {np.mean(p_pass_rates):.1%}")


# Save the trained LoRA adapter
model.save_pretrained("./rlm_forge_lora_adapter")
tokenizer.save_pretrained("./rlm_forge_lora_adapter")

# Save training log
training_log = {
    "model_id": MODEL_ID,
    "num_prompts": NUM_TRAINING_PROMPTS,
    "episodes_per_prompt": NUM_EPISODES_PER_PROMPT,
    "max_steps_per_episode": MAX_STEPS_PER_EPISODE,
    "grpo_epochs": GRPO_EPOCHS,
    "baseline_mean_reward": float(np.mean(baseline_rewards)),
    "baseline_max_reward": float(max(baseline_rewards)),
    "trained_mean_reward": float(np.mean(post_rewards)),
    "trained_max_reward": float(max(post_rewards)),
    "improvement": float(np.mean(post_rewards) - np.mean(baseline_rewards)),
    "baseline_test_pass_rate": float(np.mean(b_pass_rates)),
    "trained_test_pass_rate": float(np.mean(p_pass_rates)),
}

with open("training_log.json", "w") as f:
    json.dump(training_log, f, indent=2)

print("Saved LoRA adapter to ./rlm_forge_lora_adapter")
print("Saved training log to training_log.json")
print(f"\nFinal summary:")
print(json.dumps(training_log, indent=2))