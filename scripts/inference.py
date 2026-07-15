"""CLI entry point for DeepSeekV4 inference.

Usage:
    python scripts/inference.py checkpoints/pretrain/ckpt_best.pt
    python scripts/inference.py checkpoints/pretrain/ckpt_best.pt --interactive
    python scripts/inference.py checkpoints/pretrain/ckpt_best.pt --input prompts.txt
    python scripts/inference.py checkpoints/pretrain/ckpt_best.pt --prompt "Once upon a time"
    python scripts/inference.py checkpoints/pretrain/ckpt_best.pt --chat --prompt "Hello"
"""

import argparse
import sys
import os

import torch

torch.set_default_dtype(torch.bfloat16)

# Ensure repo root is on sys.path so that `from src.inference import ...` works.
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from src.inference import InferenceConfig, InferenceEngine, ChatEngine


def main():
    parser = argparse.ArgumentParser(description="DeepSeekV4 Inference CLI")
    parser.add_argument(
        "checkpoint",
        nargs="?",
        default="checkpoints/pretrain/ckpt_best.pt",
        help="Path to checkpoint .pt file",
    )
    parser.add_argument("-c", "--config", default="small", help="Model config name")
    parser.add_argument("-d", "--device", default="cuda:0", help="Device")
    parser.add_argument("-b", "--max-batch-size", type=int, default=0,
                        help="Override max batch size (0 = model default)")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument(
        "--system", "--system-prompt", dest="system_prompt",
        default="You are a helpful assistant.",
        help="System prompt for chat mode",
    )
    parser.add_argument(
        "-n", "--num-samples", type=int, default=1,
        help="Number of samples per prompt (non-interactive mode)",
    )

    # Modes
    parser.add_argument("--prompt", help="Single prompt to generate from")
    parser.add_argument("--input", help="File with one prompt per line (or JSON for --chat)")
    parser.add_argument("--output", help="File to write results (only with --input)")
    parser.add_argument(
        "--interactive", action="store_true", help="Interactive REPL mode"
    )
    parser.add_argument(
        "--chat", action="store_true", help="Use ChatEngine (chat template)"
    )

    args = parser.parse_args()

    config = InferenceConfig(
        checkpoint_path=args.checkpoint,
        config_name=args.config,
        device=args.device,
        max_batch_size=args.max_batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        system_prompt=args.system_prompt,
    )

    engine_cls = ChatEngine if args.chat else InferenceEngine
    engine = engine_cls(config)
    engine.setup()

    # -- Single prompt
    if args.prompt:
        for i in range(args.num_samples):
            result = engine.generate(args.prompt)
            if args.num_samples > 1:
                print(f"--- Sample {i + 1} ---")
            print(result)
            if args.num_samples > 1:
                print()

    # -- Batch from file
    elif args.input:
        if args.chat and args.input.endswith(".json"):
            # JSON multi-turn conversations
            import json

            with open(args.input) as f:
                conversations = json.load(f)

            results = []   # list[list[str]] — one list of samples per conversation
            for conv in conversations:
                messages = conv["messages"]
                samples = [engine.chat(messages) for _ in range(args.num_samples)]
                results.append(samples)

            if args.output:
                with open(args.output, "w") as f:
                    for conv, samples in zip(conversations, results):
                        obj: dict = {"messages": conv["messages"]}
                        if args.num_samples == 1:
                            obj["messages"] = conv["messages"] + [
                                {"role": "assistant", "content": samples[0]}
                            ]
                        else:
                            obj["samples"] = samples
                        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                print(f"Wrote {len(results)} results to {args.output}")
            else:
                for conv, samples in zip(conversations, results):
                    last_user = next(
                        (m["content"] for m in reversed(conv["messages"])
                         if m["role"] == "user"), ""
                    )
                    for si, reply in enumerate(samples):
                        print("=" * 60)
                        print(f"User:    {last_user[:100]}")
                        if args.num_samples > 1:
                            print(f"Sample {si + 1}/{args.num_samples}:")
                        print(f"Assistant: {reply}")
                        print("=" * 60)
        else:
            # Plain text prompts (one per line)
            with open(args.input) as f:
                prompts = [line.rstrip("\n") for line in f if line.strip()]

            # True batch: each sample pass batches all prompts together
            all_samples = [[] for _ in prompts]  # list[list[str]]
            for si in range(args.num_samples):
                batch_results = engine.generate(prompts)
                for i, r in enumerate(batch_results):
                    all_samples[i].append(r)

            if args.output:
                with open(args.output, "w") as f:
                    for p, samples in zip(prompts, all_samples):
                        f.write(f"Prompt:  {p}\n")
                        for si, r in enumerate(samples):
                            if args.num_samples > 1:
                                f.write(f"--- Sample {si + 1} (temp={args.temperature}) ---\n")
                            f.write(f"Output:  {r}\n")
                        f.write("-" * 60 + "\n")
                print(f"Wrote {len(all_samples)} prompts x {args.num_samples} samples "
                      f"to {args.output}")
            else:
                for p, samples in zip(prompts, all_samples):
                    for si, r in enumerate(samples):
                        print("=" * 60)
                        print(f"Prompt:  {p}")
                        if args.num_samples > 1:
                            print(f"Sample {si + 1}/{args.num_samples} (temp={args.temperature}):")
                        print(f"Output:  {r}")
                        print("=" * 60)

    # -- Interactive REPL
    elif args.interactive and args.chat:
        # Multi-turn chat REPL
        history: list[dict] = [
            {"role": "system", "content": args.system_prompt}
        ]
        print(f"Chat mode. System: {args.system_prompt}")
        print("Commands: /new (reset), /history (show), quit (exit)\n")
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if user_input.lower() in ("quit", "exit"):
                break
            if user_input == "/new":
                history = [{"role": "system", "content": args.system_prompt}]
                print("[Conversation reset]\n")
                continue
            if user_input == "/history":
                for m in history:
                    role = m["role"]
                    text = m["content"][:120] + ("..." if len(m["content"]) > 120 else "")
                    print(f"  [{role}] {text}")
                print()
                continue
            if not user_input:
                continue
            history.append({"role": "user", "content": user_input})
            response = engine.chat(history)
            history.append({"role": "assistant", "content": response})
            print(f"Assistant: {response}\n")

    elif args.interactive:
        print("Interactive completion mode.  Type 'quit' to exit.\n")
        while True:
            try:
                prompt = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if prompt.lower() in ("quit", "exit"):
                break
            if not prompt:
                continue
            print(engine.generate(prompt))
            print()

    # -- Default: built-in demo prompts
    else:
        prompts = [
            "Once upon a time, there was a little dragon who",
            "The capital of France is",
            "If you multiply 123 by 456, the result is",
            "In this tutorial, we will learn how to",
            "Who are you?",
        ]
        for p in prompts:
            print("=" * 60)
            print(f"Prompt:  {p}")
            print(f"Output:  {engine.generate(p)}")
            print("=" * 60)


if __name__ == "__main__":
    main()
