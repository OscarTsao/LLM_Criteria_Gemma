"""
NLI Binary Classification Inference with Terminal Visualization

Interactive prediction script for NLI criteria matching with beautiful terminal output.
"""

import torch
import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from transformers import AutoTokenizer
from models.gemma_encoder import GemmaClassifier
from data.dsm5_criteria import DSM5_CRITERIA, get_criterion_text
from utils.terminal_viz import InferenceVisualizer, console, print_model_info


def load_model(checkpoint_path: str, device: str = 'cuda'):
    """Load trained NLI model from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get config from checkpoint
    config = checkpoint.get('config', {})
    model_config = config.get('model', {})

    # Create model
    model = GemmaClassifier(
        num_classes=2,  # Binary classification
        model_name=model_config.get('name', 'google/gemma-2-2b'),
        pooling_strategy=model_config.get('pooling_strategy', 'mean'),
        freeze_encoder=model_config.get('freeze_encoder', True),
        hidden_dropout_prob=model_config.get('hidden_dropout_prob', 0.1),
        classifier_hidden_size=model_config.get('classifier_hidden_size', None),
        use_gradient_checkpointing=False,  # Not needed for inference
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"✓ Model loaded successfully")
    print(f"  Best Val F1: {checkpoint.get('best_val_f1', 'N/A')}")
    print(f"  Fold: {checkpoint.get('fold', 'N/A')}")

    return model, config


@torch.no_grad()
def predict_single(model, tokenizer, post: str, criterion: str, device: str = 'cuda'):
    """Predict whether post matches criterion."""
    # Tokenize text pair
    encoding = tokenizer(
        post,
        criterion,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # Get prediction
    logits = model(input_ids, attention_mask)
    probs = torch.softmax(logits, dim=-1)

    prediction = torch.argmax(logits, dim=-1).item()
    probability = probs[0, prediction].item()

    return {
        'prediction': prediction,
        'probability': probability,
        'probs': probs[0].cpu().numpy(),
    }


def interactive_mode(model, tokenizer, device: str = 'cuda'):
    """Interactive prediction mode."""
    viz = InferenceVisualizer()

    if console:
        console.print("\n[bold cyan]═══════════════════════════════════════[/bold cyan]")
        console.print("[bold cyan]  Interactive NLI Prediction Mode[/bold cyan]")
        console.print("[bold cyan]═══════════════════════════════════════[/bold cyan]\n")
        console.print("[yellow]Enter 'quit' or 'exit' to stop[/yellow]\n")
    else:
        print("\n" + "=" * 60)
        print("Interactive NLI Prediction Mode")
        print("=" * 60)
        print("Enter 'quit' or 'exit' to stop\n")

    while True:
        # Get post
        if console:
            console.print("[bold]Enter Reddit post:[/bold]", style="cyan")
        else:
            print("\nEnter Reddit post:")

        post = input("> ").strip()

        if post.lower() in ['quit', 'exit', 'q']:
            break

        if not post:
            print("Please enter a post.")
            continue

        # Select criterion
        if console:
            console.print("\n[bold]Available DSM-5 Criteria:[/bold]", style="cyan")
        else:
            print("\nAvailable DSM-5 Criteria:")

        symptoms = list(DSM5_CRITERIA.keys())
        for i, symptom in enumerate(symptoms, 1):
            print(f"  {i}. {symptom}")

        if console:
            console.print("\n[bold]Select criterion number (or enter custom text):[/bold]", style="cyan")
        else:
            print("\nSelect criterion number (or enter custom text):")

        criterion_input = input("> ").strip()

        # Parse criterion input
        if criterion_input.isdigit():
            idx = int(criterion_input) - 1
            if 0 <= idx < len(symptoms):
                criterion = get_criterion_text(symptoms[idx], use_short=False)
            else:
                print("Invalid number. Using custom text.")
                criterion = criterion_input
        else:
            criterion = criterion_input

        if not criterion:
            print("Please enter a criterion.")
            continue

        # Make prediction
        result = predict_single(model, tokenizer, post, criterion, device)

        # Display result
        viz.print_prediction(
            post=post,
            criterion=criterion,
            prediction=result['prediction'],
            probability=result['probability']
        )


def batch_predict_mode(model, tokenizer, post_file: str, criterion: str, device: str = 'cuda'):
    """Batch prediction mode from file."""
    viz = InferenceVisualizer()

    # Load posts
    posts_path = Path(post_file)
    if not posts_path.exists():
        print(f"Error: File not found: {post_file}")
        return

    with open(posts_path, 'r') as f:
        posts = [line.strip() for line in f if line.strip()]

    print(f"\nLoaded {len(posts)} posts from {post_file}")

    # Get criterion text
    if criterion in DSM5_CRITERIA:
        criterion_text = get_criterion_text(criterion, use_short=False)
    else:
        criterion_text = criterion

    print(f"Criterion: {criterion_text[:100]}...\n")

    # Make predictions
    predictions = []

    for i, post in enumerate(posts):
        result = predict_single(model, tokenizer, post, criterion_text, device)
        predictions.append({
            'post': post,
            'prediction': result['prediction'],
            'probability': result['probability'],
        })

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(posts)} posts...")

    # Display results
    viz.print_batch_predictions(predictions, show_details=True)


def demo_mode(model, tokenizer, device: str = 'cuda'):
    """Demo mode with example predictions."""
    viz = InferenceVisualizer()

    if console:
        console.print("\n[bold cyan]═══════════════════════════════════════[/bold cyan]")
        console.print("[bold cyan]  Demo: Example Predictions[/bold cyan]")
        console.print("[bold cyan]═══════════════════════════════════════[/bold cyan]\n")
    else:
        print("\n" + "=" * 60)
        print("Demo: Example Predictions")
        print("=" * 60 + "\n")

    # Example 1: Should match DEPRESSED_MOOD
    examples = [
        {
            'post': "I've been feeling really down lately. Everything feels hopeless and I can't seem to find joy in anything anymore. I just want to stay in bed all day.",
            'criterion': 'DEPRESSED_MOOD',
            'expected': 1,
        },
        {
            'post': "I used to love playing video games, but now I don't even want to turn on my console. Nothing seems fun anymore.",
            'criterion': 'ANHEDONIA',
            'expected': 1,
        },
        {
            'post': "I've been sleeping 12 hours a day and still feel exhausted. Can't seem to get out of bed in the morning.",
            'criterion': 'SLEEP_ISSUES',
            'expected': 1,
        },
        {
            'post': "Today was a great day! Went hiking and had an amazing time with friends.",
            'criterion': 'DEPRESSED_MOOD',
            'expected': 0,
        },
        {
            'post': "I'm so tired all the time. No energy to do anything, even simple tasks feel overwhelming.",
            'criterion': 'FATIGUE',
            'expected': 1,
        },
    ]

    predictions = []

    for i, example in enumerate(examples, 1):
        criterion_text = get_criterion_text(example['criterion'], use_short=False)

        result = predict_single(model, tokenizer, example['post'], criterion_text, device)

        predictions.append({
            'post': example['post'],
            'criterion': example['criterion'],
            'prediction': result['prediction'],
            'probability': result['probability'],
            'true_label': example['expected'],
            'correct': result['prediction'] == example['expected'],
        })

        if console:
            console.print(f"\n[bold]Example {i}/{len(examples)}[/bold]")

        viz.print_prediction(
            post=example['post'],
            criterion=criterion_text,
            prediction=result['prediction'],
            probability=result['probability'],
            true_label=example['expected']
        )

        input("\nPress Enter for next example...")

    # Summary
    correct = sum(1 for p in predictions if p['correct'])
    accuracy = correct / len(predictions)

    if console:
        console.print(f"\n[bold green]Demo Complete![/bold green]")
        console.print(f"Accuracy: {correct}/{len(predictions)} ({accuracy:.1%})")
    else:
        print(f"\nDemo Complete!")
        print(f"Accuracy: {correct}/{len(predictions)} ({accuracy:.1%})")


def main():
    parser = argparse.ArgumentParser(description='NLI Binary Classification Inference')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--mode', type=str, default='interactive',
                       choices=['interactive', 'batch', 'demo'],
                       help='Inference mode')
    parser.add_argument('--post_file', type=str, help='File with posts (for batch mode)')
    parser.add_argument('--criterion', type=str, help='Criterion for batch mode')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')

    args = parser.parse_args()

    # Load model
    device = args.device if torch.cuda.is_available() else 'cpu'
    model, config = load_model(args.checkpoint, device)

    # Load tokenizer
    model_name = config.get('model', {}).get('name', 'google/gemma-2-2b')
    print(f"\nLoading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Run mode
    if args.mode == 'interactive':
        interactive_mode(model, tokenizer, device)
    elif args.mode == 'batch':
        if not args.post_file or not args.criterion:
            print("Error: --post_file and --criterion required for batch mode")
            return
        batch_predict_mode(model, tokenizer, args.post_file, args.criterion, device)
    elif args.mode == 'demo':
        demo_mode(model, tokenizer, device)

    print("\n✓ Inference complete!")


if __name__ == '__main__':
    main()
