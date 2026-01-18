"""
Evaluation Script for PASTO
Usage: python evaluate.py --checkpoint checkpoints/pasto_best.pth --dataset oulad
"""

import argparse
import torch
import numpy as np
import yaml
import json
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))

from src.models.pasto import PASTO, PASTOSimulator
from src.data.dataset_loader import OULADDatasetLoader, create_dataloaders
from src.data.preprocess import preprocess_pipeline, FeatureEngineer
from src.evaluation.metrics import (
    compute_all_metrics,
    print_metrics_table,
    compute_cumulative_regret
)


def load_checkpoint(checkpoint_path: str, device: str = 'cuda'):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    config = checkpoint['config']
    model = PASTO(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, config, checkpoint.get('epoch', 0)


def evaluate_model(
    model: PASTO,
    test_loader: torch.utils.data.DataLoader,
    config: dict,
    device: str = 'cuda'
):
    """
    Comprehensive model evaluation
    
    Args:
        model: PASTO model
        test_loader: Test data loader
        config: Configuration dict
        device: Device to evaluate on
        
    Returns:
        Dictionary of results
    """
    model.eval()
    
    # Collect all predictions
    all_dropout_probs = []
    all_dropout_labels = []
    all_trajectory_preds = []
    all_trajectory_labels = []
    all_actions = []
    all_rewards = []
    all_risk_scores = []
    all_engagement_scores = []
    all_priority_scores = []
    all_student_ids = []
    
    print("\n" + "="*70)
    print("EVALUATING MODEL")
    print("="*70)
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Processing batches"):
            sequences = batch['sequence'].to(device)
            dropout_labels = batch['dropout_label'].to(device)
            trajectory_labels = batch['trajectory_label'].to(device)
            student_ids = batch['student_id'].numpy()
            
            # Forward pass
            outputs = model(sequences)
            
            # Sample actions
            actions, priorities, _ = model.sample_actions(
                sequences,
                budget_ratio=config['policy']['budget_ratio'],
                deterministic=True
            )
            
            # Compute rewards
            rewards = model.compute_reward(
                dropout_labels,
                trajectory_labels,
                outputs['trajectory_pred'],
                actions
            )
            
            # Collect results
            all_dropout_probs.append(outputs['dropout_probs'].cpu().numpy())
            all_dropout_labels.append(dropout_labels.cpu().numpy())
            all_trajectory_preds.append(outputs['trajectory_pred'].cpu().numpy())
            all_trajectory_labels.append(trajectory_labels.cpu().numpy())
            all_actions.append(actions.cpu().numpy())
            all_rewards.append(rewards.cpu().numpy())
            all_risk_scores.append(outputs['risk_scores'].cpu().numpy())
            all_engagement_scores.append(outputs['engagement_scores'].cpu().numpy())
            all_priority_scores.append(priorities.cpu().numpy())
            all_student_ids.append(student_ids)
    
    # Concatenate all results
    results = {
        'dropout_probs': np.concatenate(all_dropout_probs),
        'dropout_labels': np.concatenate(all_dropout_labels),
        'trajectory_preds': np.concatenate(all_trajectory_preds),
        'trajectory_labels': np.concatenate(all_trajectory_labels),
        'actions': np.concatenate(all_actions),
        'rewards': np.concatenate(all_rewards),
        'risk_scores': np.concatenate(all_risk_scores),
        'engagement_scores': np.concatenate(all_engagement_scores),
        'priority_scores': np.concatenate(all_priority_scores),
        'student_ids': np.concatenate(all_student_ids)
    }
    
    return results


def run_policy_simulation(
    model: PASTO,
    test_loader: torch.utils.data.DataLoader,
    config: dict,
    device: str = 'cuda',
    num_runs: int = 5
):
    """
    Run policy simulation over multiple episodes
    
    Args:
        model: PASTO model
        test_loader: Test data loader
        config: Configuration
        device: Device
        num_runs: Number of simulation runs
        
    Returns:
        Simulation results
    """
    print("\n" + "="*70)
    print("RUNNING POLICY SIMULATION")
    print("="*70)
    
    simulator = PASTOSimulator(model, config, device)
    
    all_rewards = []
    all_retention_rates = []
    all_cumulative_rewards = []
    
    for run in range(num_runs):
        print(f"\nRun {run + 1}/{num_runs}")
        
        # Get a batch
        batch = next(iter(test_loader))
        sequences = batch['sequence'].to(device)
        dropout_labels = batch['dropout_label'].to(device)
        trajectory_labels = batch['trajectory_label'].to(device)
        
        # Run simulation
        sim_results = simulator.simulate_episode(
            sequences,
            dropout_labels,
            trajectory_labels,
            budget_ratio=config['policy']['budget_ratio']
        )
        
        all_rewards.append(sim_results['rewards'])
        all_retention_rates.append(sim_results['retention_rate'])
        all_cumulative_rewards.append(sim_results['cumulative_reward'])
    
    # Average results
    avg_rewards = np.mean([r[-1] for r in all_cumulative_rewards])
    avg_retention = np.mean([r[-1] for r in all_retention_rates])
    
    print(f"\nSimulation Results (averaged over {num_runs} runs):")
    print(f"  Average Cumulative Reward: {avg_rewards:.4f}")
    print(f"  Average Final Retention: {avg_retention:.4f}")
    
    return {
        'rewards': all_rewards,
        'retention_rates': all_retention_rates,
        'cumulative_rewards': all_cumulative_rewards
    }


def visualize_results(
    results: dict,
    save_dir: Path
):
    """
    Create visualizations of evaluation results
    
    Args:
        results: Evaluation results
        save_dir: Directory to save figures
    """
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (10, 6)
    
    # 1. Precision-Recall Curve
    from sklearn.metrics import precision_recall_curve
    
    precision, recall, _ = precision_recall_curve(
        results['dropout_labels'],
        results['dropout_probs']
    )
    
    plt.figure()
    plt.plot(recall, precision, linewidth=2)
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig(save_dir / 'precision_recall_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved precision-recall curve")
    
    # 2. Risk vs Engagement Scatter
    plt.figure()
    scatter = plt.scatter(
        results['engagement_scores'],
        results['risk_scores'],
        c=results['dropout_labels'],
        cmap='RdYlGn_r',
        alpha=0.6,
        s=20
    )
    plt.colorbar(scatter, label='Dropout')
    plt.xlabel('Engagement Score', fontsize=12)
    plt.ylabel('Risk Score', fontsize=12)
    plt.title('Risk vs Engagement Distribution', fontsize=14)
    plt.savefig(save_dir / 'risk_engagement_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved risk-engagement scatter plot")
    
    # 3. Action Distribution
    plt.figure()
    action_counts = [
        (1 - results['actions']).sum(),
        results['actions'].sum()
    ]
    plt.bar(['No Intervention', 'Intervention'], action_counts, color=['skyblue', 'coral'])
    plt.ylabel('Count', fontsize=12)
    plt.title('Intervention Allocation', fontsize=14)
    plt.savefig(save_dir / 'action_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved action distribution")
    
    # 4. Reward Distribution
    plt.figure()
    plt.hist(results['rewards'], bins=50, alpha=0.7, color='green', edgecolor='black')
    plt.axvline(results['rewards'].mean(), color='red', linestyle='--', 
                label=f'Mean: {results["rewards"].mean():.3f}')
    plt.xlabel('Reward', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Reward Distribution', fontsize=14)
    plt.legend()
    plt.savefig(save_dir / 'reward_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved reward distribution")
    
    # 5. Priority Score Distribution by Outcome
    plt.figure()
    dropout_priorities = results['priority_scores'][results['dropout_labels'] == 1]
    retain_priorities = results['priority_scores'][results['dropout_labels'] == 0]
    
    plt.hist(retain_priorities, bins=30, alpha=0.5, label='Retained', color='green')
    plt.hist(dropout_priorities, bins=30, alpha=0.5, label='Dropout', color='red')
    plt.xlabel('Priority Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Priority Score Distribution by Outcome', fontsize=14)
    plt.legend()
    plt.savefig(save_dir / 'priority_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved priority distribution")
    
    print(f"\n✓ All visualizations saved to: {save_dir}")


def main(args):
    """Main evaluation function"""
    
    # Set device
    device = f'cuda:{args.gpu}' if args.gpu >= 0 and torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    model, config, epoch = load_checkpoint(args.checkpoint, device)
    print(f"✓ Loaded model from epoch {epoch}")
    
    # Prepare data
    from train import prepare_data
    _, _, test_loader, _ = prepare_data(config)
    
    # Evaluate model
    results = evaluate_model(model, test_loader, config, device)
    
    # Compute metrics
    metrics = compute_all_metrics(
        results['dropout_probs'],
        results['dropout_labels'],
        results['trajectory_preds'],
        results['trajectory_labels'],
        results['actions'],
        results['rewards']
    )
    
    # Print metrics
    print_metrics_table(metrics)
    
    # Save metrics
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_path = output_dir / 'evaluation_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✓ Metrics saved to: {metrics_path}")
    
    # Save detailed results
    results_path = output_dir / 'evaluation_results.npz'
    np.savez(results_path, **results)
    print(f"✓ Detailed results saved to: {results_path}")
    
    # Run policy simulation
    if args.run_simulation:
        sim_results = run_policy_simulation(
            model, test_loader, config, device, num_runs=args.num_sim_runs
        )
    
    # Generate visualizations
    if args.visualize:
        visualize_results(results, output_dir / 'figures')
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate PASTO model')
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default='oulad',
        help='Dataset name'
    )
    
    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='GPU device ID (-1 for CPU)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results/evaluation',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate visualization plots'
    )
    
    parser.add_argument(
        '--run_simulation',
        action='store_true',
        help='Run policy simulation'
    )
    
    parser.add_argument(
        '--num_sim_runs',
        type=int,
        default=5,
        help='Number of simulation runs'
    )
    
    args = parser.parse_args()
    
    main(args)
