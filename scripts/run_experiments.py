"""
Script to run all experiments and comparisons from the paper
"""

import argparse
import sys
from pathlib import Path
import subprocess
import json
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))


def run_command(cmd: list, description: str):
    """Run shell command and handle errors"""
    print(f"\n{'='*70}")
    print(f"Running: {description}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print(f"\n✗ Failed: {description}")
        return False
    else:
        print(f"\n✓ Completed: {description}")
        return True


def run_pasto_experiment(gpu: int = 0):
    """Run PASTO model training"""
    cmd = [
        'python', 'train.py',
        '--config', 'configs/pasto_oulad.yaml',
        '--gpu', str(gpu)
    ]
    
    return run_command(cmd, "PASTO Training")


def run_baseline_experiments(gpu: int = 0):
    """Run all baseline experiments"""
    
    baselines = [
        ('XGBoost', 'baselines/train_xgboost.py'),
        ('Random Forest', 'baselines/train_random_forest.py'),
        ('DQN', 'baselines/train_dqn.py'),
        ('Whittle RMAB', 'baselines/train_whittle_rmab.py')
    ]
    
    results = {}
    
    for name, script in baselines:
        cmd = ['python', script, '--gpu', str(gpu)]
        success = run_command(cmd, f"{name} Baseline")
        results[name] = success
    
    return results


def compare_results():
    """Compare results across all models"""
    print(f"\n{'='*70}")
    print("COMPARING RESULTS")
    print(f"{'='*70}\n")
    
    results_dir = Path('results')
    
    # Collect metrics from all models
    all_metrics = {}
    
    models = ['pasto', 'xgboost', 'random_forest', 'dqn', 'whittle_rmab']
    
    for model in models:
        metrics_file = results_dir / model / 'test_metrics.json'
        
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
                all_metrics[model] = metrics
            print(f"✓ Loaded metrics for {model}")
        else:
            print(f"✗ Metrics not found for {model}")
    
    # Create comparison table
    if all_metrics:
        print(f"\n{'='*70}")
        print("RESULTS COMPARISON TABLE")
        print(f"{'='*70}\n")
        
        # Key metrics to compare
        key_metrics = [
            'auc_pr', 'recall', 'f2', 'ks_statistic',
            'retention_rate', 'mean_reward',
            'ier', 'earg'
        ]
        
        # Create dataframe
        comparison_data = {}
        for model, metrics in all_metrics.items():
            comparison_data[model] = [
                metrics.get(m, 0) for m in key_metrics
            ]
        
        df = pd.DataFrame(
            comparison_data,
            index=key_metrics
        ).T
        
        print(df.to_string())
        
        # Save comparison
        comparison_path = results_dir / 'comparison.csv'
        df.to_csv(comparison_path)
        print(f"\n✓ Comparison saved to {comparison_path}")
        
        # Highlight best results
        print(f"\n{'='*70}")
        print("BEST RESULTS PER METRIC")
        print(f"{'='*70}\n")
        
        for metric in key_metrics:
            best_model = df[metric].idxmax()
            best_value = df[metric].max()
            print(f"{metric:20s}: {best_model:15s} ({best_value:.4f})")


def generate_paper_figures():
    """Generate all figures from the paper"""
    print(f"\n{'='*70}")
    print("GENERATING PAPER FIGURES")
    print(f"{'='*70}\n")
    
    figures_script = Path('scripts/generate_figures.py')
    
    if figures_script.exists():
        cmd = ['python', str(figures_script)]
        run_command(cmd, "Figure Generation")
    else:
        print("✗ Figure generation script not found")


def run_ablation_studies(gpu: int = 0):
    """Run ablation studies"""
    print(f"\n{'='*70}")
    print("RUNNING ABLATION STUDIES")
    print(f"{'='*70}\n")
    
    ablations = [
        'No Trajectory Learning',
        'No Incentive Design',
        'Fixed Predictor',
        'Greedy Policy'
    ]
    
    for ablation in ablations:
        config_name = ablation.lower().replace(' ', '_')
        config_path = f'configs/ablations/{config_name}.yaml'
        
        if Path(config_path).exists():
            cmd = [
                'python', 'train.py',
                '--config', config_path,
                '--gpu', str(gpu)
            ]
            run_command(cmd, f"Ablation: {ablation}")
        else:
            print(f"✗ Config not found: {config_path}")


def main(args):
    """Main experiment runner"""
    
    print("\n" + "="*70)
    print("PASTO EXPERIMENTAL SUITE")
    print("="*70)
    
    # Create results directory
    Path('results').mkdir(exist_ok=True)
    
    success_count = 0
    total_count = 0
    
    # 1. Run PASTO
    if args.run_pasto:
        total_count += 1
        if run_pasto_experiment(args.gpu):
            success_count += 1
    
    # 2. Run baselines
    if args.run_baselines:
        baseline_results = run_baseline_experiments(args.gpu)
        total_count += len(baseline_results)
        success_count += sum(baseline_results.values())
    
    # 3. Run ablations
    if args.run_ablations:
        run_ablation_studies(args.gpu)
    
    # 4. Compare results
    if args.compare:
        compare_results()
    
    # 5. Generate figures
    if args.generate_figures:
        generate_paper_figures()
    
    # Summary
    print(f"\n{'='*70}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*70}")
    print(f"Completed: {success_count}/{total_count} experiments")
    print(f"Success rate: {100*success_count/total_count if total_count > 0 else 0:.1f}%")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run PASTO experiments'
    )
    
    parser.add_argument(
        '--run_pasto',
        action='store_true',
        help='Run PASTO model'
    )
    
    parser.add_argument(
        '--run_baselines',
        action='store_true',
        help='Run all baseline models'
    )
    
    parser.add_argument(
        '--run_ablations',
        action='store_true',
        help='Run ablation studies'
    )
    
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare all results'
    )
    
    parser.add_argument(
        '--generate_figures',
        action='store_true',
        help='Generate paper figures'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run everything'
    )
    
    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='GPU device ID'
    )
    
    args = parser.parse_args()
    
    # If --all, enable everything
    if args.all:
        args.run_pasto = True
        args.run_baselines = True
        args.run_ablations = True
        args.compare = True
        args.generate_figures = True
    
    # If nothing selected, show help
    if not any([args.run_pasto, args.run_baselines, args.run_ablations, 
                args.compare, args.generate_figures]):
        parser.print_help()
        sys.exit(1)
    
    main(args)
