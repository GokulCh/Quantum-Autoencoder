import argparse

from experiments import exp1_ansatz_comparison
from experiments import exp2_entanglement_study
from experiments import exp3_noise_robustness
from experiments import exp4_scaling
from experiments import exp5_baselines

def main():
    parser = argparse.ArgumentParser(description="Quantum Autoencoder Project CLI")
    parser.add_argument('--exp', type=int, choices=[1, 2, 3, 4, 5], help='Experiment number to run (1: Ansatz Comparison, 2: Entanglement Study, 3: Noise Robustness, 4: Scaling, 5: Baselines)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    if args.exp == 1:
        exp1_ansatz_comparison.run_experiment(seed=args.seed)
    elif args.exp == 2:
        exp2_entanglement_study.run_experiment(seed=args.seed)
    elif args.exp == 3:
        exp3_noise_robustness.run_experiment(seed=args.seed)
    elif args.exp == 4:
        exp4_scaling.run_experiment(seed=args.seed)
    elif args.exp == 5:
        exp5_baselines.run_experiment(seed=args.seed)
    else:
        print("Please specify an experiment number using --exp [1, 2, 3, 4, 5]")
        parser.print_help()

if __name__ == "__main__":
    main()
