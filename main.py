import argparse

from experiments import exp1_ansatz_comparison
from experiments import exp2_entanglement_study
from experiments import exp3_noise_robustness

def main():
    parser = argparse.ArgumentParser(description="Quantum Autoencoder Project CLI")
    parser.add_argument('--exp', type=int, choices=[1, 2, 3], help='Experiment number to run (1: Ansatz Comparison, 2: Entanglement Study, 3: Noise Robustness)')
    
    args = parser.parse_args()
    
    if args.exp == 1:
        exp1_ansatz_comparison.run_experiment()
    elif args.exp == 2:
        exp2_entanglement_study.run_experiment()
    elif args.exp == 3:
        exp3_noise_robustness.run_experiment()
    else:
        print("Please specify an experiment number using --exp [1, 2, 3]")
        parser.print_help()

if __name__ == "__main__":
    main()
