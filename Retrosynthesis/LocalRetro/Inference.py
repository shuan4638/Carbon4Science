"""
Simple inference interface for LocalRetro.

Usage:
    from Inference import run
    results = run("CCO")  # Returns DataFrame with predicted reactants
"""

import os
import sys

# Set default paths relative to this file's location
_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Add scripts directory to path for imports
_SCRIPTS_DIR = os.path.join(_ROOT_DIR, 'scripts')
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

_DEFAULT_ARGS = {
    'dataset': 'USPTO_50K',
    'data_dir': os.path.join(_ROOT_DIR, 'data', 'USPTO_50K'),
    'config_path': os.path.join(_ROOT_DIR, 'data', 'configs', 'default_config.json'),
    'model_path': os.path.join(_ROOT_DIR, 'models', 'LocalRetro_USPTO_50K.pth'),
    'device': 'cuda:0',
}

_model = None


def _get_model():
    """Lazy-load the model on first use."""
    global _model
    if _model is None:
        import torch
        from Retrosynthesis import LocalRetro

        args = _DEFAULT_ARGS.copy()
        if not torch.cuda.is_available():
            args['device'] = 'cpu'

        _model = LocalRetro(args)
    return _model


def run(smiles, top_k=10, verbose=False):
    """
    Predict retrosynthesis for a given product SMILES.

    Args:
        smiles: Product SMILES string
        top_k: Number of top predictions to return (default: 10)
        verbose: Print detailed prediction info (default: False)

    Returns:
        DataFrame with columns: SMILES, Predicted site, Local reaction template, Score, Molecule
    """
    model = _get_model()
    return model.retrosnythesis(smiles, top_k=top_k, verbose=verbose)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python Inference.py <SMILES> [top_k]")
        print("Example: python Inference.py 'CCO' 10")
        sys.exit(1)

    smiles = sys.argv[1]
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    results = run(smiles, top_k=top_k, verbose=True)
    print("\nResults:")
    print(results[['SMILES', 'Score']].to_string())
