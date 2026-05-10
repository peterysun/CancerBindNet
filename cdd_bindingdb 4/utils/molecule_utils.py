"""Molecule utilities — convert SMILES to features for the model."""

import numpy as np
from typing import Optional

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("Warning: RDKit not installed. Run: pip install rdkit")


def smiles_to_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048) -> Optional[np.ndarray]:
    """Convert SMILES to Morgan fingerprint (binary vector of length n_bits)."""
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit required")

    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    return np.array(fp, dtype=np.float32)


def smiles_to_descriptors(smiles: str) -> Optional[np.ndarray]:
    """Compute 6 normalized molecular descriptors used in drug-likeness assessment."""
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit required")

    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None

    try:
        # Normalize each value to roughly the [0, 1] range
        descriptors = np.array([
            Descriptors.MolWt(mol) / 500.0,                         # molecular weight
            (Descriptors.MolLogP(mol) + 2) / 7.0,                   # lipophilicity
            rdMolDescriptors.CalcNumHBD(mol) / 5.0,                 # H-bond donors
            rdMolDescriptors.CalcNumHBA(mol) / 10.0,                # H-bond acceptors
            rdMolDescriptors.CalcNumRotatableBonds(mol) / 10.0,     # rotatable bonds
            Descriptors.TPSA(mol) / 150.0,                          # polar surface area
        ], dtype=np.float32)

        return np.clip(descriptors, 0, 1)
    except Exception:
        return None


def smiles_to_features(smiles: str, radius: int = 2, n_bits: int = 2048,
                       include_descriptors: bool = True) -> Optional[np.ndarray]:
    """Full feature vector: fingerprint concatenated with descriptors."""
    fp = smiles_to_fingerprint(smiles, radius=radius, n_bits=n_bits)
    if fp is None:
        return None

    if include_descriptors:
        desc = smiles_to_descriptors(smiles)
        if desc is not None:
            return np.concatenate([fp, desc])

    return fp


def batch_smiles_to_features(smiles_list: list, radius: int = 2, n_bits: int = 2048,
                              include_descriptors: bool = True,
                              show_progress: bool = True) -> tuple[np.ndarray, list]:
    """Convert a list of SMILES into a feature matrix. Returns (features, valid_indices)."""
    try:
        from tqdm import tqdm
        iterator = tqdm(smiles_list, desc="Computing fingerprints") if show_progress else smiles_list
    except ImportError:
        iterator = smiles_list

    features = []
    valid_indices = []

    for i, smi in enumerate(iterator):
        feat = smiles_to_features(smi, radius=radius, n_bits=n_bits,
                                   include_descriptors=include_descriptors)
        if feat is not None:
            features.append(feat)
            valid_indices.append(i)

    if not features:
        return np.zeros((0, n_bits + (6 if include_descriptors else 0))), []

    return np.array(features, dtype=np.float32), valid_indices


def passes_lipinski(smiles: str) -> bool:
    """
    Lipinski's Rule of 5 — a basic drug-likeness filter.
    Returns True if the molecule violates at most one rule.
    """
    if not RDKIT_AVAILABLE:
        return True

    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return False

    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)

    violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
    return violations <= 1


if __name__ == '__main__':
    if not RDKIT_AVAILABLE:
        print("Install RDKit: pip install rdkit")
        exit()

    # Test with Gefitinib (approved EGFR inhibitor)
    gefitinib = 'COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1'

    fp = smiles_to_fingerprint(gefitinib)
    print(f"Fingerprint shape: {fp.shape}, set bits: {fp.sum():.0f}")

    desc = smiles_to_descriptors(gefitinib)
    print(f"Descriptors: {desc}")

    full = smiles_to_features(gefitinib)
    print(f"Full feature shape: {full.shape}")
    print(f"Lipinski compliant: {passes_lipinski(gefitinib)}")
