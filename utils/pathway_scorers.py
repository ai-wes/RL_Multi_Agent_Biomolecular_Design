from rdkit.Chem import Descriptors
import logging




"""# Utility function to normalize values between 0 and 1 with epsilon
def _normalize_value(value, min_val, max_val, epsilon=0.01):
    normalized = (value - min_val) / (max_val - min_val)
    normalized = max(epsilon, min(1.0, normalized))
    print(f"Normalizing value: {value}, min_val: {min_val}, max_val: {max_val}, normalized_value: {normalized}")
    return normalized"""




def _normalize_value(value, min_val, max_val):
    return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))

# 1. Cellular Plasticity Promotion
def score_cellular_plasticity_promotion(mol):
    mw = Descriptors.ExactMolWt(mol)
    logp = Descriptors.MolLogP(mol)
    
    mw_score = min(max((mw - 50) / (500 - 50), 0), 1)  # Changed min from 150 to 50
    logp_score = min(max((logp + 2) / (5 + 2), 0), 1)  # No change, but included for completeness
    
    return (mw_score + logp_score) / 2


# 2. Proteostasis Enhancement
def score_proteostasis_enhancement(mol):
    """
    Enhances proteostasis by optimizing hydrogen bond donors and acceptors.
    """
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    print(f"Proteostasis Enhancement - HBD: {hbd}, HBA: {hba}")
    
    # Optimal HBD: 0 - 7 (more lenient range)
    hbd_score = _normalize_value(hbd, 0, 7)
    
    # Optimal HBA: 0 - 12 (more lenient range)
    hba_score = _normalize_value(hba, 0, 12)
    
    # Average the two scores
    score = (hbd_score + hba_score) / 2
    print(f"Proteostasis Enhancement - Score: {score}")
    return score

# 3. DNA Repair Enhancement
def score_dna_repair_enhancement(mol):
    """
    Enhances DNA repair capabilities by optimizing molecular flexibility and polarity.
    """
    rotatable_bonds = Descriptors.NumRotatableBonds(mol)
    tpsa = Descriptors.TPSA(mol)
    print(f"DNA Repair Enhancement - Rotatable Bonds: {rotatable_bonds}, TPSA: {tpsa}")
    
    # Desired rotatable bonds: 2 - 18 (more lenient range)
    rb_score = _normalize_value(rotatable_bonds, 2, 18)
    
    # Desired TPSA: 20 - 180 Å² (more lenient range)
    tpsa_score = _normalize_value(tpsa, 20, 180)
    
    # Average the two scores
    score = (rb_score + tpsa_score) / 2
    print(f"DNA Repair Enhancement - Score: {score}")
    return score

# 4. Cellular Reprogramming
def score_cellular_reprogramming(mol):
    """
    Facilitates cellular reprogramming by optimizing aromaticity and molecular complexity.
    """
    aromatic_rings = Descriptors.NumAromaticRings(mol)
    tpsa = Descriptors.TPSA(mol)
    print(f"Cellular Reprogramming - Aromatic Rings: {aromatic_rings}, TPSA: {tpsa}")
    
    # Desired aromatic rings: 0 - 4 (more lenient range)
    aromatic_score = _normalize_value(aromatic_rings, 0, 4)
    
    # Desired TPSA: 30 - 180 Å² (more lenient range)
    tpsa_score = _normalize_value(tpsa, 30, 180)
    
    score = (aromatic_score + tpsa_score) / 2
    print(f"Cellular Reprogramming - Score: {score}")
    return score

# 5. Protein Aggregation Inhibition
def score_protein_aggregation_inhibition(mol):
    """
    Inhibits protein aggregation by optimizing molecular size and flexibility.
    """
    mol_weight = Descriptors.ExactMolWt(mol)
    rotatable_bonds = Descriptors.NumRotatableBonds(mol)
    print(f"Protein Aggregation Inhibition - Molecular Weight: {mol_weight}, Rotatable Bonds: {rotatable_bonds}")
    
    # Desired molecular weight: 100 - 500 g/mol (more lenient range)
    mw_score = _normalize_value(mol_weight, 100, 500)
    
    # Desired rotatable bonds: 1 - 15 (more lenient range)
    rb_score = _normalize_value(rotatable_bonds, 1, 15)
    
    score = (mw_score + rb_score) / 2
    print(f"Protein Aggregation Inhibition - Score: {score}")
    return score

# 6. Genomic Instability Prevention
def score_genomic_instability_prevention(mol):
    """
    Prevents genomic instability by optimizing molecular polarity and size.
    """
    logp = Descriptors.MolLogP(mol)
    mol_weight = Descriptors.ExactMolWt(mol)
    print(f"Genomic Instability Prevention - LogP: {logp}, Molecular Weight: {mol_weight}")
    
    # Desired LogP: 0 - 5 (more lenient range)
    logp_score = _normalize_value(logp, 0, 5)
    
    # Desired molecular weight: 200 - 550 g/mol (more lenient range)
    mw_score = _normalize_value(mol_weight, 200, 550)
    
    score = (logp_score + mw_score) / 2
    print(f"Genomic Instability Prevention - Score: {score}")
    return score

# 7. Stem Cell Niche Enhancement
def score_stem_cell_niche_enhancement(mol):
    """
    Enhances the stem cell niche by optimizing hydrogen bonding and molecular flexibility.
    """
    hba = Descriptors.NumHAcceptors(mol)
    rotatable_bonds = Descriptors.NumRotatableBonds(mol)
    print(f"Stem Cell Niche Enhancement - HBA: {hba}, Rotatable Bonds: {rotatable_bonds}")
    
    # Desired HBA: 1 - 10 (more lenient range)
    hba_score = _normalize_value(hba, 1, 10)
    
    # Desired rotatable bonds: 2 - 12 (more lenient range)
    rb_score = _normalize_value(rotatable_bonds, 2, 12)
    
    score = (hba_score + rb_score) / 2
    print(f"Stem Cell Niche Enhancement - Score: {score}")
    return score

# 8. Metabolic Flexibility Enhancement
def score_metabolic_flexibility_enhancement(mol):
    """
    Enhances metabolic flexibility by optimizing TPSA and molecular weight.
    """
    tpsa = Descriptors.TPSA(mol)
    mol_weight = Descriptors.ExactMolWt(mol)
    print(f"Metabolic Flexibility Enhancement - TPSA: {tpsa}, Molecular Weight: {mol_weight}")
    
    # Desired TPSA: 40 - 220 Å² (more lenient range)
    tpsa_score = _normalize_value(tpsa, 40, 220)
    
    # Desired molecular weight: 150 - 500 g/mol (more lenient range)
    mw_score = _normalize_value(mol_weight, 150, 500)
    
    score = (tpsa_score + mw_score) / 2
    print(f"Metabolic Flexibility Enhancement - Score: {score}")
    return score

# 9. Mitochondrial Dynamics Regulation
def score_mitochondrial_dynamics_regulation(mol):
    """
    Regulates mitochondrial dynamics by optimizing LogP and hydrogen bonding.
    """
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    print(f"Mitochondrial Dynamics Regulation - LogP: {logp}, HBD: {hbd}")
    
    # Desired LogP: 0 - 4 (more lenient range)
    logp_score = _normalize_value(logp, 0, 4)
    
    # Desired HBD: 0 - 4 (more lenient range)
    hbd_score = 1 - _normalize_value(hbd, 0, 4)
    
    score = (logp_score + hbd_score) / 2
    print(f"Mitochondrial Dynamics Regulation - Score: {score}")
    return score

# 10. Proteolysis Modulation
def score_proteolysis_modulation(mol):
    """
    Modulates proteolysis by optimizing molecular size and TPSA.
    """
    mol_weight = Descriptors.ExactMolWt(mol)
    tpsa = Descriptors.TPSA(mol)
    print(f"Proteolysis Modulation - Molecular Weight: {mol_weight}, TPSA: {tpsa}")
    
    # Desired molecular weight: 150 - 450 g/mol (more lenient range)
    mw_score = _normalize_value(mol_weight, 150, 450)
    
    # Desired TPSA: 10 - 170 Å² (more lenient range)
    tpsa_score = _normalize_value(tpsa, 10, 170)
    
    score = (mw_score + tpsa_score) / 2
    print(f"Proteolysis Modulation - Score: {score}")
    return score

# 11. Telomere Protection
def score_telomere_protection(mol):
    """
    Protects telomeres by optimizing aromaticity and molecular weight.
    """
    aromatic_rings = Descriptors.NumAromaticRings(mol)
    mol_weight = Descriptors.ExactMolWt(mol)
    print(f"Telomere Protection - Aromatic Rings: {aromatic_rings}, Molecular Weight: {mol_weight}")
    
    # Desired aromatic rings: 0 - 5 (more lenient range)
    aromatic_score = _normalize_value(aromatic_rings, 0, 5)
    
    # Desired molecular weight: 200 - 550 g/mol (more lenient range)
    mw_score = _normalize_value(mol_weight, 200, 550)
    
    score = (aromatic_score + mw_score) / 2
    print(f"Telomere Protection - Score: {score}")
    return score

# 12. NAD+ Metabolism Modulation
def score_nad_metabolism_modulation(mol):
    """
    Modulates NAD+ metabolism by optimizing LogP and hydrogen bonding.
    """
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    print(f"NAD+ Metabolism Modulation - LogP: {logp}, HBD: {hbd}")
    
    # Desired LogP: 1 - 4
    logp_score = _normalize_value(logp, 1, 4)
    
    # Desired HBD: 1 - 4
    hbd_score = _normalize_value(hbd, 1, 4)
    
    score = (logp_score + hbd_score) / 2
    print(f"NAD+ Metabolism Modulation - Score: {score}")
    return score

# 13. Stem Cell Exhaustion Prevention
def score_stem_cell_exhaustion_prevention(mol):
    """
    Prevents stem cell exhaustion by optimizing molecular flexibility and TPSA.
    """
    rotatable_bonds = Descriptors.NumRotatableBonds(mol)
    tpsa = Descriptors.TPSA(mol)
    logger = logging.getLogger(__name__)
    logger.debug(f"Stem Cell Exhaustion Prevention - Rotatable Bonds: {rotatable_bonds}, TPSA: {tpsa}")
    
    rb_score = _normalize_value(rotatable_bonds, 2, 8)
    tpsa_score = _normalize_value(tpsa, 30, 150)
    
    score = (rb_score + tpsa_score) / 2
    logger.debug(f"Stem Cell Exhaustion Prevention - Score: {score}")
    return score

# 14. Autophagy-Lysosomal Pathway Enhancement
def score_autophagy_lysosomal_pathway_enhancement(mol):
    """
    Enhances the autophagy-lysosomal pathway by optimizing molecular weight and LogP.
    """
    mol_weight = Descriptors.ExactMolWt(mol)
    logp = Descriptors.MolLogP(mol)
    logger = logging.getLogger(__name__)
    logger.debug(f"Autophagy-Lysosomal Pathway Enhancement - Molecular Weight: {mol_weight}, LogP: {logp}")
    
    mw_score = _normalize_value(mol_weight, 180, 450)
    logp_score = _normalize_value(logp, 1, 3)
    
    score = (mw_score + logp_score) / 2
    logger.debug(f"Autophagy-Lysosomal Pathway Enhancement - Score: {score}")
    return score
# 15. Lipid Metabolism Regulation
def score_lipid_metabolism_regulation(mol):
    """
    Regulates lipid metabolism by optimizing LogP and TPSA.
    """
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    print(f"Lipid Metabolism Regulation - LogP: {logp}, TPSA: {tpsa}")
    
    # Desired LogP: 2 - 5
    logp_score = _normalize_value(logp, 2, 5)
    
    # Desired TPSA: 20 - 100 Å²
    tpsa_score = _normalize_value(tpsa, 20, 100)
    
    score = (logp_score + tpsa_score) / 2
    print(f"Lipid Metabolism Regulation - Score: {score}")
    return score

# 16. Cellular Energy Metabolism Optimization
def score_cellular_energy_metabolism_optimization(mol):
    """
    Optimizes cellular energy metabolism by balancing molecular flexibility and polarity.
    """
    rotatable_bonds = Descriptors.NumRotatableBonds(mol)
    tpsa = Descriptors.TPSA(mol)
    print(f"Cellular Energy Metabolism Optimization - Rotatable Bonds: {rotatable_bonds}, TPSA: {tpsa}")
    
    # Desired rotatable bonds: 3 - 10
    rb_score = _normalize_value(rotatable_bonds, 3, 10)
    
    # Desired TPSA: 25 - 150 Å²
    tpsa_score = _normalize_value(tpsa, 25, 150)
    
    score = (rb_score + tpsa_score) / 2
    print(f"Cellular Energy Metabolism Optimization - Score: {score}")
    return score

# 17. Cellular Senescence-Associated Secretory Phenotype (SASP) Modulation
def score_cellular_senescence_associated_secretory_phenotype_saspm_modulation(mol):
    """
    Modulates SASP by optimizing hydrogen bonding and molecular size.
    """
    hbd = Descriptors.NumHDonors(mol)
    mol_weight = Descriptors.ExactMolWt(mol)
    print(f"Cellular Senescence-Associated Secretory Phenotype (SASP) Modulation - HBD: {hbd}, Molecular Weight: {mol_weight}")
    
    # Desired HBD: 1 - 5
    hbd_score = _normalize_value(hbd, 1, 5)
    
    # Desired molecular weight: 200 - 500 g/mol
    mw_score = _normalize_value(mol_weight, 200, 500)
    
    score = (hbd_score + mw_score) / 2
    print(f"Cellular Senescence-Associated Secretory Phenotype (SASP) Modulation - Score: {score}")
    return score

# 18. Epigenetic Clock Modulation
def score_epigenetic_clock_modulation(mol):
    """
    Modulates the epigenetic clock by optimizing TPSA and aromatic rings.
    """
    tpsa = Descriptors.TPSA(mol)
    aromatic_rings = Descriptors.NumAromaticRings(mol)
    print(f"Epigenetic Clock Modulation - TPSA: {tpsa}, Aromatic Rings: {aromatic_rings}")
    
    # Desired TPSA: 40 - 150 Å²
    tpsa_score = _normalize_value(tpsa, 40, 150)
    
    # Desired aromatic rings: 0 - 3
    aromatic_score = _normalize_value(aromatic_rings, 0, 3)
    
    score = (tpsa_score + aromatic_score) / 2
    print(f"Epigenetic Clock Modulation - Score: {score}")
    return score
