
from transform_and_path import MULTI_TARGET_THRESHOLD, ALL_FRAGMENTS, FRAGMENT_COMPLEXITY, PATHWAY_SCORING_FUNCTIONS, SCALING_FACTORS
from utils.scorers import predict_docking_score
from utils.pathway_scorers import *
from utils.scorers import *
from rdkit import Chem
from rdkit.Chem import Descriptors, QED
import numpy as np
import random
import logging
from utils.sascorer import calculateScore
from evaluation.evaluator import is_valid_molecule
from rdkit.Chem import GetPeriodicTable

logger = logging.getLogger(__name__)


class MoleculeEnv:
    def __init__(self, max_steps=10, max_atoms=20, curriculum_level=1, 
                 pathway_scoring_functions=None, user_objectives=None, 
                 agent_objectives=None, known_molecules_smiles=None, agent=None):
        self.all_fragments = ALL_FRAGMENTS
        self.fragment_complexity = FRAGMENT_COMPLEXITY
        self.max_atoms = max_atoms
        self.action_space_size = 1 + (len(self.all_fragments) * max_atoms)  # Include 'terminate' action
        self.max_steps = max_steps
        self.curriculum_level = curriculum_level
        self.pathway_scoring_functions = pathway_scoring_functions if pathway_scoring_functions else PATHWAY_SCORING_FUNCTIONS
        self.current_pathway = None
        self.pathway_target_score = 1.0  # Perfect score
        self.pathway_weight = 10.0  # High weight for pathway optimization
        self.current_pathway_index = 0
        self.pathways = list(self.pathway_scoring_functions.keys())
        self.known_molecules_smiles = known_molecules_smiles or []  # Store this as an attribute
        self.known_molecule_fps = self._load_known_molecule_fingerprints(self.known_molecules_smiles)
        # Define initial SMILES per curriculum level
        self.initial_smiles_per_level = {
            1: [
                'C1=CC=CC=C1',        # Benzene
                'CCO',                # Ethanol
                'CCN',                # Ethylamine
                'CCC',                # Propane
                'CC(=O)O',            # Acetic acid
                'CC(=O)N',            # Acetamide
                'CC#N',               # Acetonitrile
                'C1CCCCC1',           # Cyclohexane
                'C1=CC=CC=C1O',       # Phenol
                'CC(C)O',             # Isopropanol
            ],
            2: [
                "CC1=CC=C(C=C1)NC(=O)C",
                "COC1=C(C=C(C=C1)NC(=O)C)OC",
                "CC(=O)NC1=CC=C(C=C1)O",
                "CC1=C(C(=CC=C1)C)NC(=O)CC2=CC=CC=C2",
                "COC1=CC(=CC(=C1OC)OC)C(=O)NC2=CC=CC=C2",
                "CC1=CC=C(C=C1)NC(=O)CCCN2CCC(CC2)N3C(=O)NC4=CC=CC=C43",
                "COC1=C(C=C(C=C1NC(=O)C2=C1C=CC(=C2)C#N)NC3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4",
                "CC1=C(C=O)NC1C2=CC=CC=C2",
                "CC1=C(C=CC=C1)S(=O)(=O)NC2=CC=CC=C2",
                "CC1=CC(=NO1)C2=CC(=CC=C2)S(=O)(=O)NC3=CC=CC=C3",
                "CC1=C(C=O)C2=C(C1=O)C(=CC=C2)O",
                "CC1=CC=C(C=C1)OCC(C)(C)NC2=NC=NC3=C2C=CN3",
                "COC1=C(C=C(C=C1)CNC(=O)C2=CC=C(C=C2)CN3CCOCC3)OC",
                "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)C",
                "COC1=C(C=C(C=C1)CNC(=O)C2=CC=C(C=C2)S(=O)(=O)C(C)C)OC",
                "CC(=O)Nc1ccc(cc1)C(=O)O",
                "C1=CC=CC=C1C(=O)O",
                "CCN(CC)C(=O)C1=CC=CC=C1",
                "CC(=O)Nc1ccc(cc1)N",
                "CCOCC(=O)Nc1ccc(cc1)O",
            ],
            2.5: [
                "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)C",
                "COC1=C(C=C(C=C1)CNC(=O)C2=CC=C(C=C2)S(=O)(=O)C(C)C)OC",
                "CC1=C(C=CC(=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)C",
                "COC1=C(C=C(C=C1)CNC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)OC",
            ],
            3: [
                'C1=CC=C(C=C1)C(=O)CCCCCCCCCCCCCCCCCC',  # nonadecanophenone
                'C1=CC=C(C=C1)C(=O)CCCCCCCCCCCCCCCCCCC',  # eicosanophenone
            ]
        }
        self.reward_stats = {key: {'mean': 0.0, 'std': 1.0} for key in [
            'docking', 'pathway', 'multi_target', 'qed', 'weight', 'logp', 'sas', 'lipinski', 'diversity', 'step',
            'docking_score', 'molecular_weight', 'num_rotatable_bonds', 'num_h_acceptors', 'num_h_donors', 'tpsa',
            'similarity_to_known_active', 'synthetic_accessibility'
        ]}
        self.agent = agent
        self.current_step = 0
        self.diverse_mols = []
        self.difficulty = 1
        self.cumulative_performance = 0  # For cumulative reward tracking
        self.smiles_sequence = []
        self.user_objectives = user_objectives if user_objectives else {}
        self.agent_objectives = agent_objectives if agent_objectives else []
        self.objectives = self.agent_objectives  # Add this line
        self.current_mol = None
        self.current_episode = 0
        self.state_mol_cache = {}
        self.generated_mols = []  # Add this line to initialize the generated_mols list
        self.diversity_score = 0.0  # Initialize diversity score
        self.multitarget_score = 0.0  # Initialize multitarget score
    
    
    def set_agent(self, agent):
        self.agent = agent

    def set_curriculum_level(self, level):
        self.curriculum_level = level
        predefined_levels = [1, 2, 2.5, 3]  # Define all curriculum levels here

        if level in predefined_levels:
            if level == 1:
                self.max_steps = 10
                self.max_atoms = 20
                logger.info("Curriculum Level 1: Basic molecules.")
            elif level == 2:
                self.max_steps = 15
                self.max_atoms = 25
                logger.info("Curriculum Level 2: Intermediate molecules.")
            elif level == 2.5:
                self.max_steps = 12
                self.max_atoms = 23
                logger.info("Curriculum Level 2.5: Advanced molecules.")
            elif level == 3:
                self.max_steps = 20
                self.max_atoms = 30
                logger.info("Curriculum Level 3: Expert molecules.")
        else:
            logger.warning(f"Curriculum level {level} not recognized. No changes made.")

    def reset(self):
        try:
            initial_smiles = self.initial_smiles_per_level.get(self.curriculum_level, self.initial_smiles_per_level[1])
            self.current_mol = Chem.MolFromSmiles(random.choice(initial_smiles))
            if self.current_mol is None:
                logger.warning("Failed to initialize molecule in reset()")
                self.current_mol = Chem.MolFromSmiles('C')  # Default to methane if initialization fails
            self.current_smiles = Chem.MolToSmiles(self.current_mol)
            self.steps_taken = 0
            self.smiles_sequence = [self.current_smiles]
            self.generated_mols = [self.current_mol]  # Reset generated_mols list

            self.diversity_score = 0.0  # Reset diversity score
            self.multitarget_score = 0.0  # Reset multitarget score
            self.qed_score = 0.0  # Reset QED score
            self.generated_mols = [self.current_mol]  # Reset generated_mols list
            return self.get_state()
        except Exception as e:
            logger.error(f"Error in reset(): {e}")
            self.current_mol = Chem.MolFromSmiles('C')  # Default to methane if any error occurs
            self.current_smiles = 'C'
            self.steps_taken = 0
            self.smiles_sequence = [self.current_smiles]
            self.generated_mols = [self.current_mol]  # Reset generated_mols list
            self.diversity_score = 0.0  # Reset diversity score
            self.multitarget_score = 0.0  # Reset multitarget score
            self.qed_score = 0.0  # Reset QED score
            return self.get_state()
        
        
    def get_valid_actions(self, state):
        valid_actions = []
        for action in range(self.action_space_size):
            if action == 0:  # Terminate action
                valid_actions.append((0, 0))
            else:
                fragment_index = (action - 1) // self.max_atoms
                atom_index = (action - 1) % self.max_atoms
                if self.is_valid_action(atom_index, fragment_index):
                    valid_actions.append((atom_index, fragment_index))
        return valid_actions


    def step(self, action):
        self.current_step += 1
        done = False
        info = {}

        if isinstance(action, tuple):
            atom_index, fragment_index = action
        else:
            if action == 0:  # Terminate action
                done = True
            else:
                fragment_index = (action - 1) // self.max_atoms
                atom_index = (action - 1) % self.max_atoms

        if not done:
            if atom_index < self.current_mol.GetNumAtoms() and fragment_index < len(self.all_fragments):
                try:
                    new_mol = Chem.RWMol(self.current_mol)
                    fragment_smiles = self.all_fragments[fragment_index]
                    fragment_mol = Chem.MolFromSmiles(fragment_smiles)
                    if fragment_mol is None:
                        return self.get_state(), -1, False, {}  # Return negative reward for invalid action
                    new_mol.ReplaceAtom(atom_index, fragment_mol.GetAtomWithIdx(0))
                    new_mol = new_mol.GetMol()
                    
                    if Chem.SanitizeMol(new_mol, catchErrors=True) == 0:
                        self.current_mol = new_mol
                        self.smiles_sequence.append(Chem.MolToSmiles(self.current_mol))
                        self.generated_mols.append(self.current_mol)
                    else:
                        return self.get_state(), -1, False, {}  # Return negative reward for invalid molecule
                except Exception as e:
                    logger.error(f"Error in molecule modification: {e}")
                    return self.get_state(), -1, False, {}  # Return negative reward for any error
            else:
                return self.get_state(), -1, False, {}  # Return negative reward for invalid action

        self._calculate_diversity()  # Update diversity score after each step
        self.calculate_multitarget_score(self.current_mol)  # Update multitarget score after each step
        self.calculate_qed_score()
        # Compute reward using the existing _compute_reward method
        reward = self._compute_reward(self.current_mol)
        complexity_bonus = self.calculate_complexity_bonus(action)
        reward += complexity_bonus  # Only once
        self.cumulative_performance += reward
        

        # Check if max steps reached or other termination conditions
        if self.current_step >= self.max_steps or self._objectives_met(self.current_mol):
            done = True
        if not is_valid_molecule(self.current_mol):
            done = True
        # Update curriculum level if needed
        if done:
            self._update_curriculum()

        # Compute additional info
        info = {
            'smiles': Chem.MolToSmiles(self.current_mol),
            'mol_weight': Descriptors.ExactMolWt(self.current_mol),
            'logp': Descriptors.MolLogP(self.current_mol),
            'qed': QED.qed(self.current_mol),
            'pathway_scores': self._get_pathway_scores(self.current_mol),
            'diversity': self._calculate_diversity(),
        }

        new_state = self.get_state()
        
        if self.agent is not None:
            self.agent.update_state_mol_cache(new_state, self.current_mol)

        return new_state, reward, done, info

    def calculate_complexity_bonus(self, action):
        if isinstance(action, tuple):
            fragment_index = action[1]
        else:
            fragment_index = (action - 1) % len(self.all_fragments)
        
        fragment = self.all_fragments[fragment_index]
        complexity_level = self.fragment_complexity[fragment]
        return 0.1 * complexity_level  # Adjust this multiplier as needed


    def is_valid_action(self, atom_index, fragment_index):
        if self.current_mol is None:
            return False
        if atom_index >= self.current_mol.GetNumAtoms() or fragment_index >= len(self.all_fragments):
            return False

        atom = self.current_mol.GetAtomWithIdx(atom_index)
        fragment_smiles = self.all_fragments[fragment_index]
        fragment_mol = Chem.MolFromSmiles(fragment_smiles)
        
        if fragment_mol is None:
            return False

        periodic_table = GetPeriodicTable()
        default_valence = periodic_table.GetDefaultValence(atom.GetAtomicNum())
        
        # If default valence is -1, we can't determine validity based on valence
        if default_valence == -1:
            return True  # Allow the action, but you might want to add additional checks

        current_valence = atom.GetTotalValence()
        if current_valence >= default_valence:
            return False

        # Check if adding the fragment would exceed the default valence
        fragment_atom = fragment_mol.GetAtomWithIdx(0)
        new_valence = current_valence + fragment_atom.GetTotalValence() - 1  # Subtract 1 for the bond being formed

        if new_valence > default_valence:
            return False

        # Check for potential ring formation
        if fragment_mol.GetNumAtoms() > 1:
            combined_mol = Chem.RWMol(self.current_mol)
            combined_mol.AddAtom(fragment_atom)
            combined_mol.AddBond(atom_index, combined_mol.GetNumAtoms() - 1, Chem.BondType.SINGLE)
            
            try:
                Chem.SanitizeMol(combined_mol)
            except:
                return False

        # Check for potential stereochemistry issues
        if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
            return False

        # Check for potential aromaticity breaking
        if atom.GetIsAromatic() and not fragment_atom.GetIsAromatic():
            return False

        return True    
        
    def _compute_reward(self, mol):
        smiles = Chem.MolToSmiles(mol)
        reward_components = {}
        # Docking score (if applicable)
        if self.curriculum_level >= 3:
            try:
                docking_score = predict_docking_score(smiles)
                reward_components['docking'] = -docking_score / 10 * SCALING_FACTORS['docking']  # Normalize and scale
            except Exception as e:
                logger.error(f"Error calculating docking score: {e}")
                reward_components['docking'] = 0.0
        else:
            reward_components['docking'] = 0.0
        # Pathway targeting
        pathway_scores = self._get_pathway_scores(mol)
        if pathway_scores:
            reward_components['pathway'] = np.mean(pathway_scores) * SCALING_FACTORS['pathway']
        else:
            reward_components['pathway'] = 0.0
        # Multi-target bonus
        reward_components['multitarget'] = self.multitarget_score * SCALING_FACTORS['multi_target']

        # Drug-likeness (Lipinski's Rule of Five)
        lipinski_violations = self._lipinski_violations(mol)
        reward_components['drug_likeness'] = max(0, 4 - lipinski_violations) * SCALING_FACTORS['drug_likeness']
        # Synthetic accessibility
        sa_score = calculateScore(mol)
        reward_components['synthetic_accessibility'] = (10 - sa_score) / 2 * SCALING_FACTORS['synthetic_accessibility']
        # Novelty
        reward_components['novelty'] = self._calculate_novelty(mol) * SCALING_FACTORS['novelty']
        # Diversity
        reward_components['diversity'] = self._calculate_diversity() * SCALING_FACTORS['diversity']
        
        # QED (Quantitative Estimate of Drug-likeness)
        qed = QED.qed(mol)
        reward_components['qed'] = qed * SCALING_FACTORS['qed']
        logger.info(f"QED Score: {qed}")
        # Apply difficulty multiplier based on curriculum level
        difficulty_multiplier = 1 + (self.curriculum_level - 1) * 0.5  # Increases with curriculum level
        for key in reward_components:
            reward_components[key] *= difficulty_multiplier  # Adjust multiplier as needed
        # Normalize rewards
        normalized_components = {}
        for component, value in reward_components.items():
            self._update_reward_stats(component, value)
            stats = self.reward_stats[component]
            normalized_value = (value - stats['mean']) / (stats['std'] + 1e-8)
            normalized_components[component] = normalized_value
        # Calculate total reward
        total_reward = sum(normalized_components.values())
        # Penalty for termination without meeting objectives
        if not self._objectives_met(mol):
            total_reward -= 5
        # Clip reward to prevent extreme values
        total_reward = np.clip(total_reward, -10, 10)
        logger.debug(f"Total Reward: {total_reward}")
        return total_reward

    def _update_reward_stats(self, component, value):
        if component not in self.reward_stats:
            self.reward_stats[component] = {'mean': value, 'std': 1.0}
        else:
            # Update mean and std using exponential moving average
            alpha = 0.99
            self.reward_stats[component]['mean'] = alpha * self.reward_stats[component]['mean'] + (1 - alpha) * value
            self.reward_stats[component]['std'] = alpha * self.reward_stats[component]['std'] + (1 - alpha) * abs(value - self.reward_stats[component]['mean'])

    def _load_known_molecule_fingerprints(self, known_molecules_smiles):
        fingerprints = []
        for smiles in known_molecules_smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                fingerprints.append(fp)
            else:
                logger.warning(f"Invalid SMILES in known_molecules_smiles: {smiles}")
        return fingerprints

    def _lipinski_violations(self, mol):
        violations = 0
        if Descriptors.MolWt(mol) > 500: violations += 1
        if Descriptors.MolLogP(mol) > 5: violations += 1
        if Descriptors.NumHDonors(mol) > 5: violations += 1
        if Descriptors.NumHAcceptors(mol) > 10: violations += 1
        return violations


    def calculate_qed_score(self):
        if self.current_mol is not None:
            self.qed_score = QED.qed(self.current_mol)
        else:
            self.qed_score = 0.0
        return self.qed_score
    
    
        
    def calculate_multitarget_score(self, mol):
        pathway_scores = self._get_pathway_scores(mol)
        logger.debug(f"Raw pathway scores: {pathway_scores}")
        if self.agent_objectives:
            active_pathways = sum(score > MULTI_TARGET_THRESHOLD for score in pathway_scores)
            self.multitarget_score = active_pathways / len(self.agent_objectives)
            logger.info(f"Multitarget Score: {self.multitarget_score} (Active Pathways: {active_pathways}/{len(self.agent_objectives)})")
            logger.debug(f"Pathway scores above threshold: {[score > MULTI_TARGET_THRESHOLD for score in pathway_scores]}")
        else:
            self.multitarget_score = 0.0
            logger.warning("No agent objectives set, multitarget score defaulting to 0.0")
        return self.multitarget_score

    
    def _calculate_novelty(self, mol):
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        if not self.known_molecule_fps:
            return 1.0  # If no known molecules, consider it completely novel
        max_similarity = max(DataStructs.TanimotoSimilarity(fp, known_fp) for known_fp in self.known_molecule_fps)
        return 1 - max_similarity

    def _calculate_diversity(self):
        if len(self.generated_mols) < 2:
            self.diversity_score = 0.0
            return self.diversity_score
        
        fps = [Chem.RDKFingerprint(mol) for mol in self.generated_mols]
        similarities = []
        for i in range(len(fps)):
            for j in range(i+1, len(fps)):
                similarity = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                similarities.append(similarity)
        
        self.diversity_score = 1 - (sum(similarities) / len(similarities)) if similarities else 0.0
        logger.info(f"Diversity Score: {self.diversity_score}")
        return self.diversity_score

    def create_initial_molecule(self):
        # Create a new molecule with a single carbon atom
        mol = Chem.MolFromSmiles('C')
        if mol is None:
            logger.warning("Failed to create initial molecule")
            return None
        return mol

    def get_state(self):
        if self.current_mol is None:
            logger.warning("current_mol is None in get_state()")
            return np.zeros(2048)  # Return a zero vector if current_mol is None
        return self.mol_to_features(self.current_mol)

    def mol_to_features(self, mol):
        if mol is None:
            logger.warning("mol is None in mol_to_features()")
            return np.zeros(2048, dtype=np.float32)  # Return a zero vector as a fallback
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        return np.array(fp)

    def _get_pathway_scores(self, mol):
        scores = []
        for pathway in self.agent_objectives:
            scoring_function = self.pathway_scoring_functions.get(pathway)
            if scoring_function:
                score = scoring_function(mol)
                scores.append(score if score is not None else 0)
            else:
                scores.append(0)
        return scores

    def _objectives_met(self, mol):
        pathway_scores = self._get_pathway_scores(mol)
        return any(score > MULTI_TARGET_THRESHOLD for score in pathway_scores)

    def _update_curriculum(self):
        # This method can be used for additional curriculum updates if needed
        pass

    def state_to_mol(self, state):
        # Convert state (fingerprint) to a unique string representation
        state_key = ''.join(map(str, state))
        
        # Check if we've already converted this state
        if state_key in self.state_mol_cache:
            return self.state_mol_cache[state_key]
        
        # If not, we need to reconstruct the molecule
        # This assumes that the state is a Morgan fingerprint
        # and that we're using ECFP4 (radius 2)
        all_mols = self.get_all_mols()  # Ensure this method is correctly implemented
        
        best_match = None
        best_similarity = -1
        
        for mol in all_mols:
            mol_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=len(state))
            similarity = DataStructs.TanimotoSimilarity(mol_fp, state)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = mol
        
        # Cache the result
        self.state_mol_cache[state_key] = best_match
        
        return best_match

    def get_all_mols(self):
        return self.diverse_mols + [self.current_mol]
    def set_agent(self, agent):
        self.agent = agent

