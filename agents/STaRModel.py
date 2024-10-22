from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import torch.nn as nn
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class STaRModel(nn.Module):
    def __init__(self, model_name='seyonec/ChemBERTa-zinc-base-v1', max_length=512):
        super(STaRModel, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.to(device)
        self.max_length = max_length
        self.training_buffer = []

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)

    def generate_rationale(self, prompt, mol_repr, env, agent_objectives):
        try:
            if env.current_mol is None:
                print("Warning: current_mol is None")
                return "No molecule to analyze."
            
            mol = env.current_mol
            print(f"Generate rationale: mol type: {type(mol)}")
            
            smiles = Chem.MolToSmiles(mol)
            mol_weight = Descriptors.ExactMolWt(mol)
            logp = Descriptors.MolLogP(mol)
            pathway_scores = env._get_pathway_scores(mol)
            
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            rotatable_bonds = Descriptors.NumRotatableBonds(mol)
            diversity_score = env._calculate_diversity()
            
            if agent_objectives:
                multitarget_score = sum(score > 0.9 for score in pathway_scores) / len(agent_objectives)
            else:
                multitarget_score = 0.0
            
            rationale = f"Addressing the prompt: '{prompt}'\n\n"
            rationale += f"The molecule's representation has {len(mol_repr)} features. "
            rationale += f"The mean value of these features is {mol_repr.mean():.4f}, "
            rationale += f"with a standard deviation of {mol_repr.std():.4f}.\n\n"
            rationale += f"The generated molecule has a molecular weight of {mol_weight:.2f} g/mol and a LogP of {logp:.2f}. "
            rationale += f"It has {hbd} hydrogen bond donors and {hba} hydrogen bond acceptors. "
            rationale += f"The number of rotatable bonds is {rotatable_bonds}. "
            
            if pathway_scores:
                rationale += f"The molecule's average pathway score is {np.mean(pathway_scores):.4f}, "
            else:
                rationale += f"No pathway scores available. "
            
            rationale += f"with a diversity score of {diversity_score:.4f} and a multi-target score of {multitarget_score:.4f}.\n\n"
            
            if pathway_scores:
                rationale += "Top affected aging pathways:\n"
                for pathway, score in zip(agent_objectives, pathway_scores):
                    rationale += f"- {pathway}: {score:.4f}\n"
                    print(f"Debug: {pathway} score: {score:.4f}")
                rationale += "\n"
            else:
                rationale += "No pathway scores to analyze.\n\n"
            if multitarget_score > 0.5:
                rationale += "This molecule shows promise in targeting multiple aging pathways. "
            elif pathway_scores and np.mean(pathway_scores) > 0.7:
                rationale += "While not highly multi-targeted, this molecule shows strong activity in at least one aging pathway. "
            else:
                rationale += "This molecule may require further optimization to improve its effects on aging pathways. "
            if 2 <= logp <= 5:
                rationale += "The LogP value suggests good oral bioavailability. "
            elif logp > 5:
                rationale += "The high LogP value may lead to poor oral bioavailability. "
            else:
                rationale += "The low LogP value may result in rapid clearance from the body. "
            if 150 <= mol_weight <= 500:
                rationale += "The molecular weight is within a desirable range for drug-like molecules. "
            elif mol_weight > 500:
                rationale += "The high molecular weight may negatively impact oral bioavailability. "
            else:
                rationale += "The low molecular weight suggests this could be a fragment or building block for larger molecules. "
            # Conclusion
            rationale += "\n\nIn conclusion, "
            if pathway_scores and np.mean(pathway_scores) > 0.7 and 2 <= logp <= 5 and 150 <= mol_weight <= 500:
                rationale += "this molecule shows promising characteristics for targeting aging pathways with good drug-like properties."
            elif pathway_scores and (np.mean(pathway_scores) > 0.5 or (2 <= logp <= 5 and 150 <= mol_weight <= 500)):
                rationale += "this molecule has some positive attributes but may need further optimization to improve its overall profile."
            else:
                rationale += "this molecule may require significant modifications to enhance its potential as a drug candidate for targeting aging pathways."
            return rationale

        except Exception as e:
            print(f"Error in generate_rationale: {e}")
            return "No molecule to analyze."

    def add_to_buffer(self, prompt, rationale, mol_repr):
        self.training_buffer.append({"prompt": prompt, "rationale": rationale, "mol_repr": mol_repr})

    def train_on_buffer(self, learning_rate=1e-5, epochs=1, batch_size=2):
        if not self.training_buffer:
            return
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.model.train()
        for epoch in range(epochs):
            random.shuffle(self.training_buffer)
            total_loss = 0
            for i in range(0, len(self.training_buffer), batch_size):
                batch = self.training_buffer[i:i+batch_size]
                inputs = self.tokenizer([item["prompt"] + item["rationale"] for item in batch], 
                                        return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")
        self.training_buffer = []