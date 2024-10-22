
import os
import torch
import torch_geometric
from torch_geometric.data import Data, Batch
from rdkit import Chem
from rdkit.Chem import AllChem
from utils.sascorer import calculateScore
from transform_and_path import PATHWAY_ACTIVES
import logging
from rdkit.Chem import DataStructs

# GNN model definition (same as in training script)
class GNN(torch.nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        self.conv1 = torch_geometric.nn.GCNConv(1, 64)
        self.conv2 = torch_geometric.nn.GCNConv(64, 64)
        self.conv3 = torch_geometric.nn.GCNConv(64, 64)
        self.lin = torch.nn.Linear(64, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.conv3(x, edge_index)
        
        x = torch_geometric.nn.global_mean_pool(x, batch)
        x = self.lin(x)
        
        return x


# Add this before loading the model
def reinitialize_model_parameters(model):
    for param in model.parameters():
        if param.dim() > 1:
            torch.nn.init.xavier_uniform_(param)
        else:
            torch.nn.init.zeros_(param)
    print("Model parameters reinitialized")


# Load the trained model
model = GNN()
print("Model structure:")
print(model)

checkpoint = torch.load(r'C:\Users\wes\MOL_RL_AGENT_v2\models\gnn_docking_surrogate_final.pth', map_location=torch.device('cpu'))
print("Checkpoint keys:", checkpoint.keys())

if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model state dict loaded successfully")
else:
    print("Error: 'model_state_dict' not found in checkpoint")
    print("Available keys:", checkpoint.keys())

model.eval()

# Add this after loading the model if parameters are still NaN
if any(torch.isnan(param).any() for param in model.parameters()):
    print("NaN values detected in model parameters. Reinitializing...")
    reinitialize_model_parameters(model)



def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    
    # Get node features
    node_features = []
    for atom in mol.GetAtoms():
        node_features.append(atom.GetAtomicNum())
    x = torch.tensor(node_features, dtype=torch.float).view(-1, 1)
    
    # Get edge index
    edges = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edges.append((i, j))
        edges.append((j, i))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    # Create Data object
    data = Data(x=x, edge_index=edge_index)
    
    return data


def predict_docking_score(smiles):
    graph_data = smiles_to_graph(smiles)
    with torch.no_grad():
        batch = Batch.from_data_list([graph_data])
        prediction = model(batch)
        print(f"Raw prediction: {prediction}")
        if torch.isnan(prediction).any():
            print("Warning: NaN detected in prediction")
    return prediction.item()




######################################################################################################################################
######################################################################################################################################
######################################################################################################################################
     



# Create similarity scoring functions
def create_similarity_scoring_function(pathway):
    # Precompute fingerprints for known actives
    active_smiles = PATHWAY_ACTIVES.get(pathway, [])
    active_mols = []
    for smiles in active_smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            active_mols.append(mol)
        else:
            print(f"Invalid SMILES for pathway '{pathway}': {smiles}")
    active_fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024) for mol in active_mols]

    def score_function(mol):
        if not active_fps:
            return 0.0  # Return 0.0 if no valid active fingerprints
        mol_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        similarities = [DataStructs.TanimotoSimilarity(mol_fp, active_fp) for active_fp in active_fps]
        max_similarity = max(similarities)
        return max_similarity  # Return the maximum similarity score

    return score_function


def calculate_sas_score(mol):
    return calculateScore(mol)
