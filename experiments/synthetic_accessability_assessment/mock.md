Experiment 5: Synthetic Accessibility Assessment

Objective: Evaluate how easily the generated molecules can be synthesized using available chemical reactions.

Implementation Steps:

    Use RDKit's SAScore:
        Calculate the Synthetic Accessibility (SA) score for each molecule.

    Aggregate Scores:
        After each episode, compute average SA scores.

    Logging:
        Record SA scores for analysis.

Code Integration:

Ensure you have an SA scorer implemented (you have sascorer.calculateScore). Modify your logging to include SA scores.

python

# Inside the training loop, after step execution

for idx, agent in enumerate(multi_agent_system.agents): # Existing logging code...

    # Calculate Synthetic Accessibility Score
    mol = agent.env.current_mol
    sa_score = agent.env.calculate_sas_score(mol)

    # Update log entry
    log_entry.update({
        'SA_Score': sa_score
    })

    # Modify `fieldnames` to include 'SA_Score'
    fieldnames = [
        'Episode', 'Agent_ID', 'Curriculum_Level', 'Reward', 'Running_Reward',
        'SMILES_Sequence', 'Objectives', 'Pathway_Score', 'Diversity_Score',
        'Multitarget_Score', 'Rationale', 'Avg_Complexity', 'QED_Score',
        'Lipinski_Violations', 'SA_Score', 'Individual_Objectives'
    ]

Explanation:

    SA Score: Lower scores indicate easier synthesis (range typically ~1 to 10).
    Thresholds: Define acceptable SA score ranges based on your synthesis capabilities.
