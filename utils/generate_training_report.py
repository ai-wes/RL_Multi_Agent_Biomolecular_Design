from termcolor import colored
import csv
import os
from collections import defaultdict
import numpy as np
import json  # Import json to read best_run.json
from transform_and_path import AGING_PATHWAYS  # Import the list of pathways
import logging

logger = logging.getLogger(__name__)

def generate_training_report(log_file, log_data, output_dir, best_run_path='best_run.json'):
    # Save log data as JSON
    log_file = os.path.join(output_dir, 'log_data.json')
    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=2)

    # Calculate overall statistics
    overall_stats = {}
    if log_data:
        for key in log_data[0].keys():
            if key not in ['Episode', 'Agent_ID', 'SMILES_Sequence', 'Objectives', 'Rationale', 'Individual_Objectives']:
                values = [float(entry[key]) for entry in log_data if entry[key] != '']
                overall_stats[key] = {
                    'min': min(values) if values else 0,
                    'max': max(values) if values else 0,
                    'avg': sum(values) / len(values) if values else 0
                }
    else:
        logger.warning("log_data is empty. No statistics can be calculated.")
        overall_stats = {
            'Reward': {'min': 0, 'max': 0, 'avg': 0},
            'Running_Reward': {'min': 0, 'max': 0, 'avg': 0},
            'QED_Score': {'min': 0, 'max': 0, 'avg': 0},
            'Pathway_Score': {'min': 0, 'max': 0, 'avg': 0},
            'Diversity_Score': {'min': 0, 'max': 0, 'avg': 0},
            'Multitarget_Score': {'min': 0, 'max': 0, 'avg': 0}
        }
    # Initialize statistics
    curriculum_stats = defaultdict(lambda: defaultdict(lambda: {'sum': 0, 'count': 0}))
    sequence_stats = defaultdict(lambda: defaultdict(list))
    pathway_stats = {pathway: {'Reward': [], 'Pathway_Score': []} for pathway in AGING_PATHWAYS}
    best_sequences = {}

    # Initialize new statistics
    qed_scores = []
    diversity_scores = []
    multitarget_scores = []

    # Process each row
    for row in log_data:
        episode = int(row['Episode'])
        curriculum_level = float(row['Curriculum_Level'])
        reward = float(row['Reward'])
        running_reward = float(row['Running_Reward'])
        # qed_score might not be present in all rows
        qed_score = float(row.get('QED_Score', 0))
        smiles_sequence = row['SMILES_Sequence']
        # Pathway might not be directly present; adjust based on actual log structure
        pathway = row.get('Pathway', 'Unknown')
        pathway_score = float(row.get('Pathway_Score', 0))
        diversity_score = float(row.get('Diversity_Score', 0))
        multitarget_score = float(row.get('Multitarget_Score', 0))
        # Update pathway statistics
        if pathway in pathway_stats:
            pathway_stats[pathway]['Reward'].append(reward)
            pathway_stats[pathway]['Pathway_Score'].append(pathway_score)

        # Update overall statistics
        for metric in ['Reward', 'Running_Reward', 'Loss', 'QED_Score', 'Pathway_Score', 'Diversity_Score', 'Multitarget_Score']:
            value = float(row.get(metric, 0))
            overall_stats[metric]['min'] = min(overall_stats[metric]['min'], value)
            overall_stats[metric]['max'] = max(overall_stats[metric]['max'], value)
            overall_stats[metric]['sum'] += value
            overall_stats[metric]['count'] += 1

        # Update curriculum-specific statistics
        for metric in ['Reward', 'QED_Score', 'Pathway_Score', 'Diversity_Score', 'Multitarget_Score']:
            value = float(row.get(metric, 0))
            curriculum_stats[curriculum_level][metric]['sum'] += value
            curriculum_stats[curriculum_level][metric]['count'] += 1

        # Update sequence statistics
        sequence_stats[smiles_sequence]['Reward'].append(reward)
        sequence_stats[smiles_sequence]['QED_Score'].append(qed_score)
        sequence_stats[smiles_sequence]['Curriculum_Level'].append(curriculum_level)
        sequence_stats[smiles_sequence]['Diversity_Score'].append(diversity_score)
        sequence_stats[smiles_sequence]['Multitarget_Score'].append(multitarget_score)

        # Track best sequences
        if reward > overall_stats['Reward']['max']:
            best_sequences['Highest_Reward'] = {
                'Episode': episode,
                'Curriculum_Level': curriculum_level,
                'Reward': reward,
                'QED_Score': qed_score,
                'SMILES_Sequence': smiles_sequence,
                'Pathway': pathway,
                'Pathway_Score': pathway_score,
                'Diversity_Score': diversity_score,
                'Multitarget_Score': multitarget_score
            }
        if qed_score > overall_stats['QED_Score']['max']:
            best_sequences['Highest_QED'] = {
                'Episode': episode,
                'Curriculum_Level': curriculum_level,
                'Reward': reward,
                'QED_Score': qed_score,
                'SMILES_Sequence': smiles_sequence,
                'Pathway': pathway,
                'Pathway_Score': pathway_score,
                'Diversity_Score': diversity_score,
                'Multitarget_Score': multitarget_score
            }
        if diversity_score > overall_stats['Diversity_Score']['max']:
            best_sequences['Highest_Diversity'] = {
                'Episode': episode,
                'Curriculum_Level': curriculum_level,
                'Reward': reward,
                'QED_Score': qed_score,
                'SMILES_Sequence': smiles_sequence,
                'Pathway': pathway,
                'Pathway_Score': pathway_score,
                'Diversity_Score': diversity_score,
                'Multitarget_Score': multitarget_score
            }
        if multitarget_score > overall_stats['Multitarget_Score']['max']:
            best_sequences['Highest_Multitarget'] = {
                'Episode': episode,
                'Curriculum_Level': curriculum_level,
                'Reward': reward,
                'QED_Score': qed_score,
                'SMILES_Sequence': smiles_sequence,
                'Pathway': pathway,
                'Pathway_Score': pathway_score,
                'Diversity_Score': diversity_score,
                'Multitarget_Score': multitarget_score
            }

        # Extract and store new scores
        qed_scores.append(float(row.get('QED_Score', 0)))
        diversity_scores.append(float(row.get('Diversity_Score', 0)))
        multitarget_scores.append(float(row.get('Multitarget_Score', 0)))

    # Calculate averages for overall statistics
    for metric in overall_stats:
        if overall_stats[metric]['count'] > 0:
            overall_stats[metric]['avg'] = overall_stats[metric]['sum'] / overall_stats[metric]['count']
        else:
            overall_stats[metric]['avg'] = 0  # or some other default value
    # Calculate curriculum-specific averages
    for level in curriculum_stats:
        for metric in curriculum_stats[level]:
            total = curriculum_stats[level][metric]['sum']
            count = curriculum_stats[level][metric]['count']
            if count > 0:
                curriculum_stats[level][metric]['avg'] = total / count
            else:
                curriculum_stats[level][metric]['avg'] = 0  # or some other default value

    # Calculate sequence averages and sort
    sequence_avg_scores = []
    for sequence, metrics in sequence_stats.items():
        avg_reward = np.mean(metrics['Reward']) if metrics['Reward'] else 0
        avg_qed = np.mean(metrics['QED_Score']) if metrics['QED_Score'] else 0
        avg_curriculum = np.mean(metrics['Curriculum_Level']) if metrics['Curriculum_Level'] else 0
        avg_diversity = np.mean(metrics['Diversity_Score']) if metrics['Diversity_Score'] else 0
        avg_multitarget = np.mean(metrics['Multitarget_Score']) if metrics['Multitarget_Score'] else 0
        sequence_avg_scores.append((sequence, avg_reward, avg_qed, avg_curriculum, avg_diversity, avg_multitarget))
    sequence_avg_scores.sort(key=lambda x: x[1], reverse=True)
    # Calculate pathway averages
    pathway_avg_scores = []
    for pathway, metrics in pathway_stats.items():
        avg_reward = np.mean(metrics['Reward']) if metrics['Reward'] else 0
        avg_pathway_score = np.mean(metrics['Pathway_Score']) if metrics['Pathway_Score'] else 0
        pathway_avg_scores.append((pathway, avg_reward, avg_pathway_score))
    pathway_avg_scores.sort(key=lambda x: x[1], reverse=True)

    # Calculate averages for new scores
    avg_qed = np.mean(qed_scores) if qed_scores else 0
    avg_diversity = np.mean(diversity_scores) if diversity_scores else 0
    avg_multitarget = np.mean(multitarget_scores) if multitarget_scores else 0

    # Include new scores in the report
    report_data = {
        # ... (existing report data)
        'avg_qed_score': avg_qed,
        'avg_diversity_score': avg_diversity,
        'avg_multitarget_score': avg_multitarget,
    }

    # Load Best Run Data
    if os.path.exists(best_run_path):
        with open(best_run_path, 'r') as f:
            best_run = json.load(f)
        best_average_reward = best_run.get('average_reward', None)
        best_model_name = best_run.get('model_name', 'Unknown')
    else:
        best_run = None
        best_average_reward = None
        best_model_name = 'No previous best run'

    # Generate text report
    txt_file_path = os.path.join(output_dir, 'training_report.txt')
    with open(txt_file_path, 'w') as f:
        f.write("Training Statistics Report\n")
        f.write("==========================\n\n")

        # Overall Statistics
        f.write("Overall Statistics:\n")
        f.write("-----------------\n")
        for metric in overall_stats:
            f.write(f"{metric}:\n")
            avg_value = overall_stats[metric]['avg']
            if best_average_reward is not None and metric == 'Reward':
                comparison = avg_value - best_average_reward
                color = "green" if comparison >= 0 else "red"
                f.write(f"  Average: {avg_value:.4f} ({'+' if comparison >=0 else ''}{comparison:.4f})\n")
            else:
                f.write(f"  Average: {avg_value:.4f}\n")
            f.write(f"  Minimum: {overall_stats[metric]['min']:.4f}\n")
            f.write(f"  Maximum: {overall_stats[metric]['max']:.4f}\n\n")
        # Curriculum Level Statistics
        f.write("Curriculum Level Statistics:\n")
        f.write("----------------------------\n")
        for level in sorted(curriculum_stats.keys()):
            f.write(f"Curriculum Level {level}:\n")
            for metric in curriculum_stats[level]:
                avg_metric = curriculum_stats[level][metric].get('avg', 0)
                f.write(f"  Average {metric}: {avg_metric:.4f}\n")
            f.write("\n")

        # Comparison with Best Run
        if best_run:
            f.write(f"Comparison with Best Run:\n")
            f.write(f"------------------------\n")
            f.write(f"Best Run Model Name: {best_model_name}\n")
            f.write(f"Best Run Average Reward: {best_average_reward:.4f}\n")
            current_avg_reward = overall_stats['Reward']['avg']
            comparison = current_avg_reward - best_average_reward
            color = "green" if comparison >= 0 else "red"
            f.write(f"Current Run Average Reward: {current_avg_reward:.4f} ({'+' if comparison >=0 else ''}{comparison:.4f})\n")
            f.write("\n")

        # SMILES Sequences Ranked by Average Reward
        f.write("SMILES Sequences Ranked by Average Reward:\n")
        f.write("------------------------------------------\n")
        for i, (sequence, avg_reward, avg_qed, avg_curriculum, avg_diversity, avg_multitarget) in enumerate(sequence_avg_scores[:20], 1):
            f.write(f"{i}. SMILES: {sequence}\n")
            f.write(f"   Average Reward: {avg_reward:.4f}\n")
            f.write(f"   Average QED Score: {avg_qed:.4f}\n")
            f.write(f"   Average Curriculum Level: {avg_curriculum:.2f}\n")
            f.write(f"   Average Diversity Score: {avg_diversity:.4f}\n")
            f.write(f"   Average Multitarget Score: {avg_multitarget:.4f}\n\n")

        # Pathway Statistics
        f.write("Pathway Statistics:\n")
        f.write("-------------------\n")
        for pathway, avg_reward, avg_pathway_score in pathway_avg_scores:
            f.write(f"Pathway: {pathway}\n")
            if avg_reward > 0 or avg_pathway_score > 0:
                f.write(f"  Average Reward: {avg_reward:.4f}\n")
                f.write(f"  Average Pathway Score: {avg_pathway_score:.4f}\n")
            else:
                f.write("  No data available for this pathway\n")
            f.write("\n")

        # Best Sequences
        f.write("Best Sequences:\n")
        f.write("---------------\n")
        for category, sequence in best_sequences.items():
            f.write(f"{category}:\n")
            for key, value in sequence.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
    # Generate HTML report
    html_file_path = os.path.join(output_dir, 'training_report.html')
    with open(html_file_path, 'w') as f:
        f.write("""
        <html>
        <head>
            <style>
                body { 
                    font-family: Arial, sans-serif; 
                    background-color: #1e1e1e; 
                    color: #d4d4d4; 
                    line-height: 1.6;
                    padding: 20px;
                }
                h1, h2, h3 { color: #569cd6; }
                .green { color: #6A9955; }
                .red { color: #F44747; }
                table { 
                    border-collapse: collapse; 
                    width: 100%;
                    margin-bottom: 20px;
                }
                th, td { 
                    border: 1px solid #3c3c3c; 
                    padding: 12px; 
                    text-align: left;
                }
                th { 
                    background-color: #252526; 
                    color: #569cd6;
                }
                tr:nth-child(even) { background-color: #252526; }
                tr:nth-child(odd) { background-color: #1e1e1e; }
                tr:hover { background-color: #2d2d2d; }
            </style>
        </head>
        <body>
        <h1>Training Statistics Report</h1>
        
        <h2>Overall Statistics:</h2>
        <table>
            <tr><th>Metric</th><th>Average</th><th>Minimum</th><th>Maximum</th></tr>
        """)
        for metric in overall_stats:
            f.write(f"<tr><td>{metric}</td>")
            if best_average_reward is not None and metric == 'Reward':
                avg_value = overall_stats[metric]['avg']
                comparison = avg_value - best_average_reward
                color_class = "green" if comparison >= 0 else "red"
                comparison_str = f"{'+' if comparison >=0 else ''}{comparison:.4f}"
                f.write(f"""<td class="{color_class}">{avg_value:.4f} ({comparison_str})</td>""")
            else:
                avg_value = overall_stats[metric]['avg']
                f.write(f"<td>{avg_value:.4f}</td>")
            f.write(f"<td>{overall_stats[metric]['min']:.4f}</td><td>{overall_stats[metric]['max']:.4f}</td></tr>\n")

        f.write("</table>")

        # Curriculum Level Statistics
        f.write("<h2>Curriculum Level Statistics:</h2>")
        for level in sorted(curriculum_stats.keys()):
            f.write(f"<h3>Curriculum Level {level}:</h3>")
            f.write("<table><tr><th>Metric</th><th>Average</th></tr>")
            for metric in curriculum_stats[level]:
                avg_metric = curriculum_stats[level][metric].get('avg', 0)
                f.write(f"<tr><td>{metric}</td><td>{avg_metric:.4f}</td></tr>")
            f.write("</table>")
        # Comparison with Best Run
        if best_run:
            f.write(f"<h2>Comparison with Best Run:</h2>")
            f.write(f"<p><strong>Best Run Model Name:</strong> {best_model_name}</p>")
            f.write(f"<p><strong>Best Run Average Reward:</strong> {best_average_reward:.4f}</p>")
            current_avg_reward = overall_stats['Reward']['avg']
            comparison = current_avg_reward - best_average_reward
            color_class = "green" if comparison >= 0 else "red"
            comparison_str = f"{'+' if comparison >=0 else ''}{comparison:.4f}"
            f.write(f"<p class='{color_class}'><strong>Current Run Average Reward:</strong> {current_avg_reward:.4f} ({comparison_str})</p>")
            f.write("<br>")

        # SMILES Sequences Ranked by Average Reward
        f.write("<h2>SMILES Sequences Ranked by Average Reward:</h2>")
        f.write("<table><tr><th>Rank</th><th>SMILES</th><th>Average Reward</th><th>Average QED Score</th><th>Average Curriculum Level</th><th>Average Diversity Score</th><th>Average Multitarget Score</th></tr>")
        for i, (sequence, avg_reward, avg_qed, avg_curriculum, avg_diversity, avg_multitarget) in enumerate(sequence_avg_scores[:20], 1):
            f.write(f"""
            <tr>
                <td>{i}</td>
                <td>{sequence}</td>
                <td>{avg_reward:.4f}</td>
                <td>{avg_qed:.4f}</td>
                <td>{avg_curriculum:.2f}</td>
                <td>{avg_diversity:.4f}</td>
                <td>{avg_multitarget:.4f}</td>
            </tr>
            """)


        # Pathway Statistics
        f.write("<h2>Pathway Statistics:</h2>")
        f.write("<table><tr><th>Pathway</th><th>Average Reward</th><th>Average Pathway Score</th></tr>")
        for pathway, avg_reward, avg_pathway_score in pathway_avg_scores:
            f.write(f"<tr><td>{pathway}</td>")
            if avg_reward > 0 or avg_pathway_score > 0:
                f.write(f"<td>{avg_reward:.4f}</td><td>{avg_pathway_score:.4f}</td></tr>")
            else:
                f.write(f"<td colspan='2'>No data available for this pathway</td></tr>")
        f.write("</table>")

        # Best Sequences
        f.write("<h2>Best Sequences:</h2>")
        f.write("---------------\n")
        for category, sequence in best_sequences.items():
            f.write(f"<h3>{category}:</h3>")
            f.write("<ul>")
            for key, value in sequence.items():
                f.write(f"<li><strong>{key}:</strong> {value}</li>")
            f.write("</ul>")

        f.write("""
        </body>
        </html>
        """)

    print(f"Training report generated and saved to:")
    print(f"  Text file: {txt_file_path}")
    print(f"  HTML file: {html_file_path}")

    return txt_file_path, html_file_path, log_file