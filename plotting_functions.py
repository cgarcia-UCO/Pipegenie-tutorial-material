import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import logging
import requests
from datetime import datetime
import sys

def download_file(url, save_path):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an error for bad status codes
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
        #print(f"File downloaded successfully and saved as '{save_path}'")
    except requests.exceptions.RequestException as e:
        print(f"Download failed: {e}")

def detect_log_format(text):
    if '-> Fitness:' in text:
        return 'type1'
    elif text.startswith("pipeline") or '\tpipeline\t' in text or 'pipeline\t' in text:
        return 'type2'
    return None

def parse_type1(text):
    pipeline_blocks = re.findall(r'\d+:\s+Pipeline\((.*?)\)\s+->\s+Fitness', text, re.DOTALL)
    #return [re.findall(r'\d+:\s+([A-Za-z0-9_]+)\(', block) for block in pipeline_blocks]
    pipelines = [extract_top_level_function_calls(block.replace('\n','')) for block in pipeline_blocks]
    return pipelines

def extract_top_level_function_calls(pipeline_str):
    calls = []
    i = 0
    n = len(pipeline_str)

    while i < n:
        # Skip whitespace or arrows
        if pipeline_str[i:].startswith('->'):
            i += 2
            continue
        elif pipeline_str[i].isspace():
            i += 1
            continue

        # Match outer function name
        match = re.match(r'[a-zA-Z_][a-zA-Z0-9_]*', pipeline_str[i:])
        if not match:
            i += 1
            continue

        func_start = i
        func_name = match.group()
        i += len(func_name)

        # Expecting opening parenthesis
        if i >= n or pipeline_str[i] != '(':
            continue

        # Parse through nested parentheses to find full argument list
        i += 1
        depth = 1
        arg_start = i
        while i < n and depth > 0:
            if pipeline_str[i] == '(':
                depth += 1
            elif pipeline_str[i] == ')':
                depth -= 1
            i += 1

        # Extract inner content (e.g. scale inside normalize(scale(data)))
        inner_expr = pipeline_str[arg_start:i-1].strip()
        inner_match = re.search(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', inner_expr)
        if inner_match:
            arg_func_name = inner_match.group(1)
            calls.append(f"{func_name}({arg_func_name})")
        else:
            calls.append(func_name)

    return calls

def parse_type2(text):
    lines = text.strip().splitlines()
    pipelines = []
    for line in lines[1:]:  # Skip header
        if not line.strip():
            continue
        parts = line.split('\t')
        if len(parts) < 2:
            continue
        pipeline_str, fitness = parts[0], parts[1]
        if 'invalid_ind' in fitness or 'eval_error' in fitness:
            continue
        # extract function names (e.g., fastICA, adaBoost)
        #steps = re.findall(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", pipeline_str)
        steps = extract_top_level_function_calls(pipeline_str)
        if steps:
            pipelines.append(steps)

    return pipelines

def split_steps(pipelines):
    preprocessor_counts = defaultdict(int)
    classifier_counts = defaultdict(int)

    for steps in pipelines:
        if not steps:
            continue
        classifier = steps[-1]
        preprocessors = steps[:-1]
        classifier_counts[classifier] += 1
        for prep in preprocessors:
            preprocessor_counts[prep] += 1

    return preprocessor_counts, classifier_counts

def plot_heatmap(counts_dict, title):
    df = pd.DataFrame.from_dict(counts_dict, orient='index', columns=['Frequency'])
    df.sort_values(by='Frequency', ascending=False, inplace=True)
    df = df.T  # Transpose to make one row, multiple columns
    plt.figure(figsize=(max(6, len(df.columns) * 0.6), 2.5))
    sns.heatmap(df, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(title)
    plt.yticks(rotation=0)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def process_pipeline_log(log_text):
    log_type = detect_log_format(log_text)
    if log_type == 'type1':
        parsed_pipelines = parse_type1(log_text)
    elif log_type == 'type2':
        parsed_pipelines = parse_type2(log_text)
    else:
        raise ValueError("Unknown log format.")

    preprocessor_counts, classifier_counts = split_steps(parsed_pipelines)
    plot_heatmap(preprocessor_counts, "Preprocessor Usage Frequency")
    plot_heatmap(classifier_counts, "Classifier Usage Frequency")

def parse_type2_generations(log_text, pipelines_per_gen):
    lines = log_text.strip().splitlines()[1:]  # Skip header
    valid_pipelines = []

    for line in lines:
        parts = line.split('\t')
        if len(parts) < 2:
            continue
        pipeline_str, fitness = parts[0], parts[1]
        # if 'invalid_ind' in fitness or 'eval_error' in fitness:
        #     continue
        steps = re.findall(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", pipeline_str)
        if steps:
            valid_pipelines.append(steps)

    valid_pipelines = parse_type2(log_text)

    # Group into generations
    generations = []
    for i in range(0, len(valid_pipelines), pipelines_per_gen):
        generations.append(valid_pipelines[i:i+pipelines_per_gen])
    return generations

def count_components(generations):
    preprocessor_counts = []
    classifier_counts = []

    for gen in generations:
        prep_counter = Counter()
        clf_counter = Counter()
        for steps in gen:
            if not steps:
                continue
            clf = steps[-1]
            preps = steps[:-1]
            clf_counter[clf] += 1
            for prep in preps:
                prep_counter[prep] += 1
        preprocessor_counts.append(prep_counter)
        classifier_counts.append(clf_counter)

    return preprocessor_counts, classifier_counts

def compute_usage_over_time(component_list, counts_per_gen):
    usage_data = {component: [] for component in component_list}

    for counts in counts_per_gen:
        total = sum(counts.values()) if counts else 1
        for comp in component_list:
            usage_data[comp].append(100 * counts.get(comp, 0) / total)

    return pd.DataFrame(usage_data)

def plot_usage(df, title):
    plt.figure(figsize=(10, 5))
    for column in df.columns:
        plt.plot(df.index, df[column], marker='o', label=column)
    plt.title(title)
    plt.xlabel('Generation')
    plt.ylabel('Usage Percentage (%)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def analyze_pipeline_evolution(log_text, pipelines_per_gen,
                               num_showed_components, num_last_gens):
    generations = parse_type2_generations(log_text, pipelines_per_gen)
    if not generations:
        print("No valid pipelines found.")
        return

    preprocessor_counts, classifier_counts = count_components(generations)

    # Get most common from last generation
    preprocessor_flattened = Counter()
    for i in range(len(preprocessor_counts)-1,max(-1,len(preprocessor_counts)-num_last_gens-1),-1):
      preprocessor_flattened = preprocessor_flattened + Counter(preprocessor_counts[i])

    classifier_flattened = Counter()
    for i in range(len(classifier_counts)-1,max(-1,len(classifier_counts)-num_last_gens-1),-1):
      classifier_flattened = classifier_flattened + Counter(classifier_counts[i])

    # top = [x[0] for x in b_flattened.most_common(num_showed_components)]
    top_last_preprocessors = [x[0] for x in preprocessor_counts[-1].most_common(num_showed_components)]
    top_last_classifiers = [x[0] for x in classifier_counts[-1].most_common(num_showed_components)]
    top_preprocessors = set([x[0] for x in preprocessor_flattened.most_common(num_showed_components)] + top_last_preprocessors)
    top_classifiers = set([x[0] for x in classifier_flattened.most_common(num_showed_components)] + top_last_classifiers)

    # Compute usage over generations
    df_preps = compute_usage_over_time(top_preprocessors, preprocessor_counts)
    df_clfs = compute_usage_over_time(top_classifiers, classifier_counts)

    # Plot
    plot_usage(df_preps, f"Top {num_showed_components} Preprocessors Usage Over Generations (+ top last one)")
    plot_usage(df_clfs, f"Top {num_showed_components} Classifiers Usage Over Generations (+ top last one)")

def plot_fitness_and_size(log_file_path, minimization=False):
    """
    Reads a log file and plots:
    - Fitness avg and std on the primary y-axis.
    - Size avg and std on the secondary y-axis.
    """
    # Read log file
    with open(log_file_path, 'r') as f:
        lines = [
            line.strip() for line in f.readlines()
            if line.strip() and not line.startswith("---") and not line.startswith("gen") and not line.startswith("-")
        ]

    generations = []
    fitness_avg = []
    fitness_std = []
    size_avg = []
    size_std = []

    # Parse each line
    for line in lines:
        parts = line.split()
        if len(parts) >= 12:
            generations.append(int(parts[0]))
            fitness_avg.append(float(parts[3 if not minimization else 2]))
            fitness_std.append(float(parts[4]))
            size_avg.append(float(parts[7]))
            size_std.append(float(parts[8]))

    # Plotting
    fig, ax1 = plt.subplots()

    # Primary y-axis: fitness
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Fitness Max / Avg', color='tab:blue')
    ax1.plot(generations, fitness_avg, label='Fitness Max', marker='o', color='tab:blue')
    ax1.plot(generations, fitness_std, label='Fitness Avg', marker='x', linestyle='--', color='tab:cyan')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.legend(loc='upper left')

    # Secondary y-axis: size
    ax2 = ax1.twinx()
    ax2.set_ylabel('Size Max / Avg', color='tab:red')
    ax2.plot(generations, size_avg, label='Size Max', marker='s', color='tab:red')
    ax2.plot(generations, size_std, label='Size Avg', marker='^', linestyle='--', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.legend(loc='upper right')

    plt.title("Evolution of Fitness and Size over Generations")
    plt.tight_layout()
    plt.grid(True)
    plt.show()

from pipegenie.classification import PipegenieClassifier
from pipegenie.regression import PipegenieRegressor

class MultiLineFormatter(logging.Formatter):
    def format(self, record):
        timestamp = datetime.now().strftime('%H:%M:%S')
        original_message = super().format(record)
        lines = original_message.splitlines()
        return '\n'.join(f'{timestamp} - {line}' for line in lines)

def format_logger(a_logger, model):
  if a_logger.hasHandlers():
    a_logger.handlers.clear()

  a_logger.setLevel(logging.INFO)
  handler = logging.FileHandler(model.outdir_path.joinpath("evolution.txt"), mode="w")
  a_logger.addHandler(handler)
  console = logging.StreamHandler(sys.stdout)
  #formatter = logging.Formatter('%(message)s')
  #formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S')
  formatter = MultiLineFormatter('%(message)s')
  console.setFormatter(formatter)
  a_logger.addHandler(console)
  a_logger.propagate = False

def format_pipegenie_evolution_logger(func):
  def wrapper(*args, **kwargs):
        a_logger = logging.getLogger("pipegenie_individuals")
        if a_logger.hasHandlers():
          a_logger.handlers.clear()
        a_logger.propagate = False
        model = func(*args, **kwargs)
        format_logger(logging.getLogger("pipegenie_evolution"), model)
        return model
  return wrapper

PipegenieClassifier = format_pipegenie_evolution_logger(PipegenieClassifier)
PipegenieRegressor = format_pipegenie_evolution_logger(PipegenieRegressor)
