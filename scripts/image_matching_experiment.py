from dotenv import load_dotenv
load_dotenv()

import os
import numpy as np
import json
from PIL import Image
import argparse
from tqdm import tqdm
import logging
logging.basicConfig(level=logging.INFO)

# Import your existing functions
from lib.analyses.behavior.experiment_helper import yield_trial_images, pdistance
from bonner.caching import BONNER_CACHING_HOME
import datetime


def create_set_assignment_tracker(data_dir, num_subjects, subject_id=0):
    """
    Create a tracking file to manage subject ID assignments.
    """
    tracker_path = os.path.join(data_dir, "subject_assignment_tracker.json")
    
    tracker_data = {
        "total_subjects": num_subjects,
        "next_participant_id": subject_id,  # Sequential participant counter
        "available_ids": [],  # NEW: Pool of canceled IDs for reuse
        "participant_ids": {},
        "participant_status": {},  # NEW: Track status (started/completed/canceled)
        "start_times": {},  # NEW: Track when participants started
        "completion_times": {},  # NEW: Track when participants completed
        "subject_assignments": {},
        "last_update": datetime.datetime.now().isoformat()
    }
    
    with open(tracker_path, 'w') as f:
        json.dump(tracker_data, f, indent=2)
    
    logging.info(f"Created subject assignment tracker at {tracker_path}")
    
    return tracker_path

def save_pil_image(pil_img, output_path, resize_to=(112, 112)):
    """
    Save a PIL image to a file with resizing.
    
    Parameters:
    pil_img: PIL Image object
    output_path: Path to save the image
    resize_to: Tuple (width, height) for resizing the image
    """
    # Resize the image to reduce file size
    resized_img = pil_img.resize(resize_to, Image.LANCZOS)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the resized image
    resized_img.save(output_path, optimize=True, quality=85)

def setup_experiment_folders(output_dir, subject_id):
    """Set up folders for the experiment."""
    # Create main output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories
    directories = {
        'images': os.path.join(output_dir, "images"),
        'data': os.path.join(output_dir, "data", f"subject_{subject_id:02d}"),
        'html': os.path.join(output_dir, "html"),
        'css': os.path.join(output_dir, "css"),
        'php': os.path.join(output_dir, "php")
    }
    
    for dir_path in directories.values():
        os.makedirs(dir_path, exist_ok=True)

    # Create CSS file for target-cluster layout
    css_file = os.path.join(directories['css'], "style.css")
    with open(css_file, 'w') as f:
        f.write("""
/* Target-Cluster Experiment Styles */
body {
    font-family: Arial, sans-serif;
    margin: 0;
    padding: 20px;
    background-color: #f5f5f5;
    max-width: 1200px;
    margin: 0 auto;
    box-sizing: border-box;
}

.trial-container {
    background-color: white;
    padding: 20px;
    border-radius: 8px;
    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
    margin-bottom: 20px;
    width: 100%;
    box-sizing: border-box;
    overflow-x: hidden;
}

/* Target image section */
.target-section {
    text-align: center;
    margin-bottom: 30px;
    padding: 20px;
    background-color: #f8f9ff;
    border-radius: 10px;
    border: 2px solid #e0e0e0;
}

.target-image {
    margin: 15px 0;
}

.target-image img {
    width: 150px;
    height: 150px;
    border: 3px solid #333;
    border-radius: 5px;
    object-fit: cover;
}

/* Clusters section */
.clusters-container {
    display: flex;
    flex-direction: row;
    gap: 20px;
    justify-content: space-between;
    margin-bottom: 20px;
    width: 100%;
    box-sizing: border-box;
}

.cluster {
    flex: 1;
    max-width: calc(50% - 10px);
    background-color: #f9f9f9;
    border-radius: 8px;
    padding: 15px;
    box-shadow: 0 0 5px rgba(0, 0, 0, 0.1);
    border: 2px solid transparent;
    transition: all 0.3s ease;
    box-sizing: border-box;
}

.clickable-cluster {
    cursor: pointer;
    user-select: none;
}

.clickable-cluster:hover {
    border-color: #4CAF50;
    box-shadow: 0 0 10px rgba(76, 175, 80, 0.3);
    transform: translateY(-2px);
}

.clickable-cluster:active {
    transform: translateY(0px);
    box-shadow: 0 0 5px rgba(76, 175, 80, 0.5);
}

.cluster.selected {
    border-color: #4CAF50;
    background-color: #e8f5e8;
    box-shadow: 0 0 15px rgba(76, 175, 80, 0.4);
}

.cluster-heading {
    text-align: center;
    margin-bottom: 15px;
}

.cluster-label {
    font-size: 20px;
    font-weight: bold;
    color: #333;
}

.click-hint {
    font-size: 12px;
    color: #666;
    font-style: italic;
    margin-top: 2px;
}

/* Image grid for clusters */
.cluster-images {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    grid-template-rows: repeat(2, 1fr);
    gap: 8px;
    justify-items: center;
    align-items: center;
    width: 100%;
    max-width: 300px;
    margin: 0 auto;
}

.cluster-images img {
    width: 80px;
    height: 80px;
    object-fit: cover;
    border-radius: 4px;
    border: 1px solid #ddd;
    max-width: 100%;
    box-sizing: border-box;
}

/* Instructions and UI */
.question {
    font-size: 18px;
    margin-bottom: 15px;
    font-weight: bold;
    text-align: center;
    color: #333;
}

.keyboard-instructions {
    text-align: center;
    margin-bottom: 20px;
    font-size: 14px;
    color: #666;
}

.key-indicator {
    display: inline-block;
    padding: 4px 8px;
    background-color: #f0f0f0;
    border: 1px solid #ccc;
    border-radius: 4px;
    font-family: monospace;
    font-weight: bold;
    margin: 0 2px;
}

.progress-bar {
    width: 100%;
    background-color: #e0e0e0;
    border-radius: 5px;
    margin-bottom: 20px;
    height: 25px;
}

.progress {
    height: 25px;
    background-color: #4CAF50;
    border-radius: 5px;
    text-align: center;
    color: white;
    font-weight: bold;
    line-height: 25px;
    transition: width 0.3s ease;
}

/* Responsive design */
@media (max-width: 768px) {
    .clusters-container {
        flex-direction: column;
        gap: 20px;
    }
    
    .cluster {
        max-width: 100%;
        flex: none;
    }
    
    .target-image img {
        width: 120px;
        height: 120px;
    }
    
    .cluster-images {
        grid-template-columns: repeat(2, 1fr);
        grid-template-rows: repeat(3, 1fr);
        max-width: 200px;
    }
    
    .cluster-images img {
        width: 60px;
        height: 60px;
    }
}

@media (max-width: 1000px) {
    .cluster-images img {
        width: 70px;
        height: 70px;
    }
    
    .cluster-images {
        max-width: 250px;
    }
}

/* Instructions and completion screens */
.instructions {
    margin-bottom: 30px;
    font-size: 16px;
    line-height: 1.6;
}

.completion-message {
    text-align: center;
    color: #4CAF50;
    margin-bottom: 20px;
}

.completion-code {
    font-size: 24px;
    font-weight: bold;
    text-align: center;
    background-color: #e8f5e8;
    padding: 15px;
    border-radius: 5px;
    border: 2px solid #4CAF50;
    margin: 20px 0;
}

.spacebar-button {
    display: block;
    text-align: center;
    font-size: 16px;
    padding: 15px 30px;
    background-color: #f5f5f5;
    border-radius: 5px;
    border: 1px solid #ddd;
    margin: 20px auto;
    max-width: 300px;
    cursor: pointer;
    transition: background-color 0.3s ease;
    user-select: none;
}

.spacebar-button:hover,
.clickable-start:hover {
    background-color: #e8e8e8;
    border-color: #4CAF50;
}

.example-notice {
    text-align: center;
    font-size: 24px;
    font-weight: bold;
    color: #333;
    margin-bottom: 20px;
    padding: 10px;
    background-color: #fffbf0;
    border: 2px solid #ffc107;
    border-radius: 8px;
}
        """)
    
    return directories

def generate_example_trial(
    dataset: str,
    output_images_dir: str,
    n_within_cluster: int = 5,
    seed: int = None,
    image_size: tuple = (112, 112)
):
    """
    Generate an example trial with obvious correct answer (all same images as target).
    """
    from lib.datasets import load_stimulus_set
    
    if seed is None:
        from lib.utilities import SEED
        seed = SEED
    
    np.random.seed(seed + 9999)  # Different seed for example trial
    
    logging.info("Generating example trial...")
    
    stimulus_set = load_stimulus_set(f"{dataset}_test")
    total_stimuli = len(stimulus_set)
    
    # Select target image
    target_idx = np.random.randint(0, total_stimuli)
    target_img = stimulus_set[target_idx]
    
    # Save target image
    target_filename = f"img_{target_idx}.png"
    target_path = os.path.join(output_images_dir, target_filename)
    save_pil_image(target_img, target_path, resize_to=image_size)
    
    # Correct cluster (all same as target)
    correct_cluster_paths = []
    for _ in range(n_within_cluster):
        filename = f"img_{target_idx}.png"
        correct_cluster_paths.append(filename)
    
    # Wrong cluster (random different images)
    wrong_cluster_paths = []
    available_indices = list(range(total_stimuli))
    available_indices.remove(target_idx)
    wrong_indices = np.random.choice(available_indices, n_within_cluster, replace=False)

    for stim_idx in wrong_indices:
        wrong_img = stimulus_set[stim_idx]
        filename = f"img_{stim_idx}.png"
        full_path = os.path.join(output_images_dir, filename)
        save_pil_image(wrong_img, full_path, resize_to=image_size)
        wrong_cluster_paths.append(filename)
    
    # Randomly decide which cluster is correct
    correct_is_a = np.random.choice([True, False])
    
    example_trial = {
        'trial_id': 'example_trial',
        'trial_num': -1,  # Special number for example
        'is_example_trial': True,
        'target': {
            'image_path': target_filename
        },
        'clusters': [
            {
                'label': 'A',
                'images': correct_cluster_paths if correct_is_a else wrong_cluster_paths,
                'is_correct': correct_is_a
            },
            {
                'label': 'B',
                'images': wrong_cluster_paths if correct_is_a else correct_cluster_paths,
                'is_correct': not correct_is_a
            }
        ],
        'correct_answer': 'A' if correct_is_a else 'B',
        'target_idx': int(target_idx),
        'correct_indices': [int(target_idx)] * n_within_cluster,  # All same as target
        'wrong_indices': [int(idx) for idx in wrong_indices],
    }
    
    return example_trial

def generate_catch_trials(
    dataset: str,
    output_images_dir: str,
    max_trials: int,
    n_within_cluster: int = 5,
    seed: int = None,
    image_size: tuple = (112, 112),
):
    """
    Generate catch trials by creating actual catch trial images.
    """
    from lib.datasets import load_stimulus_set
    
    if seed is None:
        from lib.utilities import SEED
        seed = SEED
    
    np.random.seed(seed)
    
    # Calculate number of catch trials (1/20 of main trials)
    num_catch_trials = max(1, max_trials // 20)
    
    logging.info(f"Generating {num_catch_trials} catch trials...")
    
    # Load stimulus set
    stimulus_set = load_stimulus_set(f"{dataset}_test")
    total_stimuli = len(stimulus_set)
    
    catch_trials = []
    
    for i in range(num_catch_trials):
        # Select target image
        target_idx = np.random.randint(0, total_stimuli)
        target_img = stimulus_set[target_idx]
        
        # Save target image
        target_filename = f"img_{target_idx}.png"
        target_path = os.path.join(output_images_dir, target_filename)
        save_pil_image(target_img, target_path, resize_to=image_size)
        
        # Generate correct cluster (5 copies of target)
        correct_cluster_paths = []
        for _ in range(n_within_cluster):
            filename = f"img_{target_idx}.png"
            correct_cluster_paths.append(filename)
        
        # Generate wrong cluster (5 random different images)
        wrong_cluster_paths = []
        available_indices = list(range(total_stimuli))
        available_indices.remove(target_idx)
        wrong_indices = np.random.choice(available_indices, n_within_cluster, replace=False)

        for stim_idx in wrong_indices:
            wrong_img = stimulus_set[stim_idx]
            filename = f"img_{stim_idx}.png"
            full_path = os.path.join(output_images_dir, filename)
            save_pil_image(wrong_img, full_path, resize_to=image_size)
            wrong_cluster_paths.append(filename)

        # Randomly decide which cluster is correct
        correct_is_a = np.random.choice([True, False])
        
        catch_trial = {
            'trial_id': f"catch_trial_{i:03d}",
            'trial_num': i,
            'is_catch_trial': True,
            'target': {
                'image_path': target_filename
            },
            'clusters': [
                {
                    'label': 'A',
                    'images': correct_cluster_paths if correct_is_a else wrong_cluster_paths,
                    'is_correct': correct_is_a
                },
                {
                    'label': 'B',
                    'images': wrong_cluster_paths if correct_is_a else correct_cluster_paths,
                    'is_correct': not correct_is_a
                }
            ],
            'correct_answer': 'A' if correct_is_a else 'B',
            'target_idx': int(target_idx),
            'correct_indices': [int(target_idx)] * n_within_cluster,  # All same as target
            'wrong_indices': [int(idx) for idx in wrong_indices],
        }
        
        catch_trials.append(catch_trial)
    
    return catch_trials

def generate_image_matching_trials(
    analysis: str,
    dataset: str,
    load_dataset_kwargs: dict,
    scorer_kwargs: dict,
    time: float,
    distance_fn: callable,
    output_images_dir: str,
    residual: bool = False,
    n_within_cluster: int = 5,
    threshold_percentile: int = 80,
    seed: int = None,
    image_size: tuple = (112, 112),
    max_trials: int = 100,
    start_trial: int = 0,
):
    """
    Generate trials using either shared images or creating new ones.
    """
    
    trials = []
    image_path_mapping = {}
    
    # Original behavior - generate new images
    logging.info(f"Generating new images for time={time}, analysis={analysis}...")
        
    # Generate main trials with subset selection
    trial_generator = yield_trial_images(
        analysis=analysis,
        dataset=dataset,
        load_dataset_kwargs=load_dataset_kwargs,
        scorer_kwargs=scorer_kwargs,
        time=time,
        distance_fn=distance_fn,
        residual=residual,
        n_within_cluster=n_within_cluster,
        threshold_percentile=threshold_percentile,
        seed=seed,
    )
    
    # Skip to start_trial index and take max_trials from there
    trial_list = list(trial_generator)
    
    # random list of trials
    random_indices = np.random.permutation(len(trial_list))
    selected_trials = [trial_list[i] for i in random_indices[start_trial:start_trial + max_trials]]
    
    for trial_num, trial_data in enumerate(tqdm(selected_trials, desc="Processing main trials")):
            
        # Save target image
        target_filename = f"img_{trial_data.get('target_idx', -1)}.png"
        target_path = os.path.join(output_images_dir, target_filename)
        save_pil_image(trial_data['target'], target_path, resize_to=image_size)
        target_rel_path = target_filename
        
        # Save cluster 0 images
        cluster_0_paths = []
        for img_idx, img in enumerate(trial_data['cluster_0']):
            filename = f"img_{trial_data.get('cluster_0_indices', [])[img_idx]}.png"
            full_path = os.path.join(output_images_dir, filename)
            save_pil_image(img, full_path, resize_to=image_size)
            rel_path = filename
            cluster_0_paths.append(rel_path)
        
        # Save cluster 1 images
        cluster_1_paths = []
        for img_idx, img in enumerate(trial_data['cluster_1']):
            filename = f"img_{trial_data.get('cluster_1_indices', [])[img_idx]}.png"
            full_path = os.path.join(output_images_dir, filename)
            save_pil_image(img, full_path, resize_to=image_size)
            rel_path = filename
            cluster_1_paths.append(rel_path)
        
        # Create trial structure
        trial = {
            'trial_id': f"{analysis}.time={time}_trial_{trial_num:03d}",
            'subject_trial_num': trial_num,
            'pair_trial_idx': start_trial + trial_num,
            'time': time,
            'analysis': analysis,
            'is_catch_trial': False,
            'target': {
                'image_path': target_rel_path
            },
            'clusters': [
                {
                    'label': 'A',
                    'images': cluster_0_paths,
                    'is_correct': trial_data['cluster_correct'] == 0
                },
                {
                    'label': 'B',
                    'images': cluster_1_paths,
                    'is_correct': trial_data['cluster_correct'] == 1
                }
            ],
            'correct_answer': 'A' if trial_data['cluster_correct'] == 0 else 'B',
            'target_idx': trial_data.get('target_idx', -1),
            'cluster_0_indices': trial_data.get('cluster_0_indices', []).tolist(),
            'cluster_1_indices': trial_data.get('cluster_1_indices', []).tolist(),
        }
        
        trials.append(trial)
    
    logging.info(f"Generated {len(trials)} main trials")
    
    # Generate catch trials
    catch_trials = generate_catch_trials(
        dataset=dataset,
        output_images_dir=output_images_dir,
        max_trials=max_trials,
        n_within_cluster=n_within_cluster,
        seed=seed,
        image_size=image_size,
    )

    # Add catch trials to main trials
    trials.extend(catch_trials)

    logging.info(f"Total trials (including catch): {len(trials)}")
    return trials

def create_html_experiment(html_dir, max_trials):
    """Create HTML file for the target-cluster matching experiment."""
    main_html = os.path.join(html_dir, "experiment.html")
    
    # Create the HTML content without f-string to avoid issues with JavaScript braces
    html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image-matching Experiment</title>
    <link rel="stylesheet" href="../css/style.css">
</head>
<body>
    <div id="experiment-container">
        <!-- Welcome Screen -->
        <div id="welcome-screen" class="trial-container" style="display: block;">
            <h1>Image-matching Experiment</h1>
            <div class="instructions">
                <p>Welcome! In this experiment, you will see a target image and two groups of images.</p>
                
                <p><strong>Your task:</strong> Decide which group the target image belongs to based on visual similarity.</p>
                
                <p><strong>Instructions:</strong></p>
                <ol>
                    <li>Look at the target image (shown at the top)</li>
                    <li>Examine both groups of images below</li>
                    <li>Press <span class="key-indicator">1</span> if the target belongs to Group 1</li>
                    <li>Press <span class="key-indicator">2</span> if the target belongs to Group 2</li>
                </ol>
                
                <p>Make your decision based on which group contains images that are most similar to the target image.</p>
                
                <p><strong>Important:</strong> The task can be challenging as similarities may be subtle. Make your decision based on your first impression (within a few seconds).</p>
                
                <p><strong>Note:</strong> We will start with an example trial where the correct answer is obvious, but in the actual experiment, the correct answer will be less obvious. There will also be a few easy catch trials mixed in to ensure you are paying attention.</p>
                
                <p>You will complete ''' + str(max_trials) + ''' trials.</p>
                
                <div class="spacebar-button clickable-start">Press <span class="key-indicator">Space</span> to Start<br><em>or click here</em></div>
            </div>
        </div>
        
        <!-- Loading Screen -->
        <div id="loading-screen" class="trial-container" style="display: none;">
            <h1>Loading Experiment</h1>
            <p>Please wait while we prepare your trials...</p>
            <div class="progress-bar">
                <div id="loading-bar" class="progress" style="width: 0%;">0%</div>
            </div>
        </div>
        
        <!-- Example Trial Screen -->
        <div id="example-screen" class="trial-container" style="display: none;">
            <div class="example-notice">Example Trial</div>
            
            <!-- Target Image Section -->
            <div class="target-section">
                <h3>Target Image</h3>
                <div class="target-image">
                    <img id="example-target-img" src="" alt="Target Image">
                </div>
                <p class="question">Which group should this image belong to?</p>
            </div>
            
            <!-- Instructions -->
            <div class="keyboard-instructions">
                Press <span class="key-indicator">1</span> for Group 1 or <span class="key-indicator">2</span> for Group 2<br>
                <em>Or click on a group to select it</em>
            </div>
            
            <!-- Clusters -->
            <div class="clusters-container">
                <div class="cluster clickable-cluster" id="example-cluster-a" data-group="1">
                    <div class="cluster-heading">
                        <div class="cluster-label">Group 1</div>
                        <div class="click-hint">Click to select</div>
                    </div>
                    <div class="cluster-images" id="example-cluster-a-images">
                        <!-- Images will be inserted here -->
                    </div>
                </div>
                
                <div class="cluster clickable-cluster" id="example-cluster-b" data-group="2">
                    <div class="cluster-heading">
                        <div class="cluster-label">Group 2</div>
                        <div class="click-hint">Click to select</div>
                    </div>
                    <div class="cluster-images" id="example-cluster-b-images">
                        <!-- Images will be inserted here -->
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Post-Example Screen -->
        <div id="post-example-screen" class="trial-container" style="display: none;">
            <h1>Good job!</h1>
            <div class="instructions">
                <p>In the example trial, the correct answer was obvious because all images in one group were identical to the target.</p>
                
                <p>In the actual experiment, the similarities will be much more subtle. The images won't be identical, but you'll need to decide which group is more visually similar to the target.</p>
                
                <p>Remember: Make your decision based on your first impression within a few seconds.</p>
                
                <div class="spacebar-button clickable-start">Press <span class="key-indicator">Space</span> to Begin<br><em>or click here</em></div>
            </div>
        </div>
        
        <!-- Main Trial Screen -->
        <div id="trial-screen" class="trial-container" style="display: none;">
            <div class="progress-bar">
                <div id="progress" class="progress" style="width: 0%;">Trial 1 of ''' + str(max_trials) + '''</div>
            </div>
            
            <!-- Target Image Section -->
            <div class="target-section">
                <h3>Target Image</h3>
                <div class="target-image">
                    <img id="target-img" src="" alt="Target Image">
                </div>
                <p class="question">Which group should this image belong to?</p>
            </div>
            
            <!-- Instructions -->
            <div class="keyboard-instructions">
                Press <span class="key-indicator">1</span> for Group 1 or <span class="key-indicator">2</span> for Group 2<br>
                <em>Or click on a group to select it</em>
            </div>
            
            <!-- Clusters -->
            <div class="clusters-container">
                <div class="cluster clickable-cluster" id="cluster-a" data-group="1">
                    <div class="cluster-heading">
                        <div class="cluster-label">Group 1</div>
                        <div class="click-hint">Click to select</div>
                    </div>
                    <div class="cluster-images" id="cluster-a-images">
                        <!-- Images will be inserted here -->
                    </div>
                </div>
                
                <div class="cluster clickable-cluster" id="cluster-b" data-group="2">
                    <div class="cluster-heading">
                        <div class="cluster-label">Group 2</div>
                        <div class="click-hint">Click to select</div>
                    </div>
                    <div class="cluster-images" id="cluster-b-images">
                        <!-- Images will be inserted here -->
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Completion Screen -->
        <div id="completion-screen" class="trial-container" style="display: none;">
            <h1 class="completion-message">Experiment Complete!</h1>
            <p>Thank you for participating in our study.</p>
            <p>Your responses have been saved.</p>
            <div class="completion-code">COMPLETED</div>
        </div>
    </div>

    <script>
        // Experiment state
        let exampleTrial = null;
        let allTrials = [];
        let currentTrialIndex = 0;
        let responses = [];
        let experimentStartTime = Date.now();
        let trialStartTime = Date.now();
        let inExamplePhase = false;
        
        // DOM elements
        const welcomeScreen = document.getElementById('welcome-screen');
        const loadingScreen = document.getElementById('loading-screen');
        const exampleScreen = document.getElementById('example-screen');
        const postExampleScreen = document.getElementById('post-example-screen');
        const trialScreen = document.getElementById('trial-screen');
        const completionScreen = document.getElementById('completion-screen');
        
        // Fisher-Yates shuffle algorithm
        function shuffleArray(array) {
            // Get a unique seed from the Prolific ID or timestamp if not available
            const prolificId = new URLSearchParams(window.location.search).get("PROLIFIC_PID") || '';
            let seed = 0;
            
            // Create a simple hash from the Prolific ID
            if (prolificId) {
                for (let i = 0; i < prolificId.length; i++) {
                    seed = ((seed << 5) - seed) + prolificId.charCodeAt(i);
                    seed = seed >>> 0;  // Convert to 32-bit unsigned
                }
            } else {
                // If no Prolific ID, use current timestamp
                seed = Date.now();
            }
            
            // Set a seeded random function
            const seededRandom = function() {
                seed = (seed * 9301 + 49297) % 233280;
                return seed / 233280;
            };
            
            // Perform Fisher-Yates shuffle
            for (let i = array.length - 1; i > 0; i--) {
                const j = Math.floor(seededRandom() * (i + 1));
                [array[i], array[j]] = [array[j], array[i]];
            }
            
            return array;
        }
        
        async function getAssignedSubject() {
            const prolificId = new URLSearchParams(window.location.search).get("PROLIFIC_PID") || 'TEST-PARTICIPANT';
            
            try {
                const response = await fetch(`../php/assign_subject.php?prolific_pid=${prolificId}`);
                const data = await response.json();
                
                if (response.ok) {
                    console.log(`Assigned participant number: ${data.participant_number}, subject folder: ${data.subject_folder}`);
                    window.participantNumber = data.participant_number;
                    window.subjectId = data.subject_folder.replace('subject_', '');  // Extract just the ID
                    return data.subject_folder;
                } else {
                    console.error('Error getting subject assignment:', data.error);
                    window.subjectId = '00';  // Default
                    return 'subject_00';
                }
            } catch (error) {
                console.error('Error fetching subject assignment:', error);
                window.subjectId = '00';  // Default
                return 'subject_00';
            }
        }
        
        // Load trial data
        async function loadTrials() {
            try {
                // Get assigned subject folder
                const subjectFolder = await getAssignedSubject();
                
                console.log(`Attempting to load trials from ../data/${subjectFolder}/all_trials.json`);
                const response = await fetch(`../data/${subjectFolder}/all_trials.json`);
                
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
                
                const data = await response.json();
                
                // Separate example trial and main trials
                exampleTrial = data.example_trial;
                allTrials = data.main_trials;
                
                if (!Array.isArray(allTrials) || allTrials.length === 0) {
                    throw new Error('No trials found in data file');
                }
                
                // Shuffle trials for this participant
                shuffleArray(allTrials);
                
                console.log(`Successfully loaded ${allTrials.length} trials plus example`);
                return true;
            } catch (error) {
                console.error('Error loading trials:', error);
                alert(`Error loading experiment data: ${error.message}\\n\\nPlease check:\\n1. Server is running from correct directory\\n2. Trial data was generated successfully\\n3. all_trials.json exists in data/ folder`);
                return false;
            }
        }
        
        // Show example trial
        function showExampleTrial() {
            if (!exampleTrial) return;
            
            inExamplePhase = true;
            trialStartTime = Date.now();
            
            // Display target image  
            const targetImg = document.getElementById('example-target-img');
            if (exampleTrial.target.image_path.startsWith('../')) {
                targetImg.src = exampleTrial.target.image_path;
            } else {
                targetImg.src = `../images/${exampleTrial.target.image_path}`;
            }
            
            // Display cluster images
            const clusterAContainer = document.getElementById('example-cluster-a-images');
            const clusterBContainer = document.getElementById('example-cluster-b-images');
            
            clusterAContainer.innerHTML = '';
            clusterBContainer.innerHTML = '';
            
            // Find clusters A and B
            const clusterA = exampleTrial.clusters.find(c => c.label === 'A');
            const clusterB = exampleTrial.clusters.find(c => c.label === 'B');
            
            // Add cluster A images
            if (clusterA && clusterA.images) {
                clusterA.images.forEach(imgPath => {
                    const img = document.createElement('img');
                    if (imgPath.startsWith('../')) {
                        img.src = imgPath;
                    } else {
                        img.src = `../images/${imgPath}`;
                    }
                    img.alt = 'Cluster A Image';
                    clusterAContainer.appendChild(img);
                });
            }
            
            // Add cluster B images
            if (clusterB && clusterB.images) {
                clusterB.images.forEach(imgPath => {
                    const img = document.createElement('img');
                    if (imgPath.startsWith('../')) {
                        img.src = imgPath;
                    } else {
                        img.src = `../images/${imgPath}`;
                    }
                    img.alt = 'Cluster B Image';
                    clusterBContainer.appendChild(img);
                });
            }
            
            // Add click listeners to clusters
            document.getElementById('example-cluster-a').addEventListener('click', handleExampleClick);
            document.getElementById('example-cluster-b').addEventListener('click', handleExampleClick);
            
            // Add key listener for example
            document.addEventListener('keydown', handleExampleKeyPress);
        }
        
        // Handle example trial responses
        function handleExampleKeyPress(e) {
            if (e.key === '1' || e.key === '2') {
                handleExampleResponse(e.key);
            }
        }
        
        function handleExampleClick(e) {
            const cluster = e.currentTarget;
            const selectedGroup = cluster.getAttribute('data-group');
            
            // Visual feedback
            cluster.classList.add('selected');
            setTimeout(() => {
                cluster.classList.remove('selected');
            }, 200);
            
            handleExampleResponse(selectedGroup);
        }
        
        function handleExampleResponse(selectedGroup) {
            const selectedLabel = selectedGroup === '1' ? 'A' : 'B';
            const selectedCluster = exampleTrial.clusters.find(c => c.label === selectedLabel);
            const isCorrect = selectedCluster.is_correct;
            
            // Calculate response time
            const trialResponseTime = Date.now() - trialStartTime;
            
            // Record example response
            responses.push({
                trial_id: 'example_trial',
                trial_num: -1,
                is_example_trial: true,
                selected_group: selectedGroup,
                selected_label: selectedLabel,
                correct_answer: exampleTrial.correct_answer,
                is_correct: isCorrect,
                response_time_ms: trialResponseTime,
                timestamp: new Date().toISOString()
            });
            
            // Clean up listeners
            document.removeEventListener('keydown', handleExampleKeyPress);
            document.getElementById('example-cluster-a').removeEventListener('click', handleExampleClick);
            document.getElementById('example-cluster-b').removeEventListener('click', handleExampleClick);
            
            // Move to post-example screen
            exampleScreen.style.display = 'none';
            postExampleScreen.style.display = 'block';
            inExamplePhase = false;
        }
        
        // Show current trial
        function showCurrentTrial() {
            const trial = allTrials[currentTrialIndex];
            if (!trial) return;
            
            // Record when this trial started
            trialStartTime = Date.now();
            
            // Update progress
            const progress = Math.round(((currentTrialIndex + 1) / allTrials.length) * 100);
            const progressBar = document.getElementById('progress');
            progressBar.style.width = `${progress}%`;
            progressBar.textContent = `Trial ${currentTrialIndex + 1} of ${allTrials.length}`;
            
            // Display target image  
            const targetImg = document.getElementById('target-img');
            if (trial.target.image_path.startsWith('../')) {
                targetImg.src = trial.target.image_path;
            } else {
                targetImg.src = `../images/${trial.target.image_path}`;
            }
            
            // Display cluster images
            const clusterAContainer = document.getElementById('cluster-a-images');
            const clusterBContainer = document.getElementById('cluster-b-images');
            
            clusterAContainer.innerHTML = '';
            clusterBContainer.innerHTML = '';
            
            // Find clusters A and B
            const clusterA = trial.clusters.find(c => c.label === 'A');
            const clusterB = trial.clusters.find(c => c.label === 'B');
            
            // Add cluster images
            if (clusterA && clusterA.images) {
                clusterA.images.forEach(imgPath => {
                    const img = document.createElement('img');
                    // Handle shared images that already have ../ prefix
                    if (imgPath.startsWith('../')) {
                        img.src = imgPath;
                    } else {
                        img.src = `../images/${imgPath}`;
                    }
                    img.alt = 'Cluster A Image';
                    clusterAContainer.appendChild(img);
                });
            }
            
            // Add cluster B images
            if (clusterB && clusterB.images) {
                clusterB.images.forEach(imgPath => {
                    const img = document.createElement('img');
                    // Handle shared images that already have ../ prefix
                    if (imgPath.startsWith('../')) {
                        img.src = imgPath;
                    } else {
                        img.src = `../images/${imgPath}`;
                    }
                    img.alt = 'Cluster B Image';
                    clusterBContainer.appendChild(img);
                });
            }
            
            // Add click listeners to clusters
            document.getElementById('cluster-a').addEventListener('click', handleClusterClick);
            document.getElementById('cluster-b').addEventListener('click', handleClusterClick);
        }
        
        // Handle key responses
        function handleKeyPress(e) {
            if (e.key === '1' || e.key === '2') {
                handleTrialResponse(e.key);
            }
        }
        
        // Handle click responses
        function handleClusterClick(e) {
            const cluster = e.currentTarget;
            const selectedGroup = cluster.getAttribute('data-group');
            
            // Visual feedback
            cluster.classList.add('selected');
            setTimeout(() => {
                cluster.classList.remove('selected');
            }, 200);
            
            handleTrialResponse(selectedGroup);
        }
        
        // Common response handling function
        function handleTrialResponse(selectedGroup) {
            const trial = allTrials[currentTrialIndex];
            const selectedLabel = selectedGroup === '1' ? 'A' : 'B';
            const selectedCluster = trial.clusters.find(c => c.label === selectedLabel);
            const isCorrect = selectedCluster.is_correct;
            
            // Calculate response time for this trial only
            const trialResponseTime = Date.now() - trialStartTime;
            
            // Record response
            responses.push({
                trial_id: trial.trial_id,
                trial_num: trial.trial_num,
                time: trial.time || null,
                analysis: trial.analysis || null,
                is_catch_trial: trial.is_catch_trial || false,
                target_idx: trial.target_idx || null,  // NEW: Target image index
                target_image_path: trial.target.image_path || null,  // NEW: Target image path
                selected_group: selectedGroup,
                selected_label: selectedLabel,
                correct_answer: trial.correct_answer,
                is_correct: isCorrect,
                response_time_ms: trialResponseTime,
                timestamp: new Date().toISOString()
            });
            
            // Move to next trial
            currentTrialIndex++;
            if (currentTrialIndex < allTrials.length) {
                showCurrentTrial();
            } else {
                endExperiment();
            }
        }
        
        // End experiment
        function endExperiment() {
            document.removeEventListener('keydown', handleKeyPress);
            trialScreen.style.display = 'none';
            completionScreen.style.display = 'block';
            
            // Save results
            const resultsData = {
                prolific_pid: new URLSearchParams(window.location.search).get("PROLIFIC_PID") || 'test_participant',
                participant_number: window.participantNumber || -1,  // Sequential ID (0, 1, 2, ...)
                subject_id: window.subjectId || 'unknown',  // NEW: Subject folder ID (00, 01, 02, ...)
                responses: responses,
                total_trials: allTrials.length,
                completed_trials: responses.length,
                completion_time: new Date().toISOString(),
                user_agent: navigator.userAgent,
                experiment_start_time: experimentStartTime
            };
            
            console.log('Saving results:', resultsData);
            
            // Try PHP save first
            fetch('../php/save_results.php', {
                method: 'POST',
                headers: { 
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify(resultsData)
            })
            .then(response => {
                console.log('Response status:', response.status);
                if (response.ok) {
                    return response.json();
                } else {
                    throw new Error(`HTTP ${response.status}: PHP server not available`);
                }
            })
            .then(data => {
                console.log('✓ Results saved successfully to:', data.file);
                
                // Mark participant as completed
                const prolificId = new URLSearchParams(window.location.search).get("PROLIFIC_PID") || 'test_participant';
                fetch('../php/update_status.php', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                    body: `prolific_pid=${prolificId}&status=completed`
                }).catch(error => console.error('Failed to update status:', error));
            })
            .catch(error => {
                console.error('PHP save failed:', error.message);
                
                // Fallback: Download as JSON file
                const blob = new Blob([JSON.stringify(resultsData, null, 2)], {
                    type: 'application/json'
                });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `experiment_results_${resultsData.subject_id}_${Date.now()}.json`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
                
                console.log('✓ Results downloaded as JSON file');
                
                // Also save to localStorage as backup
                try {
                    localStorage.setItem('experiment_backup_' + Date.now(), JSON.stringify(resultsData));
                    console.log('✓ Results also saved to localStorage as backup');
                } catch (e) {
                    console.error('Could not save backup:', e);
                }
            });
        }
        
        // Start experiment
        async function startExperiment() {
            welcomeScreen.style.display = 'none';
            loadingScreen.style.display = 'block';
            
            const loaded = await loadTrials();
            if (loaded) {
                loadingScreen.style.display = 'none';
                exampleScreen.style.display = 'block';
                showExampleTrial();
            }
        }
        
        // Continue to main trials after example
        function continueToMainTrials() {
            postExampleScreen.style.display = 'none';
            trialScreen.style.display = 'block';
            showCurrentTrial();
            document.addEventListener('keydown', handleKeyPress);
        }
        
        // Initialize
        document.addEventListener('DOMContentLoaded', () => {
            // Wait for spacebar or click to start
            document.addEventListener('keydown', function startOnSpace(e) {
                if (e.code === 'Space' || e.key === ' ') {
                    document.removeEventListener('keydown', startOnSpace);
                    startExperiment();
                }
            });
            
            // Add click listener to start button
            document.querySelector('.clickable-start').addEventListener('click', function() {
                startExperiment();
            });
            
            // Add listener for post-example continue
            postExampleScreen.addEventListener('keydown', function(e) {
                if (e.code === 'Space' || e.key === ' ') {
                    continueToMainTrials();
                }
            });
            
            postExampleScreen.querySelector('.clickable-start').addEventListener('click', function() {
                continueToMainTrials();
            });
        });
        
        window.addEventListener('beforeunload', function(e) {
            if (currentTrialIndex > 0 && currentTrialIndex < allTrials.length) {
                // Participant started but didn't finish
                const prolificId = new URLSearchParams(window.location.search).get("PROLIFIC_PID") || 'test_participant';
                
                // Use sendBeacon for reliability
                navigator.sendBeacon('../php/update_status.php', 
                    new URLSearchParams({
                        prolific_pid: prolificId,
                        status: 'canceled'
                    })
                );
            }
        });
    </script>
</body>
</html>'''

    with open(main_html, 'w') as f:
            f.write(html_content)

    return main_html

def convert_numpy_types(obj):
    """Convert numpy types to regular Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    else:
        return obj

def create_php_files(php_dir):
    """Create PHP files for data handling."""
    # Create save_results.php
    save_results_path = os.path.join(php_dir, "save_results.php")
    with open(save_results_path, 'w') as f:
        f.write('''<?php
// Enable error reporting for debugging
error_reporting(E_ALL);
ini_set('display_errors', 1);

// Set headers to allow CORS and JSON content type
header("Access-Control-Allow-Origin: *");
header("Access-Control-Allow-Methods: POST, OPTIONS");
header("Access-Control-Allow-Headers: Content-Type, Accept");
header("Content-Type: application/json");

// Handle preflight OPTIONS request
if ($_SERVER['REQUEST_METHOD'] === 'OPTIONS') {
    http_response_code(200);
    exit();
}

// Only allow POST requests
if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    http_response_code(405);
    echo json_encode(["error" => "Method not allowed"]);
    exit();
}

$log_dir = "../data/responses/";

// Create log directory if it doesn't exist
if (!file_exists($log_dir)) {
    if (!mkdir($log_dir, 0777, true)) {
        http_response_code(500);
        echo json_encode(["error" => "Failed to create responses directory"]);
        exit();
    }
}

// Check if directory is writable
if (!is_writable($log_dir)) {
    http_response_code(500);
    echo json_encode(["error" => "Responses directory is not writable"]);
    exit();
}

// Get the JSON data from the request
$json_data = file_get_contents('php://input');

if (empty($json_data)) {
    http_response_code(400);
    echo json_encode(["error" => "No data received"]);
    exit();
}

$data = json_decode($json_data, true);

if (json_last_error() !== JSON_ERROR_NONE) {
    http_response_code(400);
    echo json_encode(["error" => "Invalid JSON: " . json_last_error_msg()]);
    exit();
}

if ($data) {
    // Extract metadata
    $subject_id = isset($data['subject_id']) ? $data['subject_id'] : 'anonymous';
    $completed_trials = isset($data['completed_trials']) ? $data['completed_trials'] : 0;
    
    // Sanitize the filename
    $subject_id_safe = preg_replace('/[^a-zA-Z0-9_-]/', '', $subject_id);
    
    // Create a filename with timestamp
    $timestamp = date('Y-m-d_H-i-s');
    $filename = "{$log_dir}{$subject_id_safe}_{$timestamp}_trials{$completed_trials}.json";
    
    // Save to file
    $bytes_written = file_put_contents($filename, $json_data);
    
    if ($bytes_written !== false) {
        // Success response
        $response = [
            'status' => 'success',
            'message' => 'Data saved successfully',
            'file' => basename($filename),
            'bytes_written' => $bytes_written,
            'timestamp' => $timestamp
        ];
        echo json_encode($response);
    } else {
        // Error writing to file
        http_response_code(500);
        $response = [
            'status' => 'error',
            'message' => 'Failed to write data to file: ' . $filename,
            'php_error' => error_get_last()
        ];
        echo json_encode($response);
    }
} else {
    // Invalid data received
    http_response_code(400);
    $response = [
        'status' => 'error',
        'message' => 'Failed to decode JSON data'
    ];
    echo json_encode($response);
}
?>''')
    
    # Create assign_subject.php
    assign_subject_path = os.path.join(php_dir, "assign_subject.php")
    with open(assign_subject_path, 'w') as f:
        f.write('''<?php
// Set headers
header("Access-Control-Allow-Origin: *");
header("Access-Control-Allow-Methods: GET, POST");
header("Access-Control-Allow-Headers: Content-Type");
header("Content-Type: application/json");

// Parameters
$tracker_file = "../data/subject_assignment_tracker.json";
$lock_file = "../data/tracker.lock";
$timeout = 30;

// Lock functions (same as before)
function acquire_lock($lock_file, $timeout) {
    $start_time = time();
    while (file_exists($lock_file)) {
        if (time() - filemtime($lock_file) > $timeout) {
            unlink($lock_file);
            break;
        }
        usleep(100000);
        if (time() - $start_time > $timeout) {
            return false;
        }
    }
    file_put_contents($lock_file, "1");
    return true;
}

function release_lock($lock_file) {
    if (file_exists($lock_file)) {
        unlink($lock_file);
    }
}

// Get Prolific ID
$prolific_pid = isset($_GET['prolific_pid']) ? $_GET['prolific_pid'] : '';
if (empty($prolific_pid)) {
    http_response_code(400);
    echo json_encode(["error" => "Missing prolific_pid parameter"]);
    exit;
}

$prolific_pid = preg_replace('/[^a-zA-Z0-9_-]/', '', $prolific_pid);

// Acquire lock
if (!acquire_lock($lock_file, $timeout)) {
    http_response_code(503);
    echo json_encode(["error" => "Service currently unavailable"]);
    exit;
}

try {
    if (!file_exists($tracker_file)) {
        http_response_code(500);
        echo json_encode(["error" => "Tracker file not found"]);
        release_lock($lock_file);
        exit;
    }
    
    $tracker_data = json_decode(file_get_contents($tracker_file), true);
    
    if (isset($tracker_data['participant_ids'][$prolific_pid])) {
        // Existing participant
        $participant_number = $tracker_data['participant_ids'][$prolific_pid];
        $subject_folder = $tracker_data['subject_assignments'][$participant_number];
        
        echo json_encode([
            "participant_number" => $participant_number,
            "subject_folder" => $subject_folder,
            "status" => "existing"
        ]);
    } else {
        // New participant - check for available recycled IDs first
        if (!empty($tracker_data['available_ids'])) {
            // Reuse a canceled ID
            $participant_number = min($tracker_data['available_ids']);
            $tracker_data['available_ids'] = array_values(array_diff($tracker_data['available_ids'], [$participant_number]));
        } else {
            // Assign new sequential ID
            $participant_number = $tracker_data['next_participant_id'];
            $tracker_data['next_participant_id'] = $participant_number + 1;
        }
        
        // Assign to subject folder
        $subject_id = $participant_number;
        $subject_folder = sprintf("subject_%02d", $subject_id);
        
        // Record assignments with status
        $tracker_data['participant_ids'][$prolific_pid] = $participant_number;
        $tracker_data['participant_status'][$prolific_pid] = 'started';  // NEW
        $tracker_data['start_times'][$prolific_pid] = date('Y-m-d H:i:s');  // NEW
        $tracker_data['subject_assignments'][$participant_number] = $subject_folder;
        $tracker_data['last_update'] = date('Y-m-d H:i:s');
        
        file_put_contents($tracker_file, json_encode($tracker_data, JSON_PRETTY_PRINT));
        
        echo json_encode([
            "participant_number" => $participant_number,
            "subject_folder" => $subject_folder,
            "status" => "new"
        ]);
    }
} catch (Exception $e) {
    http_response_code(500);
    echo json_encode(["error" => "Internal server error: " . $e->getMessage()]);
} finally {
    release_lock($lock_file);
}
?>''')
    
    update_status_path = os.path.join(php_dir, "update_status.php")
    with open(update_status_path, 'w') as f:
        f.write('''<?php
// Enable error reporting for debugging
error_reporting(E_ALL);
ini_set('display_errors', 1);

// Set headers to allow CORS and JSON content type
header("Access-Control-Allow-Origin: *");
header("Access-Control-Allow-Methods: POST, OPTIONS");
header("Access-Control-Allow-Headers: Content-Type, Accept");
header("Content-Type: application/json");

// Get parameters
$prolific_pid = isset($_POST['prolific_pid']) ? $_POST['prolific_pid'] : '';
$status = isset($_POST['status']) ? $_POST['status'] : '';

if (empty($prolific_pid) || empty($status)) {
    http_response_code(400);
    echo json_encode(["error" => "Missing parameters"]);
    exit;
}

// Validate status
if (!in_array($status, ['completed', 'canceled'])) {
    http_response_code(400);
    echo json_encode(["error" => "Invalid status"]);
    exit;
}

// Acquire lock and update
if (!acquire_lock($lock_file, $timeout)) {
    http_response_code(503);
    echo json_encode(["error" => "Service currently unavailable"]);
    exit;
}

try {
    $tracker_data = json_decode(file_get_contents($tracker_file), true);
    
    if (isset($tracker_data['participant_ids'][$prolific_pid])) {
        $tracker_data['participant_status'][$prolific_pid] = $status;
        
        if ($status === 'completed') {
            $tracker_data['completion_times'][$prolific_pid] = date('Y-m-d H:i:s');
        } elseif ($status === 'canceled') {
            // Add ID back to available pool
            $participant_id = $tracker_data['participant_ids'][$prolific_pid];
            $tracker_data['available_ids'][] = $participant_id;
            sort($tracker_data['available_ids']);  // Keep sorted
        }
        
        $tracker_data['last_update'] = date('Y-m-d H:i:s');
        file_put_contents($tracker_file, json_encode($tracker_data, JSON_PRETTY_PRINT));
        
        echo json_encode(["status" => "success", "participant_status" => $status]);
    } else {
        http_response_code(404);
        echo json_encode(["error" => "Participant not found"]);
    }
} catch (Exception $e) {
    http_response_code(500);
    echo json_encode(["error" => "Internal server error"]);
} finally {
    release_lock($lock_file);
}
?>''')
    
    return save_results_path, assign_subject_path

def generate_experiment(
    analysis: str,
    dataset: str,
    load_dataset_kwargs: dict,
    scorer_kwargs: dict,
    time: float,
    distance_fn: callable,
    subject_id: int,
    output_dir: str = "image_matching_experiment",
    residual: bool = False,
    n_within_cluster: int = 5,
    threshold_percentile: int = 80,
    seed: int = None,
    image_size: tuple = (112, 112),
    max_trials: int = 100,
    start_trial: int = 0,
):
    """
    Main function to generate the target-cluster matching experiment.
    """
    logging.info(f"Generating target-cluster experiment...")
    logging.info(f"Analysis: {analysis}, Dataset: {dataset}, Time: {time}")
    logging.info(f"Max trials: {max_trials}, Cluster size: {n_within_cluster}")
    
    # Set up folders
    directories = setup_experiment_folders(output_dir, subject_id)
    
    # Generate example trial
    example_trial = generate_example_trial(
        dataset=dataset,
        output_images_dir=directories['images'],
        n_within_cluster=n_within_cluster,
        seed=seed,
        image_size=image_size
    )
    
    # Generate trials
    trials = generate_image_matching_trials(
        analysis=analysis,
        dataset=dataset,
        load_dataset_kwargs=load_dataset_kwargs,
        scorer_kwargs=scorer_kwargs,
        time=time,
        distance_fn=distance_fn,
        output_images_dir=directories['images'],
        residual=residual,
        n_within_cluster=n_within_cluster,
        threshold_percentile=threshold_percentile,
        seed=seed,
        image_size=image_size,
        max_trials=max_trials,
        start_trial=start_trial,
    )
    
    # Save trial data with example trial
    all_trials_data = {
        'example_trial': convert_numpy_types(example_trial),
        'main_trials': convert_numpy_types(trials)
    }
    trials_path = os.path.join(directories['data'], "all_trials.json")
    with open(trials_path, 'w') as f:
        json.dump(all_trials_data, f, indent=2)
    
    # Create HTML interface
    _ = create_html_experiment(directories['html'], max_trials)

    # Create PHP backend
    create_php_files(directories['php'])
    
    # Create index.html redirect
    index_path = os.path.join(output_dir, "index.html")
    with open(index_path, 'w') as f:
        f.write('''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Redirecting to Experiment...</title>
    <meta http-equiv="refresh" content="0; url=html/experiment.html">
    <style>
        body {
            font-family: Arial, sans-serif;
            text-align: center;
            margin-top: 50px;
            background-color: #f5f5f5;
        }
        .redirect-message {
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
            max-width: 400px;
            margin: 0 auto;
        }
    </style>
</head>
<body>
    <div class="redirect-message">
        <h2>Loading Experiment...</h2>
        <p>If you are not automatically redirected, <a href="html/experiment.html">click here</a>.</p>
    </div>
    
    <script>
        setTimeout(function() {
            window.location.href = 'html/experiment.html';
        }, 1000);
    </script>
</body>
</html>''')
    
    logging.info(f"\nExperiment generation complete!")
    logging.info(f"Trials data: {os.path.abspath(trials_path)}")

def main():
    parser = argparse.ArgumentParser(description="Generate target-cluster matching experiment")
    
    parser.add_argument("--analysis", type=str, default="behavior")
    parser.add_argument("--dataset", type=str, default="things_eeg_2")
    parser.add_argument("--time", type=float, default=.2)
    parser.add_argument("--model", type=str, default="linear")
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory name')
    parser.add_argument('--max_trials', type=int, default=100,
                       help='Maximum number of trials to generate')
    parser.add_argument('--cluster_size', type=int, default=5,
                       help='Number of images per cluster')
    parser.add_argument("--pca", dest="pca", action="store_false")
    parser.add_argument('--percentile', type=int, default=80,
                       help='Percentile for random sampling')
    parser.add_argument('--subject_id', type=int, default=0,
                       help='Participant ID for subject assignment')
    parser.add_argument('--num_subjects', type=int, default=1,
                       help='Total number of subjects for the experiment')
    parser.set_defaults(residual=True)
    parser.add_argument('--image_width', type=int, default=112,
                       help='Width to resize images to')
    parser.add_argument('--image_height', type=int, default=112,
                       help='Height to resize images to')
    parser.add_argument('--exp_id', type=str, default=None,
                       help='Experiment ID')

    args = parser.parse_args()
    
    assert args.exp_id is not None, "Experiment ID must be specified"
    if args.output is not None:
        output_dir = args.output
    else:
        output_dir = BONNER_CACHING_HOME / "behavior" / "image_matching_experiment"
    output_dir = f"{output_dir}/{args.exp_id}"

    # You'll need to define these based on your specific setup
    load_dataset_kwargs = {} 
    scorer_kwargs={
        ### fixed for now ###
        "model_name": args.model,
        #####################
    }
    
    match args.model:
        case "lda":
            scorer_kwargs["shrinkage"] = 1e-2
        case "linear":
            scorer_kwargs["l2_penalty"] = 1e-2
        case "ridge":
            pass
        case _:
            pass
        
    if args.num_subjects == 1:
        assert args.subject_id is not None, "If num_subjects is not specified, subject_id must be provided"
        subjects = [args.subject_id]
    else:
        subjects = list(range(args.subject_id, args.subject_id + args.num_subjects))
        
    for subject_id in subjects:
        # Use provided seed or default SEED
        experiment_seed = subject_id // 2 * 1000

        start_trial = subject_id % 2 * args.max_trials

        generate_experiment(
            analysis=args.analysis,
            dataset=args.dataset,
            load_dataset_kwargs=load_dataset_kwargs,
            scorer_kwargs=scorer_kwargs,
            time=args.time,
            distance_fn=pdistance,
            output_dir=output_dir,
            residual=args.residual,
            n_within_cluster=args.cluster_size,
            threshold_percentile=args.percentile,
            seed=experiment_seed,
            image_size=(args.image_width, args.image_height),
            max_trials=args.max_trials,
            start_trial=start_trial,
            subject_id=subject_id
        )
    
    # Create the tracker file in the main data directory
    main_data_dir = os.path.join(output_dir, "data")
    create_set_assignment_tracker(main_data_dir, args.num_subjects, subject_id=args.subject_id)

if __name__ == "__main__":
    main()