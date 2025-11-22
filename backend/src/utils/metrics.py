"""
Metrics and evaluation utilities.
"""

import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

def calculate_attack_success_rate(
    predictions: List[str],
    target: str
) -> float:
    """
    Calculate attack success rate (targeted).
    
    Args:
        predictions: List of predicted identities
        target: Target identity
        
    Returns:
        Success rate as percentage
    """
    if not predictions:
        return 0.0
    
    successes = sum(1 for pred in predictions if pred == target)
    return (successes / len(predictions)) * 100


def calculate_evasion_rate(
    predictions: List[str],
    true_identity: str
) -> float:
    """
    Calculate evasion rate (untargeted attack).
    
    Args:
        predictions: List of predicted identities
        true_identity: True identity to evade
        
    Returns:
        Evasion rate as percentage
    """
    if not predictions:
        return 0.0
    
    evasions = sum(1 for pred in predictions if pred != true_identity)
    return (evasions / len(predictions)) * 100


def calculate_precision_recall(
    predictions: List[str],
    ground_truth: List[str],
    positive_class: str
) -> Tuple[float, float, float]:
    """
    Calculate precision, recall, and F1 score.
    
    Args:
        predictions: Predicted labels
        ground_truth: True labels
        positive_class: Label to treat as positive
        
    Returns:
        Tuple of (precision, recall, f1_score)
    """
    # Convert to binary
    y_pred = [1 if p == positive_class else 0 for p in predictions]
    y_true = [1 if t == positive_class else 0 for t in ground_truth]
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    
    return precision, recall, f1


def calculate_false_acceptance_rate(
    predictions: List[str],
    ground_truth: List[str],
    impostor_label: str = "Unknown"
) -> float:
    """
    Calculate False Acceptance Rate (FAR).
    Rate at which impostors are incorrectly accepted.
    
    Args:
        predictions: Predicted labels
        ground_truth: True labels
        impostor_label: Label indicating impostor
        
    Returns:
        FAR as percentage
    """
    # Count impostor attempts
    impostor_mask = [t == impostor_label for t in ground_truth]
    
    if not any(impostor_mask):
        return 0.0
    
    # Count false accepts (impostor predicted as legitimate)
    false_accepts = sum(
        1 for pred, gt, is_imp in zip(predictions, ground_truth, impostor_mask)
        if is_imp and pred != impostor_label
    )
    
    total_impostors = sum(impostor_mask)
    return (false_accepts / total_impostors) * 100


def calculate_false_rejection_rate(
    predictions: List[str],
    ground_truth: List[str],
    impostor_label: str = "Unknown"
) -> float:
    """
    Calculate False Rejection Rate (FRR).
    Rate at which legitimate users are incorrectly rejected.
    
    Args:
        predictions: Predicted labels
        ground_truth: True labels
        impostor_label: Label indicating impostor
        
    Returns:
        FRR as percentage
    """
    # Count legitimate attempts
    legitimate_mask = [t != impostor_label for t in ground_truth]
    
    if not any(legitimate_mask):
        return 0.0
    
    # Count false rejects (legitimate user predicted as impostor)
    false_rejects = sum(
        1 for pred, gt, is_legit in zip(predictions, ground_truth, legitimate_mask)
        if is_legit and pred == impostor_label
    )
    
    total_legitimate = sum(legitimate_mask)
    return (false_rejects / total_legitimate) * 100


def calculate_equal_error_rate(
    far_values: List[float],
    frr_values: List[float],
    thresholds: List[float]
) -> Tuple[float, float]:
    """
    Calculate Equal Error Rate (EER).
    Point where FAR = FRR.
    
    Args:
        far_values: False Acceptance Rates at different thresholds
        frr_values: False Rejection Rates at different thresholds
        thresholds: Corresponding threshold values
        
    Returns:
        Tuple of (EER, threshold at EER)
    """
    # Find where FAR and FRR are closest
    differences = [abs(far - frr) for far, frr in zip(far_values, frr_values)]
    min_idx = np.argmin(differences)
    
    eer = (far_values[min_idx] + frr_values[min_idx]) / 2
    eer_threshold = thresholds[min_idx]
    
    return eer, eer_threshold


def calculate_robustness_score(
    clean_accuracy: float,
    adversarial_accuracy: float
) -> float:
    """
    Calculate robustness score.
    
    Args:
        clean_accuracy: Accuracy on clean samples
        adversarial_accuracy: Accuracy on adversarial samples
        
    Returns:
        Robustness score (0-1)
    """
    if clean_accuracy == 0:
        return 0.0
    
    return adversarial_accuracy / clean_accuracy


def calculate_attack_effectiveness(
    baseline_recognition_rate: float,
    attack_success_rate: float
) -> float:
    """
    Calculate attack effectiveness.
    
    Args:
        baseline_recognition_rate: Normal recognition rate
        attack_success_rate: Success rate with attack
        
    Returns:
        Effectiveness score
    """
    # How much the attack improved over random chance
    random_chance = 100 - baseline_recognition_rate
    improvement = attack_success_rate - random_chance
    
    return improvement


def calculate_confidence_statistics(
    confidences: List[float]
) -> Dict[str, float]:
    """
    Calculate statistics for confidence scores.
    
    Args:
        confidences: List of confidence scores
        
    Returns:
        Dictionary with statistics
    """
    if not confidences:
        return {
            'mean': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'median': 0.0
        }
    
    return {
        'mean': np.mean(confidences),
        'std': np.std(confidences),
        'min': np.min(confidences),
        'max': np.max(confidences),
        'median': np.median(confidences)
    }


def calculate_patch_visibility_score(
    original_image: np.ndarray,
    patched_image: np.ndarray
) -> float:
    """
    Calculate patch visibility/perceptibility score.
    Lower is better (less visible).
    
    Args:
        original_image: Original image [H, W, C]
        patched_image: Patched image [H, W, C]
        
    Returns:
        Visibility score (MSE)
    """
    mse = np.mean((original_image - patched_image) ** 2)
    return mse


def calculate_transferability_score(
    source_success_rate: float,
    target_success_rate: float
) -> float:
    """
    Calculate attack transferability between models.
    
    Args:
        source_success_rate: Success on source model
        target_success_rate: Success on target model
        
    Returns:
        Transferability score (0-1)
    """
    if source_success_rate == 0:
        return 0.0
    
    return target_success_rate / source_success_rate


class MetricsTracker:
    """Track metrics over multiple experiments"""
    
    def __init__(self):
        """Initialize tracker"""
        self.metrics_history = []
    
    def add_experiment(self, metrics: Dict):
        """Add experiment metrics"""
        self.metrics_history.append(metrics)
    
    def get_average_metrics(self) -> Dict:
        """Get average metrics across all experiments"""
        if not self.metrics_history:
            return {}
        
        # Get all metric keys
        keys = set()
        for metrics in self.metrics_history:
            keys.update(metrics.keys())
        
        # Calculate averages
        averages = {}
        for key in keys:
            values = [m[key] for m in self.metrics_history if key in m]
            if values:
                averages[key] = np.mean(values)
                averages[f"{key}_std"] = np.std(values)
        
        return averages
    
    def get_best_experiment(self, metric_key: str, maximize: bool = True) -> Dict:
        """
        Get experiment with best metric value.
        
        Args:
            metric_key: Metric to optimize
            maximize: Whether to maximize (True) or minimize (False)
            
        Returns:
            Best experiment metrics
        """
        if not self.metrics_history:
            return {}
        
        if maximize:
            best = max(self.metrics_history, key=lambda x: x.get(metric_key, -float('inf')))
        else:
            best = min(self.metrics_history, key=lambda x: x.get(metric_key, float('inf')))
        
        return best
    
    def save_to_file(self, filepath: str):
        """Save metrics history to JSON file"""
        import json
        
        with open(filepath, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        
        print(f"✓ Saved metrics to {filepath}")
    
    def load_from_file(self, filepath: str):
        """Load metrics history from JSON file"""
        import json
        
        with open(filepath, 'r') as f:
            self.metrics_history = json.load(f)
        
        print(f"✓ Loaded {len(self.metrics_history)} experiments from {filepath}")



def calculate_and_report_metrics(all_labels: List[int], all_preds: List[int], target_names: List[str]) -> Tuple[float, float, float, float]:
    """
    Calculates and prints the classification report and key metrics
    (Accuracy, Precision, Recall, F1) from the confusion matrix.

    Args:
        all_labels: List of true labels.
        all_preds: List of predicted labels.
        target_names: List of class names (e.g., ['Clean', 'Patch']).
        
    Returns:
        Tuple of (accuracy, precision, recall, f1) for the 'Patch' class (1).
    """
    print("\n--- Classification Report ---")
    print(classification_report(all_labels, all_preds, target_names=target_names))

    cm = confusion_matrix(all_labels, all_preds)
    
    # Check if the confusion matrix has the expected 2x2 shape
    if cm.shape != (2, 2):
        print("\nWarning: Confusion matrix is not 2x2. Cannot calculate TN, FP, FN, TP.")
        # Return zeros if unable to calculate specific binary metrics
        return 0.0, 0.0, 0.0, 0.0

    # cm.ravel() flattens the matrix: (TN, FP, FN, TP)
    tn, fp, fn, tp = cm.ravel()

    # Calculate overall accuracy
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    
    # Calculate binary metrics (Precision, Recall, F1) for the POSITIVE class (Patch, label 1)
    # Handle division by zero
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    if (precision + recall) > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    print(f'\n--- Key Metrics (Patch Class) ---')
    print(f'Accuracy: {accuracy*100:.2f}% (Overall)')
    print(f'Precision: {precision*100:.2f}%')
    print(f'Recall: {recall*100:.2f}%')
    print(f'F1-Score: {f1*100:.2f}%')
    
    return accuracy, precision, recall, f1