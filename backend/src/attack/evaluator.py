"""
Evaluation metrics for adversarial attacks.
Measures attack success rates and other metrics.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict


class AttackEvaluator:
    """Evaluate adversarial attack performance"""
    
    def __init__(self):
        """Initialize evaluator"""
        self.results = defaultdict(list)
    
    def evaluate_targeted_attack(
        self,
        predictions: List[str],
        target: str,
        confidences: List[float]
    ) -> Dict:
        """
        Evaluate targeted attack results.
        
        Args:
            predictions: List of predicted identities
            target: Target identity to impersonate
            confidences: List of confidence scores
            
        Returns:
            Dictionary with evaluation metrics
        """
        total = len(predictions)
        successes = sum(1 for pred in predictions if pred == target)
        
        success_rate = (successes / total * 100) if total > 0 else 0
        avg_confidence = np.mean(confidences) if confidences else 0
        
        # Calculate confidence for successful attacks
        success_confidences = [
            conf for pred, conf in zip(predictions, confidences)
            if pred == target
        ]
        avg_success_confidence = (
            np.mean(success_confidences) if success_confidences else 0
        )
        
        return {
            'success_rate': success_rate,
            'total_tests': total,
            'successes': successes,
            'failures': total - successes,
            'avg_confidence': avg_confidence,
            'avg_success_confidence': avg_success_confidence,
            'target': target
        }
    
    def evaluate_untargeted_attack(
        self,
        predictions: List[str],
        true_identity: str,
        confidences: List[float]
    ) -> Dict:
        """
        Evaluate untargeted attack (goal: not be recognized as self).
        
        Args:
            predictions: List of predicted identities
            true_identity: Actual identity
            confidences: Confidence scores
            
        Returns:
            Evaluation metrics
        """
        total = len(predictions)
        successes = sum(1 for pred in predictions if pred != true_identity)
        
        success_rate = (successes / total * 100) if total > 0 else 0
        
        return {
            'success_rate': success_rate,
            'total_tests': total,
            'successes': successes,
            'recognized_as_self': total - successes,
            'avg_confidence': np.mean(confidences) if confidences else 0
        }
    
    def evaluate_baseline(
        self,
        predictions: List[str],
        expected_identity: str,
        confidences: List[float]
    ) -> Dict:
        """
        Evaluate baseline (no attack) performance.
        
        Args:
            predictions: Predicted identities
            expected_identity: Expected identity ("Unknown" for attackers)
            confidences: Confidence scores
            
        Returns:
            Baseline metrics
        """
        total = len(predictions)
        correct = sum(1 for pred in predictions if pred == expected_identity)
        
        accuracy = (correct / total * 100) if total > 0 else 0
        
        return {
            'accuracy': accuracy,
            'total_tests': total,
            'correct': correct,
            'incorrect': total - correct,
            'avg_confidence': np.mean(confidences) if confidences else 0,
            'expected': expected_identity
        }
    
    def compare_attack_vs_baseline(
        self,
        baseline_results: Dict,
        attack_results: Dict
    ) -> Dict:
        """
        Compare attack performance against baseline.
        
        Args:
            baseline_results: Results without attack
            attack_results: Results with attack
            
        Returns:
            Comparison metrics
        """
        baseline_accuracy = baseline_results['accuracy']
        attack_success = attack_results['success_rate']
        
        effectiveness = attack_success - (100 - baseline_accuracy)
        
        return {
            'baseline_accuracy': baseline_accuracy,
            'attack_success_rate': attack_success,
            'effectiveness': effectiveness,
            'confidence_increase': (
                attack_results['avg_success_confidence'] -
                baseline_results['avg_confidence']
            )
        }
    
    def calculate_confusion_matrix(
        self,
        predictions: List[str],
        ground_truths: List[str],
        classes: List[str]
    ) -> np.ndarray:
        """
        Calculate confusion matrix.
        
        Args:
            predictions: Predicted classes
            ground_truths: True classes
            classes: List of all classes
            
        Returns:
            Confusion matrix [num_classes, num_classes]
        """
        n_classes = len(classes)
        class_to_idx = {c: i for i, c in enumerate(classes)}
        
        confusion = np.zeros((n_classes, n_classes), dtype=int)
        
        for pred, true in zip(predictions, ground_truths):
            pred_idx = class_to_idx.get(pred, -1)
            true_idx = class_to_idx.get(true, -1)
            
            if pred_idx >= 0 and true_idx >= 0:
                confusion[true_idx, pred_idx] += 1
        
        return confusion
    
    def calculate_transferability(
        self,
        source_model_results: Dict,
        target_model_results: Dict
    ) -> Dict:
        """
        Calculate attack transferability between models.
        
        Args:
            source_model_results: Results on source model
            target_model_results: Results on target model
            
        Returns:
            Transferability metrics
        """
        source_success = source_model_results['success_rate']
        target_success = target_model_results['success_rate']
        
        transferability = (target_success / source_success * 100) if source_success > 0 else 0
        
        return {
            'source_success_rate': source_success,
            'target_success_rate': target_success,
            'transferability': transferability,
            'success_drop': source_success - target_success
        }
    
    def generate_report(
        self,
        baseline: Dict,
        attack: Dict,
        patch_info: Dict
    ) -> str:
        """
        Generate human-readable evaluation report.
        
        Args:
            baseline: Baseline results
            attack: Attack results
            patch_info: Information about the patch
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("="*70)
        report.append("ADVERSARIAL ATTACK EVALUATION REPORT")
        report.append("="*70)
        
        report.append("\nPATCH INFORMATION:")
        report.append(f"  Type: {patch_info.get('type', 'N/A')}")
        report.append(f"  Size: {patch_info.get('size', 'N/A')} pixels")
        report.append(f"  Target: {patch_info.get('target', 'N/A')}")
        
        report.append("\nBASELINE (No Attack):")
        report.append(f"  Accuracy: {baseline['accuracy']:.1f}%")
        report.append(f"  Correct: {baseline['correct']}/{baseline['total_tests']}")
        report.append(f"  Avg Confidence: {baseline['avg_confidence']:.3f}")
        
        report.append("\nATTACK RESULTS:")
        report.append(f"  Success Rate: {attack['success_rate']:.1f}%")
        report.append(f"  Successes: {attack['successes']}/{attack['total_tests']}")
        report.append(f"  Avg Confidence: {attack['avg_success_confidence']:.3f}")
        
        comparison = self.compare_attack_vs_baseline(baseline, attack)
        
        report.append("\nEFFECTIVENESS:")
        report.append(f"  Attack Effectiveness: {comparison['effectiveness']:.1f}%")
        report.append(f"  Confidence Increase: {comparison['confidence_increase']:.3f}")
        
        if attack['success_rate'] > 50:
            report.append("\n✓ Attack is EFFECTIVE!")
        else:
            report.append("\n⚠ Attack needs improvement")
        
        report.append("="*70)
        
        return "\n".join(report)
    
    def save_results(self, filepath: str):
        """
        Save evaluation results to file.
        
        Args:
            filepath: Path to save results
        """
        import json
        
        with open(filepath, 'w') as f:
            json.dump(dict(self.results), f, indent=2)
        
        print(f"✓ Saved evaluation results to {filepath}")