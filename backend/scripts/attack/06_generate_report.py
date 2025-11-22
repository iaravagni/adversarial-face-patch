"""
Generate comprehensive attack evaluation report.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import json
from datetime import datetime


def main():
    """Generate evaluation report"""
    
    print("="*70)
    print("GENERATING EVALUATION REPORT")
    print("="*70)
    
    from src.utils.config import load_config
    config = load_config()
    
    patches_dir = config['patches_dir']
    
    # Collect patch information
    patches = []
    patch_files = [f for f in os.listdir(patches_dir) if f.endswith('_metadata.json')]
    
    for metadata_file in patch_files:
        with open(os.path.join(patches_dir, metadata_file), 'r') as f:
            metadata = json.load(f)
            patches.append(metadata)
    
    if not patches:
        print("ERROR: No patch metadata found! Run previous scripts first.")
        return
    
    # Generate report
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("ADVERSARIAL FACE RECOGNITION - ATTACK EVALUATION REPORT")
    report_lines.append("="*80)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    report_lines.append("SYSTEM CONFIGURATION")
    report_lines.append("-" * 80)
    report_lines.append(f"Classification Threshold: {config['classification_threshold']}")
    report_lines.append(f"Patch Radius: {config['patch']['radius']} pixels")
    report_lines.append(f"Optimization Iterations: {config['patch']['optimization']['iterations']}")
    report_lines.append("")
    
    report_lines.append("PATCH EVALUATION RESULTS")
    report_lines.append("-" * 80)
    
    total_success = 0
    
    for i, patch in enumerate(patches, 1):
        report_lines.append(f"\nPatch {i}: {patch.get('name', 'Unknown')}")
        report_lines.append(f"  Type: {patch.get('type', 'N/A')}")
        report_lines.append(f"  Size: {patch.get('size', 'N/A')} pixels")
        report_lines.append(f"  Target Employee: {patch.get('target_employee', 'N/A')}")
        report_lines.append(f"  Position: ({patch['position']['x']:.1f}, {patch['position']['y']:.1f})")
        report_lines.append(f"  Success Rate: {patch.get('success_rate', 0):.1f}%")
        
        total_success += patch.get('success_rate', 0)
    
    avg_success = total_success / len(patches) if patches else 0
    
    report_lines.append("")
    report_lines.append("SUMMARY STATISTICS")
    report_lines.append("-" * 80)
    report_lines.append(f"Total Patches Tested: {len(patches)}")
    report_lines.append(f"Average Success Rate: {avg_success:.1f}%")
    report_lines.append(f"Best Patch: {max(patches, key=lambda x: x.get('success_rate', 0)).get('name', 'N/A')}")
    report_lines.append(f"  Success Rate: {max(p.get('success_rate', 0) for p in patches):.1f}%")
    
    report_lines.append("")
    report_lines.append("EFFECTIVENESS RATING")
    report_lines.append("-" * 80)
    
    if avg_success >= 80:
        rating = "HIGHLY EFFECTIVE"
        symbol = "✓✓✓"
    elif avg_success >= 60:
        rating = "EFFECTIVE"
        symbol = "✓✓"
    elif avg_success >= 40:
        rating = "MODERATELY EFFECTIVE"
        symbol = "✓"
    else:
        rating = "NEEDS IMPROVEMENT"
        symbol = "⚠"
    
    report_lines.append(f"{symbol} {rating}")
    report_lines.append("")
    
    report_lines.append("SECURITY IMPLICATIONS")
    report_lines.append("-" * 80)
    report_lines.append("The adversarial patches demonstrate significant vulnerabilities in")
    report_lines.append("face recognition systems. Key findings:")
    report_lines.append("")
    report_lines.append("1. Physical Feasibility: Circular patches can be printed as stickers")
    report_lines.append("2. Transferability: May work across different FR systems")
    report_lines.append("3. Ease of Use: Simple forehead placement, no special skills needed")
    report_lines.append("4. Detection Difficulty: Small patches are hard to detect visually")
    report_lines.append("")
    
    report_lines.append("RECOMMENDED DEFENSES")
    report_lines.append("-" * 80)
    report_lines.append("1. Adversarial Training: Train models with adversarial examples")
    report_lines.append("2. Input Preprocessing: JPEG compression, blur, median filtering")
    report_lines.append("3. Patch Detection: Use frequency analysis and texture metrics")
    report_lines.append("4. Multi-Modal Authentication: Combine face with other biometrics")
    report_lines.append("5. Liveness Detection: Ensure the face is real, not an image")
    report_lines.append("")
    
    report_lines.append("="*80)
    report_lines.append("END OF REPORT")
    report_lines.append("="*80)
    
    # Print report
    report_text = "\n".join(report_lines)
    print(report_text)
    
    # Save report
    report_path = os.path.join(patches_dir, f"attack_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(report_path, 'w') as f:
        f.write(report_text)
    
    print(f"\n✓ Report saved to: {report_path}")
    
    # Also save as JSON
    json_report = {
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'patches': patches,
        'summary': {
            'total_patches': len(patches),
            'average_success_rate': avg_success,
            'effectiveness_rating': rating
        }
    }
    
    json_path = os.path.join(patches_dir, f"attack_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(json_path, 'w') as f:
        json.dump(json_report, f, indent=2)
    
    print(f"✓ JSON report saved to: {json_path}")
    
    print("\n" + "="*70)
    print("✓ REPORT GENERATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()