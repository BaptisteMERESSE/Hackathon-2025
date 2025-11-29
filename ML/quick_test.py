"""
Quick Test Script - Train a single model quickly to test the pipeline
"""
import config

# Set quick test mode
config.set_quick_test_mode()

print("\n" + "="*80)
print("QUICK TEST MODE")
print("="*80)
print("This will train a single XGBoost model on a small sample for quick testing")
print("Estimated time: 2-5 minutes")
print("="*80 + "\n")

import XGBoost

print("\n" + "="*80)
print("QUICK TEST COMPLETE!")
print("="*80)
print("If this worked well, you can:")
print("  1. Try medium mode: Edit config.py and uncomment 'set_medium_mode()'")
print("  2. Try full training: Edit config.py and uncomment 'set_full_training_mode()'")
print("  3. Run master_training.py to train all models")
print("="*80)

