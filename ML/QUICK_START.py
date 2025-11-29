"""
Quick Start Guide for PISA Math Score Prediction

Run this script to see the complete workflow
"""

print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                   PISA MATH SCORE PREDICTION                               ║
║                        Quick Start Guide                                   ║
╚════════════════════════════════════════════════════════════════════════════╝

📋 WORKFLOW:

1️⃣  PREPROCESS DATA (ONE TIME ONLY)
   Run this FIRST to preprocess your data:
   
   cd ML
   python preprocessing.py
   
   ⏱️  Takes: 5-10 minutes
   📁 Creates: data/X_train_processed.csv
   
   ⚠️  This step is REQUIRED before training any model!

2️⃣  CONFIGURE TRAINING (Optional)
   Edit ML/config.py to control:
   - Sample size (for quick testing)
   - Hyperparameter tuning iterations
   - Cross-validation folds
   
   Quick presets:
   - set_quick_test_mode()     # 1-2 min per model
   - set_medium_mode()         # Balanced
   - set_full_training_mode()  # Best accuracy

3️⃣  TRAIN MODELS
   Option A - Train one model:
     python XGBoost.py
     python Random_forest.py
     python advanced_models.py
   
   Option B - Train all models:
     python master_training.py
   
   ⏱️  Quick mode: 5-10 min total
   ⏱️  Full mode: 2-4 hours

4️⃣  CHECK OVERFITTING (Optional)
   Visualize train vs validation performance:
   
   python check_overfitting.py
   
   📊 Creates plots in results/

5️⃣  GENERATE PREDICTIONS
   Once you have a trained model:
   
   python predict.py ../data/X_test.csv
   
   📁 Creates: results/predictions.csv

╔════════════════════════════════════════════════════════════════════════════╗
║                           IMPORTANT NOTES                                  ║
╚════════════════════════════════════════════════════════════════════════════╝

✅ Always run preprocessing.py FIRST
✅ Preprocessing is done ONCE, then reused by all models
✅ Use config.py to control training speed vs accuracy
✅ Check results/overfitting_summary.csv for model comparison

💡 TIPS:
• Start with quick_test_mode to verify everything works
• Use medium_mode for development
• Use full_training_mode for final submission
• Ensemble models usually give best accuracy

📚 For more details, see README.md

Good luck with your hackathon! 🚀
""")

