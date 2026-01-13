# QUICKSTART GUIDE - Improving UAR to 70%

## 🚀 TL;DR - Run This First

```bash
cd /Users/macbook/Desktop/V1

# Run the best experiment (all improvements)
bash experiments/exp3_full_stack.sh

# Monitor training in real-time (in another terminal)
tail -f outputs/exp3_full_stack-*/log.txt | grep -E "(UAR|Class 2)"
```

Expected runtime: 2-3 hours (30-35 epochs)

---

## 📊 What Changed?

### Problem Identified
- ❌ **Wrong validation protocol**: Using test.txt for both val and test
- ❌ **Class imbalance**: Class Confusion only 1.3% of data
- ❌ **Temperature too high**: 0.5 → predictions too smooth
- ❌ **MI loss interference**: High weight causing gradient conflict

### Solution Implemented
- ✅ **Fixed validation**: train_80.txt / val_20.txt / test.txt
- ✅ **Class-balanced Focal Loss**: Automatic weights based on class frequency
- ✅ **Lower temperature**: 0.5 → 0.07 (sharper predictions)
- ✅ **Reduced MI loss**: lambda_mi 0.7 → 0.02, longer warmup
- ✅ **Enhanced logging**: Per-class recall every epoch

---

## 🎯 Running Experiments

### Option 1: Quick Test (Recommended First)
```bash
# Run baseline to verify everything works
bash experiments/exp1_proper_baseline.sh

# Check results after ~10-15 epochs
tail -30 outputs/exp1_proper_baseline-*/log.txt
```

### Option 2: Best Configuration (Target UAR >= 70%)
```bash
# Run full stack with all improvements
bash experiments/exp3_full_stack.sh

# Expected results:
#   Valid UAR: 65-70%
#   Class 2 (Confusion) Recall: 40-50%
#   Training time: 2-3 hours
```

### Option 3: Run All Experiments (Comprehensive Comparison)
```bash
# Run all experiments sequentially
for script in experiments/exp*.sh; do
    echo "Running $script..."
    bash "$script"
    echo "Completed!"
    echo ""
done

# Compare results
python experiments/compare_experiments.py
```

---

## 📈 Monitoring Training

### Real-time Monitoring
```bash
# Watch UAR and Class 2 recall
tail -f outputs/exp3_full_stack-*/log.txt | grep -E "(UAR|Class 2)"

# Watch all per-class recalls
tail -f outputs/exp3_full_stack-*/log.txt | grep -A 6 "Per-class Recall"
```

### What to Look For

**Good Signs ✅**
- Valid UAR steadily increasing (50% → 60% → 70%)
- Class 2 (Confusion) recall > 30% and increasing
- Train-Val UAR gap < 20%
- No class with recall < 30%

**Warning Signs ⚠️**
- Valid UAR not improving after epoch 10
- Class 2 recall stuck at < 20%
- Train-Val gap > 30% (overfitting)
- Training loss = NaN or Inf

---

## 📋 Understanding the Logs

### Log Format
```
Train Epoch: [15] * WAR: 82.50 | UAR: 78.30
Train Epoch: [15] Per-class Recall:
  Class 0 (Neutrality  ):  85.2% (320/376 samples)
  Class 1 (Enjoyment   ):  72.1% ( 85/118 samples)
  Class 2 (Confusion   ):  68.4% ( 26/ 38 samples)  ← KEY METRIC!
  Class 3 (Fatigue     ):  70.5% (130/184 samples)
  Class 4 (Distraction ):  78.9% (405/513 samples)

Valid Epoch: [15] * WAR: 70.20 | UAR: 68.50
Valid Epoch: [15] Per-class Recall:
  Class 0 (Neutrality  ):  74.3% (126/170 samples)
  Class 1 (Enjoyment   ):  46.9% ( 15/ 32 samples)
  Class 2 (Confusion   ):  42.9% (  3/  7 samples)  ← TARGET >= 40%!
  Class 3 (Fatigue     ):  66.7% ( 26/ 39 samples)
  Class 4 (Distraction ):  80.0% ( 84/105 samples)
```

### Key Metrics
- **UAR (Unweighted Average Recall)**: Primary metric → TARGET >= 70%
- **Class 2 Recall**: Confusion class → TARGET >= 40%
- **Train-Val Gap**: Overfitting indicator → TARGET < 20%

---

## 🔍 After Training - Check Results

### Compare Experiments
```bash
python experiments/compare_experiments.py

# Output will show:
# - Ranking by UAR
# - Train-val gap
# - Per-class recall for top experiments
```

### View Detailed Results
```bash
# Best model logs
cat outputs/exp3_full_stack-*/log.txt | grep -A 10 "Best Valid UAR"

# Confusion matrices (last 50 lines)
tail -50 outputs/exp3_full_stack-*/log.txt

# Class weights used
head -100 outputs/exp3_full_stack-*/log.txt | grep -A 10 "CLASS DISTRIBUTION"
```

---

## 🛠️ Troubleshooting

### Problem: Class 2 Recall Still Low (< 30%)

**Solution 1: Increase Focal Loss Gamma**
```bash
# Edit main.py line 204:
# Change: criterion = FocalLoss(alpha=class_weights.to(args.device), gamma=2.0)
# To:     criterion = FocalLoss(alpha=class_weights.to(args.device), gamma=3.0)
```

**Solution 2: Increase Class 2 Weight Manually**
```bash
# Edit utils/builders.py, line 50 (after computing weights):
weights[2] = weights[2] * 1.5  # Boost Class 2 weight by 50%
```

### Problem: Training Too Slow

**Solution: Reduce Batch Operations**
```bash
# In experiments/exp3_full_stack.sh, change:
--batch-size 8        # → 16 (if you have GPU memory)
--num-segments 16     # → 12 (fewer temporal segments)
--workers 4           # → 8 (more parallel loading)
```

### Problem: Out of Memory

**Solution: Reduce Memory Usage**
```bash
# In experiments/exp3_full_stack.sh, change:
--batch-size 8        # → 4
--use-amp             # (keep this, it reduces memory)
--num-segments 16     # → 12
```

### Problem: NaN Loss

**Solution: Reduce Learning Rates**
```bash
# In experiments/exp3_full_stack.sh, change:
--lr 1e-4             # → 5e-5
--lr-prompt-learner 5e-4  # → 2e-4
--grad-clip 1.0       # → 0.5 (stricter clipping)
```

---

## 📝 Code Changes Summary

### Files Modified
1. **utils/builders.py** - Added `get_class_weights_from_annotation()`
2. **utils/loss.py** - Enhanced `FocalLoss` to accept class weights
3. **trainer.py** - Added detailed per-class logging
4. **main.py** - Updated defaults (temperature, lambda_mi, warmup/ramp)

### New Files Created
1. **PLAN_IMPROVE_UAR.md** - Comprehensive action plan
2. **QUICKSTART.md** - This file
3. **experiments/exp1_proper_baseline.sh** - Baseline experiment
4. **experiments/exp2_focal_loss.sh** - Focal loss experiment
5. **experiments/exp3_full_stack.sh** - Best configuration
6. **experiments/compare_experiments.py** - Results comparison tool

---

## 🎓 Understanding Class-Balanced Focal Loss

### How It Works

1. **Count Class Frequency** (automatically from train_80.txt)
   ```
   Class 0 (Neutrality):  650 samples → weight: 0.92
   Class 1 (Enjoyment):   120 samples → weight: 2.15
   Class 2 (Confusion):    35 samples → weight: 5.87  ← High weight!
   Class 3 (Fatigue):     180 samples → weight: 1.68
   Class 4 (Distraction): 512 samples → weight: 1.05
   ```

2. **Apply Focal Loss Formula**
   ```python
   focal_loss = weight[class] * (1 - p_t)^gamma * cross_entropy_loss

   # Where:
   # - weight[class]: Higher for minority classes
   # - (1 - p_t)^gamma: Focus on hard examples
   # - gamma=2.0: Down-weight easy examples
   ```

3. **Result**: Model pays more attention to:
   - Minority classes (Class 2 gets 5.87x weight)
   - Hard examples (low probability predictions)

---

## 📞 Next Steps

### If UAR >= 70% ✅
1. Run final test evaluation (DO NOT USE TEST SET YET!)
2. Verify on held-out validation set
3. Document results
4. Consider ensemble methods for further improvement

### If UAR < 70% ⚠️
1. Check PLAN_IMPROVE_UAR.md for Phase 2 experiments
2. Try stronger augmentation for Class 2
3. Experiment with different focal loss gamma values
4. Consider threshold calibration (Phase 4)

---

## 📚 Additional Resources

- **Full Plan**: See `PLAN_IMPROVE_UAR.md` for detailed strategy
- **Log Analysis**: Use `compare_experiments.py` for automatic comparison
- **Original Log**: Check `outputs/CLIP-CAER-NEW-V1/log.txt` for baseline

---

## ✅ Success Checklist

Before considering the project complete:

- [ ] Valid UAR >= 70%
- [ ] Class 2 (Confusion) recall >= 40%
- [ ] No class with recall < 35%
- [ ] Train-Val UAR gap < 20%
- [ ] Model converges within 30 epochs
- [ ] Results are reproducible (same seed)
- [ ] Proper validation protocol used (train_80/val_20/test)
- [ ] Test set NOT touched until final evaluation

---

**Good luck! 🚀**

For questions or issues, check the troubleshooting section or review PLAN_IMPROVE_UAR.md.
