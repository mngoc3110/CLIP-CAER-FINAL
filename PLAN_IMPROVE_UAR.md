# ACTION PLAN: Improve RAER UAR to >=70% (Focus on Class Confusion)

**Created**: 2026-01-13
**Target**: Valid UAR >= 70%, Class Confusion Recall >= 40%
**Current**: Valid UAR = 53.97%, Class Confusion Recall = 14.3% (1/7 samples)

---

## 🚨 CRITICAL ISSUES IDENTIFIED

### Issue 1: WRONG VALIDATION PROTOCOL ⚠️
- **Problem**: Using test.txt for BOTH validation and test (data leakage)
- **Evidence**: `log.txt` lines 10-11
- **Impact**: Invalid model selection, overfitted to test set
- **Fix**: Use train_80.txt + val_20.txt + test.txt (proper split)
- **Priority**: P0 (MUST FIX FIRST)

### Issue 2: CLASS IMBALANCE - Class Confusion Collapsed
- **Problem**: Class 2 (Confusion) only 7/528 samples in validation (1.3%)
- **Evidence**: Confusion matrix shows 1/7 correct (14.3% recall)
- **Impact**: Model ignores minority class completely
- **Fix**: Class-balanced focal loss + weighted sampling + augmentation
- **Priority**: P1 (HIGH)

### Issue 3: SEVERE OVERFITTING
- **Problem**: Train UAR 95% vs Valid UAR 54% (41% gap)
- **Evidence**: Epoch 15 metrics
- **Impact**: Model memorizes training set, no generalization
- **Fix**: Regularization + temperature tuning + MI loss reduction
- **Priority**: P1 (HIGH)

### Issue 4: TEMPERATURE TOO HIGH
- **Problem**: temperature=0.5 makes logits too smooth
- **Evidence**: Config line 50
- **Impact**: Model not confident on minority predictions
- **Fix**: Lower to 0.07 (standard CLIP temperature)
- **Priority**: P2 (MEDIUM)

---

## 📋 IMPLEMENTATION PHASES

### **PHASE 1: FOUNDATION FIXES (Day 1, 2-3 hours)** ✅ PRIORITY

#### Task 1.1: Fix Validation Protocol
- **File**: `train_debug.sh`, `main.py`
- **Change**: Update annotation paths
- **Test**: Verify 1697/424/528 split
- **Time**: 15 minutes

#### Task 1.2: Implement Class-Balanced Focal Loss
- **Files**: `utils/builders.py`, `utils/loss.py`, `main.py`
- **Change**: Compute class weights + use FocalLoss
- **Test**: Check class weights in log
- **Time**: 45 minutes

#### Task 1.3: Add Detailed Per-Class Logging
- **File**: `trainer.py`
- **Change**: Log per-class recall every epoch
- **Test**: Verify log format
- **Time**: 30 minutes

#### Task 1.4: Lower Temperature
- **File**: `main.py` (argument default)
- **Change**: temperature = 0.5 → 0.07
- **Test**: Verify in config output
- **Time**: 5 minutes

#### Task 1.5: Reduce MI Loss Interference
- **File**: `main.py` (argument defaults)
- **Change**: lambda_mi = 0.05 → 0.02, warmup/ramp increase
- **Test**: Check MI loss weight in training log
- **Time**: 10 minutes

**Deliverable**: Working baseline with proper validation

---

### **PHASE 2: ADVANCED LOSS FUNCTIONS (Day 2, 3-4 hours)**

#### Task 2.1: Balanced Softmax with Logit Adjustment
- **File**: `trainer.py`
- **Change**: Center-adjusted logit shift
- **Test**: Sweep tau=[0.3, 0.5, 0.7, 1.0]
- **Time**: 1 hour

#### Task 2.2: LDAM Loss Implementation
- **File**: `utils/loss.py`
- **Change**: Add margin-based loss for minority classes
- **Test**: Compare with focal loss
- **Time**: 1.5 hours

#### Task 2.3: Online Hard Example Mining
- **File**: `trainer.py`
- **Change**: Filter easy majority samples
- **Test**: Monitor loss distribution
- **Time**: 1 hour

**Deliverable**: Multiple loss function options

---

### **PHASE 3: DATA AUGMENTATION (Day 3, 2-3 hours)**

#### Task 3.1: Class-Conditional Strong Augmentation
- **File**: `dataloader/video_dataloader.py`
- **Change**: Stronger augmentation for class 2
- **Test**: Visualize augmented samples
- **Time**: 1.5 hours

#### Task 3.2: Temporal Mixup in Embedding Space
- **File**: `models/Generate_Model.py`
- **Change**: Mix features for minority samples
- **Test**: Check feature diversity
- **Time**: 1.5 hours

**Deliverable**: Enhanced data diversity

---

### **PHASE 4: THRESHOLD CALIBRATION (Day 4, 2 hours)**

#### Task 4.1: Per-Class Threshold Tuning
- **File**: New `calibrate_thresholds.py`
- **Change**: Grid search on validation set
- **Test**: UAR improvement measurement
- **Time**: 2 hours

**Deliverable**: Optimized decision boundaries

---

### **PHASE 5: EVALUATION & TESTING (Day 5, 1 hour)**

#### Task 5.1: Final Test Evaluation
- **File**: `eval_final.py`
- **Change**: Comprehensive test set evaluation
- **Test**: Generate report
- **Time**: 1 hour

**Deliverable**: Final results and report

---

## 🎯 EXPERIMENTS ROADMAP

### Experiment 1: Proper Baseline (P0 - RUN FIRST)
```bash
python main.py \
    --train-annotation RAER/annotation/train_80.txt \
    --val-annotation RAER/annotation/val_20.txt \
    --test-annotation RAER/annotation/test.txt \
    --epochs 30 --batch-size 8
```
**Expected**: Valid UAR ~50-52% (proper baseline)

### Experiment 2: Focal Loss + Temperature Fix (P1)
```bash
python main.py \
    --train-annotation RAER/annotation/train_80.txt \
    --val-annotation RAER/annotation/val_20.txt \
    --class-balanced-loss \
    --temperature 0.07 \
    --epochs 30
```
**Expected**: Valid UAR ~58-62%, Class 2 recall ~30-40%

### Experiment 3: Add MI Loss Reduction (P1)
```bash
python main.py \
    --train-annotation RAER/annotation/train_80.txt \
    --val-annotation RAER/annotation/val_20.txt \
    --class-balanced-loss \
    --temperature 0.07 \
    --lambda_mi 0.02 \
    --mi-warmup 15 --mi-ramp 25 \
    --epochs 30
```
**Expected**: Valid UAR ~62-66%, reduce overfitting

### Experiment 4: Full Stack (P1)
```bash
bash experiments/exp_full_stack.sh
```
**Expected**: Valid UAR ~68-72%, Class 2 recall ~45-55%

---

## 📊 SUCCESS METRICS

### Primary Metrics (MUST ACHIEVE)
- ✅ Valid UAR >= 70%
- ✅ Class 2 (Confusion) Recall >= 40%
- ✅ No class with recall < 35%
- ✅ Train-Val UAR gap < 20%

### Secondary Metrics (NICE TO HAVE)
- Valid WAR >= 65%
- Test UAR >= 68%
- Model convergence < 25 epochs

### Monitoring During Training
```
Epoch 15:
  Train UAR: 82.5%  (target: <90% to avoid overfitting)
  Valid UAR: 68.2%  (target: >=70%)
  Train-Val Gap: 14.3%  (target: <20%)

  Per-Class Recall (Validation):
    Class 0 (Neutrality):  72.1%  ✅
    Class 1 (Enjoyment):   45.2%  ✅
    Class 2 (Confusion):   42.9%  ✅ TARGET!
    Class 3 (Fatigue):     58.7%  ✅
    Class 4 (Distraction): 75.3%  ✅
```

---

## 🔧 FILES TO MODIFY

### Priority 1 (Today)
- ✅ `train_debug.sh` - Fix annotation paths
- ✅ `utils/builders.py` - Add get_class_weights function
- ✅ `main.py` - Update defaults (temperature, lambda_mi)
- ✅ `trainer.py` - Add per-class logging

### Priority 2 (Day 2)
- `utils/loss.py` - Add LDAM, improve FocalLoss
- `trainer.py` - Add OHEM, balanced softmax

### Priority 3 (Day 3)
- `dataloader/video_dataloader.py` - Class-conditional augmentation
- `models/Generate_Model.py` - Temporal mixup

### Priority 4 (Day 4)
- New: `calibrate_thresholds.py` - Threshold tuning
- New: `eval_final.py` - Final evaluation

---

## 📁 OUTPUT STRUCTURE

```
outputs/
├── exp01_proper_baseline/
│   ├── log.txt
│   ├── model_best_uar.pth
│   ├── confusion_matrix.png
│   └── per_class_recall.png
├── exp02_focal_temp/
│   ├── log.txt
│   ├── model_best_uar.pth
│   └── ...
└── exp_final/
    ├── calibration_config.pth
    ├── test_results.json
    └── final_report.txt
```

---

## ⚠️ POTENTIAL RISKS & MITIGATION

### Risk 1: Class 2 Still Low Recall Despite Fixes
**Mitigation**:
- Use even stronger class weights (up to 10x)
- Add synthetic samples via mixup/augmentation
- Consider per-sample reweighting

### Risk 2: WAR Drops Significantly
**Mitigation**:
- Acceptable tradeoff (UAR > WAR for this task)
- Use ensemble to balance WAR and UAR
- Calibrate thresholds on validation

### Risk 3: Overfitting Persists
**Mitigation**:
- Add dropout to temporal transformer
- Stronger augmentation for all classes
- Early stopping with patience=10

### Risk 4: Training Instability with Focal Loss
**Mitigation**:
- Use gradient clipping (already enabled)
- Start with lower gamma (1.5 instead of 2.0)
- Monitor loss curves closely

---

## 📝 NOTES

### Class Distribution (Estimated from Log)
```
Train set (~1697 samples):
  Class 0 (Neutrality):  ~650 samples (38%)  <- Majority
  Class 1 (Enjoyment):   ~120 samples (7%)   <- Minority
  Class 2 (Confusion):   ~35 samples (2%)    <- Extreme Minority ⚠️
  Class 3 (Fatigue):     ~180 samples (11%)  <- Minority
  Class 4 (Distraction): ~512 samples (30%)  <- Majority

Val set (424 samples):
  Class 0: ~170 samples
  Class 1: ~32 samples
  Class 2: ~7 samples   ⚠️ VERY FEW!
  Class 3: ~39 samples
  Class 4: ~105 samples
```

### Key Insights
1. Class 2 có RẤT ÍT samples → cần aggressive reweighting
2. Validation set class 2 chỉ 7 samples → high variance in metrics
3. Model collapse vào class 0 và 4 (majority) → cần margin-based loss
4. MI loss có thể gây conflict → giảm weight hoặc disable

---

## 🚀 GETTING STARTED

### Step 1: Backup Current Code
```bash
cd /Users/macbook/Desktop/V1
git add .
git commit -m "Backup before UAR improvement implementation"
```

### Step 2: Run Phase 1 Implementation
```bash
# Will be created by implementation
bash experiments/phase1_foundation.sh
```

### Step 3: Monitor Training
```bash
# Watch per-class recall in real-time
tail -f outputs/exp*/log.txt | grep "Class 2"
```

### Step 4: Compare Results
```bash
# Will be created
python compare_experiments.py
```

---

## 📞 SUPPORT

If any experiment fails or results don't improve:
1. Check log files for errors
2. Verify class weights are computed correctly
3. Ensure validation split is proper (1697/424/528)
4. Monitor per-class recall trends
5. Adjust hyperparameters based on observations

---

**Next Action**: Proceed with Phase 1 implementation →
