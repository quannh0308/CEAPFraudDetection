# ML Evaluation Metrics for Fraud Detection

This document explains the evaluation metrics used in the fraud detection ML pipeline, their definitions, importance, and trade-offs specific to fraud detection systems.

## Table of Contents
- [Core Concepts](#core-concepts)
- [Accuracy](#1-accuracy)
- [Precision](#2-precision)
- [Recall](#3-recall-sensitivity)
- [F1 Score](#4-f1-score)
- [AUC (Area Under ROC Curve)](#5-auc-area-under-roc-curve)
- [Why We Use All Five](#why-we-use-all-five-in-fraud-detection)
- [Fraud Detection Trade-offs](#fraud-detection-trade-offs)

---

## Core Concepts

First, let's understand the **confusion matrix** for binary classification (fraud vs. legitimate):

```
                    Predicted
                 Fraud    Legitimate
Actual  Fraud     TP         FN
        Legit     FP         TN
```

### Definitions:
- **TP (True Positive)**: Correctly identified fraud transaction
- **TN (True Negative)**: Correctly identified legitimate transaction
- **FP (False Positive)**: Legitimate transaction incorrectly flagged as fraud (false alarm)
- **FN (False Negative)**: Fraud transaction incorrectly classified as legitimate (missed fraud - dangerous!)

---

## 1. Accuracy

### Formula
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

### What It Measures
Overall correctness - what percentage of all predictions were correct?

### Example
If we have 100 transactions and correctly classify 95 of them:
```
Accuracy = 95 / 100 = 0.95 (95%)
```

### Why It Matters for Fraud Detection

**Advantages:**
- ✅ Easy to understand and interpret
- ✅ Good for balanced datasets
- ✅ Provides overall model performance snapshot

**Limitations:**
- ❌ **Highly misleading for imbalanced data** (fraud is rare!)
- ❌ Can hide serious problems in minority class detection

### The Imbalanced Data Problem

Fraud detection datasets are typically highly imbalanced:
- Legitimate transactions: 99%+
- Fraud transactions: <1%

**Example of why accuracy fails:**
```
Dataset: 10,000 transactions
- Legitimate: 9,900 (99%)
- Fraud: 100 (1%)

Model A: Predicts "legitimate" for everything
- Accuracy: 9,900 / 10,000 = 99%
- Fraud caught: 0 / 100 = 0%
- Useless model with "excellent" accuracy!

Model B: Actually detects fraud
- Accuracy: 9,500 / 10,000 = 95%
- Fraud caught: 85 / 100 = 85%
- Better model with "worse" accuracy!
```

**Conclusion**: Accuracy alone is insufficient for fraud detection. We need metrics that focus on the minority class (fraud).

---

## 2. Precision

### Formula
```
Precision = TP / (TP + FP)
```

### What It Measures
Of all transactions we flagged as fraud, how many were actually fraud?

Also known as: **Positive Predictive Value (PPV)**

### Example
```
We flag 100 transactions as fraud
- 80 are actually fraud (TP = 80)
- 20 are legitimate (FP = 20)

Precision = 80 / (80 + 20) = 0.80 (80%)
```

### Why It Matters for Fraud Detection

**What precision tells us:**
- Measures the **false alarm rate**
- Low precision = many legitimate transactions get blocked
- High precision = most fraud alerts are real

**Business Impact of Low Precision:**
- 😠 Annoyed customers (legitimate transactions declined)
- 💰 Lost sales and revenue
- 📞 Increased customer support costs
- 🏦 Damage to customer trust and brand reputation
- ⏱️ Wasted investigation time on false alarms

**Real-World Scenario:**
```
Precision = 0.85 (85%)

If you investigate 1,000 fraud alerts:
- 850 are real fraud (good catches)
- 150 are false alarms (wasted effort)

Cost analysis:
- Investigation cost: $10 per alert
- False alarm cost: 150 × $10 = $1,500 wasted
- Plus customer frustration from 150 declined legitimate transactions
```

**Trade-off:**
- Increasing precision (fewer false alarms) often means decreasing recall (missing more fraud)
- You can achieve 100% precision by only flagging transactions you're absolutely certain about, but you'll miss most fraud

---

## 3. Recall (Sensitivity)

### Formula
```
Recall = TP / (TP + FN)
```

### What It Measures
Of all actual fraud transactions, how many did we catch?

Also known as: **Sensitivity, True Positive Rate (TPR), Hit Rate**

### Example
```
There are 100 fraud transactions in the dataset
- We catch 90 of them (TP = 90)
- We miss 10 of them (FN = 10)

Recall = 90 / (90 + 10) = 0.90 (90%)
```

### Why It Matters for Fraud Detection

**What recall tells us:**
- Measures the **fraud detection rate**
- Low recall = lots of fraud slips through undetected
- High recall = we catch most fraud attempts

**Business Impact of Low Recall:**
- 💸 Direct financial losses from undetected fraud
- ⚖️ Regulatory compliance issues and fines
- 📉 Damage to company reputation
- 🎯 Fraudsters learn they can succeed, leading to more attacks
- 📊 Inaccurate fraud statistics and risk assessment

**Real-World Scenario:**
```
Recall = 0.85 (85%)

If there are $1,000,000 in fraud attempts:
- We catch $850,000 (85%)
- We miss $150,000 (15%)

Impact:
- Direct loss: $150,000
- Potential regulatory fines: $50,000
- Reputation damage: Priceless
- Total cost: $200,000+
```

**Trade-off:**
- Increasing recall (catching more fraud) often means decreasing precision (more false alarms)
- You can achieve 100% recall by flagging everything as fraud, but precision will be terrible

---

## 4. F1 Score

### Formula
```
F1 Score = 2 × (Precision × Recall) / (Precision + Recall)
```

This is the **harmonic mean** of precision and recall.

### What It Measures
A single metric that balances both precision and recall concerns.

### Why Harmonic Mean?
The harmonic mean penalizes extreme imbalances more than arithmetic mean:

```
Example 1: Balanced model
Precision = 0.80, Recall = 0.90
Arithmetic mean = (0.80 + 0.90) / 2 = 0.85
F1 (harmonic) = 2 × (0.80 × 0.90) / (0.80 + 0.90) = 0.847

Example 2: Imbalanced model
Precision = 0.95, Recall = 0.50
Arithmetic mean = (0.95 + 0.50) / 2 = 0.725
F1 (harmonic) = 2 × (0.95 × 0.50) / (0.95 + 0.50) = 0.656

The F1 score correctly penalizes the imbalanced model more severely!
```

### Examples

**Scenario 1: Well-balanced model**
```
Precision = 0.85
Recall = 0.87
F1 = 2 × (0.85 × 0.87) / (0.85 + 0.87) = 0.860

Interpretation: Good balance between catching fraud and avoiding false alarms
```

**Scenario 2: High precision, low recall**
```
Precision = 0.95
Recall = 0.60
F1 = 2 × (0.95 × 0.60) / (0.95 + 0.60) = 0.737

Interpretation: Very few false alarms, but missing 40% of fraud
```

**Scenario 3: Low precision, high recall**
```
Precision = 0.60
Recall = 0.95
F1 = 2 × (0.60 × 0.95) / (0.60 + 0.95) = 0.737

Interpretation: Catching most fraud, but many false alarms
```

### Why It Matters for Fraud Detection

**When to use F1 Score:**
- ✅ When precision and recall are both important
- ✅ Comparing models with different precision/recall trade-offs
- ✅ Need a single metric for model selection
- ✅ Balanced cost of false positives and false negatives

**When NOT to use F1 Score:**
- ❌ When one metric is much more important than the other
- ❌ When costs of FP and FN are very different
- ❌ When you need to understand the specific trade-offs

**Variants:**
- **F2 Score**: Weights recall higher than precision (β=2)
  - Use when missing fraud is worse than false alarms
- **F0.5 Score**: Weights precision higher than recall (β=0.5)
  - Use when false alarms are worse than missing fraud

---

## 5. AUC (Area Under ROC Curve)

### What Is ROC?
**ROC (Receiver Operating Characteristic)** curve is a plot that shows model performance across all possible classification thresholds.

**Axes:**
- **X-axis**: False Positive Rate (FPR) = FP / (FP + TN)
  - What percentage of legitimate transactions are incorrectly flagged?
- **Y-axis**: True Positive Rate (TPR) = Recall = TP / (TP + FN)
  - What percentage of fraud transactions are correctly caught?

### What Is AUC?
**AUC (Area Under the Curve)** is the area under the ROC curve.

### Formula
```
AUC = ∫[0 to 1] TPR(FPR) d(FPR)
```

In practice, calculated using trapezoidal rule over all threshold points.

### AUC Value Interpretation

| AUC Range | Interpretation | Model Quality |
|-----------|----------------|---------------|
| 1.0 | Perfect classifier | Separates all fraud from legitimate |
| 0.9 - 1.0 | Excellent | Production-ready |
| 0.8 - 0.9 | Good | Acceptable for most use cases |
| 0.7 - 0.8 | Fair | Needs improvement |
| 0.6 - 0.7 | Poor | Significant issues |
| 0.5 | Random guessing | Coin flip performance |
| < 0.5 | Worse than random | Model is backwards! |

### What AUC Measures

**Probabilistic interpretation:**
```
AUC = Probability that the model ranks a randomly chosen 
      fraud transaction higher than a randomly chosen 
      legitimate transaction
```

**Example:**
```
AUC = 0.95

This means:
- 95% chance the model assigns a higher fraud score to actual fraud 
  than to a legitimate transaction
- Model has excellent discrimination ability
```

### Why It Matters for Fraud Detection

**Advantages:**
- ✅ **Threshold-independent**: Evaluates model quality regardless of where you set the fraud score cutoff
- ✅ Shows trade-off between catching fraud (TPR) and false alarms (FPR)
- ✅ Useful for comparing different models
- ✅ Robust to class imbalance
- ✅ Provides single metric for model ranking

**Use cases:**
1. **Model comparison**: Which model has better discrimination ability?
2. **Threshold selection**: Visualize trade-offs at different thresholds
3. **Model validation**: Ensure model generalizes well to test data

### ROC Curve Example

```
Threshold = 0.9 (very strict)
├─ TPR = 0.60 (catch 60% of fraud)
└─ FPR = 0.01 (1% false alarm rate)

Threshold = 0.7 (moderate)
├─ TPR = 0.85 (catch 85% of fraud)
└─ FPR = 0.05 (5% false alarm rate)

Threshold = 0.5 (lenient)
├─ TPR = 0.95 (catch 95% of fraud)
└─ FPR = 0.15 (15% false alarm rate)

The ROC curve plots all these points, and AUC measures the area underneath.
```

### Real-World Scenario

```
Model A: AUC = 0.95
Model B: AUC = 0.85

Model A is better at distinguishing fraud from legitimate transactions.

Now you can use the ROC curve to choose the optimal threshold:

Option 1: High precision (few false alarms)
- Threshold = 0.9
- Catch 85% of fraud with 2% false alarm rate

Option 2: High recall (catch most fraud)
- Threshold = 0.6
- Catch 95% of fraud with 8% false alarm rate

Option 3: Balanced
- Threshold = 0.75
- Catch 90% of fraud with 5% false alarm rate
```

---

## Why We Use All Five in Fraud Detection

Looking at our `EvaluateHandler` implementation, we calculate all five metrics because each provides unique insights:

```kotlin
val accuracy = calculateAccuracy(predictions, actuals)
val precision = calculatePrecision(predictions, actuals)
val recall = calculateRecall(predictions, actuals)
val f1Score = calculateF1Score(precision, recall)
val auc = calculateAUC(predictions, actuals)
```

### Metric Purposes

| Metric | Primary Purpose | Key Question Answered |
|--------|----------------|----------------------|
| **Accuracy** | Baseline quality check | Is the model better than random? |
| **Precision** | Customer impact | How many false alarms will we have? |
| **Recall** | Financial risk | How much fraud will we miss? |
| **F1 Score** | Balance check | Are precision and recall balanced? |
| **AUC** | Model discrimination | Can the model distinguish fraud from legitimate? |

### Why Each Metric Matters

**1. Accuracy (0.90 threshold in our code)**
```kotlin
if (accuracy < 0.90) {
    throw IllegalStateException(
        "Model accuracy $accuracy is below minimum threshold 0.90"
    )
}
```
- Ensures basic model quality
- Catches catastrophically bad models
- **Not sufficient alone** due to class imbalance

**2. Precision**
- Tells us **customer experience impact**
- Low precision = angry customers, lost sales
- Helps estimate investigation costs

**3. Recall**
- Tells us **financial risk exposure**
- Low recall = money lost to fraud
- Critical for regulatory compliance

**4. F1 Score**
- Quick check that precision and recall are **balanced**
- Useful for comparing models
- Single metric for optimization

**5. AUC**
- Evaluates model's **fundamental ability** to distinguish fraud
- Threshold-independent quality measure
- Best for model comparison and selection

---

## Fraud Detection Trade-offs

In production fraud detection systems, you must choose a strategy based on business needs:

### Strategy 1: Conservative (Minimize Missed Fraud)

**Goal**: Catch as much fraud as possible

**Configuration:**
- High recall (95%+)
- Accept lower precision (more false alarms)
- Lower fraud score threshold (e.g., 0.6)

**Best for:**
- High-value transactions ($10,000+)
- High-risk customers or regions
- Regulatory-sensitive industries (banking, healthcare)
- New fraud patterns emerging

**Trade-offs:**
```
Recall = 0.95 (catch 95% of fraud)
Precision = 0.70 (30% false alarm rate)
F1 = 0.81

Impact:
- Catch $950K of $1M fraud attempts
- 300 false alarms per 1,000 alerts
- Higher investigation costs
- Some customer friction
```

### Strategy 2: Balanced

**Goal**: Optimize F1 score for best overall performance

**Configuration:**
- Balanced precision and recall (85-90%)
- Moderate fraud score threshold (e.g., 0.75)

**Best for:**
- Medium-value transactions ($100-$10,000)
- General customer base
- Standard risk tolerance
- Most production systems

**Trade-offs:**
```
Recall = 0.88 (catch 88% of fraud)
Precision = 0.85 (15% false alarm rate)
F1 = 0.865

Impact:
- Catch $880K of $1M fraud attempts
- 150 false alarms per 1,000 alerts
- Reasonable investigation costs
- Acceptable customer experience
```

### Strategy 3: Customer-Friendly (Minimize False Alarms)

**Goal**: Provide excellent customer experience

**Configuration:**
- High precision (95%+)
- Accept lower recall (miss some fraud)
- Higher fraud score threshold (e.g., 0.85)

**Best for:**
- Low-value transactions (<$100)
- Trusted customers with good history
- Customer-sensitive businesses (e-commerce, subscriptions)
- High-volume, low-margin operations

**Trade-offs:**
```
Recall = 0.75 (catch 75% of fraud)
Precision = 0.95 (5% false alarm rate)
F1 = 0.84

Impact:
- Catch $750K of $1M fraud attempts
- 50 false alarms per 1,000 alerts
- Lower investigation costs
- Excellent customer experience
- Higher fraud losses
```

### Dynamic Threshold Adjustment

In practice, you might use **different thresholds for different scenarios**:

```kotlin
fun getFraudThreshold(transaction: Transaction): Double {
    return when {
        transaction.amount > 10000 -> 0.60  // Conservative for high-value
        transaction.isNewCustomer -> 0.70   // Moderate for new customers
        transaction.hasGoodHistory -> 0.85  // Lenient for trusted customers
        else -> 0.75                        // Balanced default
    }
}
```

### Cost-Based Optimization

You can also optimize based on actual costs:

```
Cost of false positive (FP): $10 (investigation + customer friction)
Cost of false negative (FN): $500 (average fraud loss)

Optimal threshold minimizes:
Total Cost = (FP × $10) + (FN × $500)
```

---

## Practical Recommendations

### For Model Development

1. **Always calculate all five metrics** during evaluation
2. **Use AUC** for model selection and comparison
3. **Use F1** for hyperparameter tuning
4. **Use precision/recall** for threshold selection
5. **Monitor all metrics** in production

### For Production Deployment

1. **Set minimum thresholds** for each metric:
   ```
   Accuracy >= 0.90
   Precision >= 0.80
   Recall >= 0.85
   F1 >= 0.82
   AUC >= 0.90
   ```

2. **Choose strategy** based on business needs (conservative, balanced, customer-friendly)

3. **Monitor metric drift** over time:
   - Fraud patterns change
   - Model performance degrades
   - Retrain when metrics drop

4. **A/B test** different thresholds:
   - Measure business impact
   - Balance fraud losses vs. customer friction
   - Optimize for total cost

### For Reporting

**To technical stakeholders:**
- Show all five metrics
- Include ROC curve
- Explain trade-offs

**To business stakeholders:**
- Focus on precision (false alarm rate) and recall (fraud catch rate)
- Translate to dollar amounts
- Show customer impact

**To executives:**
- Use F1 score as single metric
- Show trend over time
- Highlight business outcomes (fraud prevented, customer satisfaction)

---

## Summary

| Metric | Formula | Answers | Critical For |
|--------|---------|---------|--------------|
| **Accuracy** | (TP+TN)/(TP+TN+FP+FN) | Overall correctness? | Baseline check |
| **Precision** | TP/(TP+FP) | How many alerts are real? | Customer experience |
| **Recall** | TP/(TP+FN) | How much fraud do we catch? | Financial risk |
| **F1 Score** | 2×(P×R)/(P+R) | Are P and R balanced? | Model comparison |
| **AUC** | Area under ROC | Can model distinguish classes? | Model quality |

**Key Takeaway**: In fraud detection, **no single metric tells the whole story**. You need all five to understand model performance, make informed trade-offs, and optimize for business outcomes.

---

## References

- [Confusion Matrix](https://en.wikipedia.org/wiki/Confusion_matrix)
- [ROC Curve](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)
- [F1 Score](https://en.wikipedia.org/wiki/F-score)
- [Precision and Recall](https://en.wikipedia.org/wiki/Precision_and_recall)
- [Imbalanced Learning](https://imbalanced-learn.org/)

## Related Documentation

- [ML Development Lifecycle](./ML-DEVELOPMENT-LIFECYCLE.md)
- [System Architecture](./SYSTEM-ARCHITECTURE.md)
- [EvaluateHandler Implementation](../fraud-training-pipeline/src/main/kotlin/com/fraud/training/EvaluateHandler.kt)
