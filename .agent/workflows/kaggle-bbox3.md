---
description: How to run bbox3 combined optimization on Kaggle
---

# Kaggle bbox3 Optimization Workflow

## Prerequisites

1. **Kaggle datasets** (already set up):
   - `saspav/santa-2025-csv` — contains `bbox3` binary
   - `zdanovic/santa-2025-bbox3-baseline` — current best submission
   - `zdanovic/santa-2025-solver` — solver code

## Run Combined Optimization

// turbo-all

1. Push latest code to Kaggle dataset:
```bash
cd /Users/a1234/Documents/vscode_projects/kaggle/santa-2025
kaggle datasets version -p kaggle -m "Update solver code"
```

2. Push kernel to Kaggle:
```bash
cd /Users/a1234/Documents/vscode_projects/kaggle/santa-2025/kaggle
kaggle kernels push -p . -k kernel-metadata.combined-opt.json
```

3. Wait for kernel to complete (~12 hours on Kaggle)

4. Download results:
```bash
kaggle kernels output zdanovic/santa-2025-combined-opt -p results/kaggle_combined
```

5. Merge with local ensemble:
```bash
cd /Users/a1234/Documents/vscode_projects/kaggle/santa-2025
python scripts/ensemble_cascade.py \
  --inputs results/kaggle_combined/best_submission.csv \
          results/submissions/best_submission.csv \
  --output results/submissions/merged.csv
```

## Alternative: Run Individual Kernels

### bbox3 Only
```bash
kaggle kernels push -p kaggle -k kernel-metadata.bbox3-aggressive.json
```

### GB+SA Only
```bash
kaggle kernels push -p kaggle -k kernel-metadata.gb-sa-smalln.json
```

## Verify Submission

```bash
# Check for overlaps and score
python -c "
from santa2025.io import load_submission
from santa2025.validation import validate_submission
sub = load_submission('results/kaggle_combined/best_submission.csv')
result = validate_submission(sub)
print(f'Score: {result.total_score:.6f}')
print(f'Valid: {result.is_valid}')
"
```
