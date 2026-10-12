set -euo pipefail
 
EXP="student-dropout-logreg-hparams"
 
# 1. Base 
uv run train.py --experiment-name "$EXP"
 
# 2-3. Only C changes
uv run train.py --experiment-name "$EXP" --C 0.01
uv run train.py --experiment-name "$EXP" --C 100
 
# 4. Only class_weight changes
uv run train.py --experiment-name "$EXP" --class-weight balanced
 
# 5. class_weight balanced + stronger regularization (combination)
uv run train.py --experiment-name "$EXP" --class-weight balanced --C 0.1
 
# 6. Only max_iter changes (few iterations: may not converge)
uv run train.py --experiment-name "$EXP" --max-iter 20
 
# 7-8. Only the seed changes (measures variation caused by the data split)
uv run train.py --experiment-name "$EXP" --seed 7
uv run train.py --experiment-name "$EXP" --seed 123
 
echo "Done."
