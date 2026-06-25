@echo off
echo === Full IR Evaluation ===
echo Touche: all models, baseline + enhanced, FULL data
python -u run_evaluation.py --dataset touche --models all --mode both --output results/evaluation_touche_full.csv
echo.
echo Quora: all models, baseline + enhanced, FULL data
python -u run_evaluation.py --dataset quora --models all --mode both --output results/evaluation_quora_full.csv
echo.
echo Done. Check results/ folder.
