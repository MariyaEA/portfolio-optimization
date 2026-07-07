"""Run the full final workflow."""
from __future__ import annotations

import argparse

from scripts.run_task1_pipeline import main as run_task1
from scripts.run_task2_models import main as run_task2
from scripts.run_task3_4_5_final import main as run_task3_4_5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-lstm", action="store_true", help="Skip LSTM if TensorFlow is not installed.")
    parser.add_argument("--lstm-epochs", type=int, default=10)
    args = parser.parse_args()

    run_task1()
    run_task2(run_lstm=not args.no_lstm, lstm_epochs=args.lstm_epochs)
    run_task3_4_5()


if __name__ == "__main__":
    main()
