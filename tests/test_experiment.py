"""
Unit tests for Experimental Comparison Engine.
"""
import pytest
from pathlib import Path
from modules.experiment_runner import ExperimentRunner

@pytest.fixture
def runner():
    return ExperimentRunner()

def test_load_dataset(runner):
    dataset = runner.load_dataset()
    assert len(dataset) >= 40
    assert any(d["id"] == "IND-001" for d in dataset)
    assert any(d["name"] == "Spur Gear 36T" for d in dataset)

def test_run_small_validation_experiment(runner):
    # Run with 1 sample (IND-005: Spur gear), 1 seed, 3 conditions
    df = runner.run_experiment(
        sample_ids=["IND-005"],
        seeds=[42],
        conditions=["Condition_A_Naive", "Condition_B_LLM_Prompt", "Condition_C_LLM_ControlNet"],
        output_csv_filename="test_validation_experiment.csv"
    )

    assert not df.empty
    assert len(df) == 3
    assert set(df["condition"]) == {"Condition_A_Naive", "Condition_B_LLM_Prompt", "Condition_C_LLM_ControlNet"}
    assert "clip_score" in df.columns
    assert "ssim_score" in df.columns
    assert "overall_design_fidelity" in df.columns

    # Verify CSV file was actually written to disk
    from utils.helpers import EXPERIMENTS_DIR
    csv_file = EXPERIMENTS_DIR / "test_validation_experiment.csv"
    assert csv_file.exists()

    # Summary statistics computation
    summary = runner.get_summary_statistics(df)
    assert not summary.empty
