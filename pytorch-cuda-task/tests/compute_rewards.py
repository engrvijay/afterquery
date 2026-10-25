from pathlib import Path
from rewardkit import criterion
import rewardkit as rk

# Example A: Writing custom evaluation logic
@criterion
def validates_output_format(workspace: Path) -> float:
    output_path = workspace / "predictions.json"
    if not output_path.exists():
        return 0.0
    # Your custom evaluation math
    return 1.0

# Example B: Combining with Rewardkit's built-in shorthand helper assertions
@criterion
def check_essential_files(workspace: Path) -> bool:
    return rk.file_exists("README.md") and rk.file_contains("config.json", "prod")

