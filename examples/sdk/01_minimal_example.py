"""Run the Atlas SDK quickstart configuration end-to-end."""

from atlas import run


if __name__ == "__main__":
    result = run(
        task="Summarize the latest AI news",
        config_path="src/atlas_core/configs/recipe/sdk_quickstart.yaml",
    )
    print(result.final_answer)
