from datasets import load_dataset

def fetch_and_explore():
    for split_name in ['filtered', 'raw']:
        print(f"\n=== Exploring '{split_name}' split ===")
        try:
            dataset = load_dataset("allenai/reward-bench", split=split_name)
            print(f"Dataset loaded: {len(dataset)} samples")
            print(f"Features: {dataset.features.keys()}")

            print(f"\n=== First Sample from {split_name} ===")
            sample = dataset[0]
            for key, value in sample.items():
                if isinstance(value, str) and len(value) > 200:
                    print(f"{key}: {value[:200]}...")
                else:
                    print(f"{key}: {value}")

        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    fetch_and_explore()