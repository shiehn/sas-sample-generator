import json
import statistics
from pathlib import Path


def main():
    metadata_dir = Path("outputs/raw/_metadata")
    times = []

    for path in metadata_dir.glob("*.json"):
        data = json.loads(path.read_text())
        if "generation_seconds" in data:
            times.append(float(data["generation_seconds"]))

    if not times:
        print("No generation metadata found.")
        return

    avg = statistics.mean(times)
    print(f"Samples measured: {len(times)}")
    print(f"Average seconds/sample: {avg:.2f}")

    for hourly_price in [0.86, 1.19, 1.80]:
        estimated_hours = avg * 1000 / 3600
        estimated_cost = estimated_hours * hourly_price
        print(f"At ${hourly_price}/hr: ~{estimated_hours:.2f} hours, ~${estimated_cost:.2f}")


if __name__ == "__main__":
    main()
