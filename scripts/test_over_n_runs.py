"""Script to automatically test implementation and give success percentage."""

from pathlib import Path

from sim import simulate

from lsy_drone_racing.utils import load_config


def main():
    """Run the simulation N times and save the results as 'submission.csv'."""
    n_runs = 100
    config_file = "level1.toml"
    config = load_config(Path(__file__).parents[1] / "config" / config_file)
    ep_times = simulate(
        config=config_file, controller=config.controller.file, n_runs=n_runs, gui=False
    )

    # Log the number of failed runs if any
    n_failed = len([x for x in ep_times if x is None])
    if n_failed:
        print(f"{n_failed} run{'' if n_failed == 1 else 's'} failed out of {n_runs}!")
    else:
        print("All runs completed successfully!")

    if n_failed > n_runs / 2:
        print("More than 50% of all runs failed! Aborting submission.")

    ep_times = [x for x in ep_times if x is not None]

    # Calculate success percentage
    success_percentage = (len(ep_times) / n_runs) * 100

    # Calculate average time
    average_time = sum(ep_times) / len(ep_times) if ep_times else 0

    # Print results
    print(f"Success Percentage: {success_percentage:.2f}%")
    print(f"Average Time: {average_time:.2f} seconds")

if __name__ == "__main__":
    main()