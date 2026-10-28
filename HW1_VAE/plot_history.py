import argparse
import csv
from pathlib import Path


def plot_history(history, path):
    # Avoid pyplot: its style discovery reads binary macOS ._*.mplstyle
    # metadata files as UTF-8 on some external filesystems.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    if not history:
        raise ValueError("Training history is empty")
    fig = Figure(figsize=(13, 3.5))
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, 3)
    for ax, metric in zip(axes, ("loss", "reconstruction", "kl")):
        for split in ("train", "val"):
            ax.plot([float(r["epoch"]) for r in history],
                    [float(r[f"{split}_{metric}"]) for r in history], label=split)
        ax.set(xlabel="Epoch", ylabel="Per image", title=metric)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=Path("runs/vae/history.csv"))
    parser.add_argument("--output", type=Path,
                        help="Defaults to loss_curves.png beside the history file")
    args = parser.parse_args()
    with args.history.open(newline="", encoding="utf-8") as stream:
        history = list(csv.DictReader(stream))
    output = args.output or args.history.with_name("loss_curves.png")
    plot_history(history, output)
    print(f"Saved loss curves to {output}")


if __name__ == "__main__":
    main()
