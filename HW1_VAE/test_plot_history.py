import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from plot_history import plot_history


class PlotHistoryTests(unittest.TestCase):
    def test_csv_values_render_without_pyplot(self):
        history = [
            {"epoch": str(epoch), "train_loss": "12", "val_loss": "14",
             "train_reconstruction": "10", "val_reconstruction": "11",
             "train_kl": "2", "val_kl": "3"}
            for epoch in (1, 2)
        ]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "loss_curves.png"
            # Any accidental pyplot import must fail, even on a clean install.
            with patch.dict("sys.modules", {"matplotlib.pyplot": None}):
                plot_history(history, path)
            with Image.open(path) as image:
                self.assertEqual(image.size, (1950, 525))
                image.verify()

    def test_empty_history_has_clear_error(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            plot_history([], "unused.png")


if __name__ == "__main__":
    unittest.main()
