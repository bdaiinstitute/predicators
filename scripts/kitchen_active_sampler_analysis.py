"""Analysis for kitchen active sampler learning."""

import glob
import logging
import os
from typing import Callable, Dict, List, Optional, Tuple

import dill as pkl
import imageio
import matplotlib.pyplot as plt
import numpy as np
from bosdyn.client import math_helpers
from matplotlib import colormaps
from matplotlib.colors import Normalize

from predicators import utils
from predicators.ml_models import BinaryClassifier, MLPBinaryClassifier
from predicators.settings import CFG
from predicators.structs import Array, Image


def _main() -> None:
    # Parse & validate args
    args = utils.parse_args()
    utils.update_config(args)
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    logging.basicConfig(level=CFG.loglevel,
                        format="%(message)s",
                        handlers=handlers)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
    _analyze_saved_data()


def _analyze_saved_data() -> None:
    """Use this to analyze the data saved in saved_datasets/."""
    nsrt_name = "MoveTo"
    objects_tuple_str = "gripper:gripper, kettle:obj"
    prefix = f"{CFG.data_dir}/{CFG.env}_{nsrt_name}({objects_tuple_str})_"
    filepath_template = f"{prefix}*.data"
    all_saved_files = glob.glob(filepath_template)
    X: List[Array] = []
    y: List[Array] = []
    times: List[int] = []
    for filepath in all_saved_files:
        with open(filepath, "rb") as f:
            datum = pkl.load(f)
        X_i, y_i = datum["datapoint"]
        time_i = datum["time"]
        X.append(X_i)
        y.append(y_i)
        times.append(time_i)
    idxs = [i for (i, _) in sorted(enumerate(times), key=lambda i: i[1])]
    X = [X[i] for i in idxs]
    y = [y[i] for i in idxs]
    img = _create_image(X, y)
    img_outfile = "kitchen_data_analysis.png"
    imageio.imsave(img_outfile, img)
    print(f"Wrote out to {img_outfile}")
    _run_sample_efficiency_analysis(X, y)


def _create_image(X: List[Array],
                  y: List[Array]) -> Image:
    cmap = colormaps.get_cmap('RdYlGn')
    norm = Normalize(vmin=0.0, vmax=1.0)

    dims = [
        "constant", 
        "gripper x",
        "gripper y",
        "gripper z",
        "kettle x", 
        "kettle y",
        "kettle z",
        "kettle angle",
        "act dx",
        "act dy",
        "act dz"
    ]
    assert np.array(X).shape[1] == len(dims)

    fig, axes = plt.subplots(4, 3, sharex=False, sharey=False)

    rng = np.random.default_rng(0)

    # plot real data
    for i, (ax, dim) in enumerate(zip(axes.flat, dims)):
        min_x = np.inf
        max_x = -np.inf
        for datum, label in zip(X, y):
            x_pt = datum[i]
            min_x = min(min_x, x_pt)
            max_x = max(max_x, x_pt)
            color = cmap(norm(label))
            circle = plt.Circle((x_pt, label + rng.uniform(-0.1, 0.1)), 0.01, color=color, alpha=0.5)
            ax.add_patch(circle)
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlim((-0.1 + min_x, 0.1 + max_x))
        ax.set_title(dim)

    plt.tight_layout()

    return utils.fig2data(fig, dpi=150)


class _ConstantModel(BinaryClassifier):
    """Oracle hand-written model."""

    def __init__(self, seed: int, constant: bool) -> None:
        super().__init__(seed)
        self._constant = constant

    def fit(self, X: Array, y: Array) -> None:
        pass

    def classify(self, x: Array) -> bool:
        return self._constant

    def predict_proba(self, x: Array) -> float:
        return 1.0 if self.classify(x) else 0.0


def _run_sample_efficiency_analysis(X: List[Array], y: List[Array]) -> None:

    # Do k-fold cross validation.
    validation_frac = 0.1
    num_data = len(X)
    num_valid = int(num_data * validation_frac)
    num_trials = 3

    models: Dict[str, Callable[[], BinaryClassifier]] = {
        "always-true": lambda: _ConstantModel(CFG.seed, True),
        # "always-false":
        # lambda: _ConstantModel(CFG.seed, False),
        "mlp":
        lambda: MLPBinaryClassifier(
            seed=CFG.seed,
            balance_data=CFG.mlp_classifier_balance_data,
            max_train_iters=CFG.sampler_mlp_classifier_max_itr,
            learning_rate=CFG.learning_rate,
            weight_decay=CFG.weight_decay,
            use_torch_gpu=CFG.use_torch_gpu,
            train_print_every=CFG.pytorch_train_print_every,
            n_iter_no_change=CFG.mlp_classifier_n_iter_no_change,
            hid_sizes=CFG.mlp_classifier_hid_sizes,
            n_reinitialize_tries=CFG.
            sampler_mlp_classifier_n_reinitialize_tries,
            weight_init="default")
    }
    training_data_results = {}
    for training_frac in np.linspace(0.05, 1.0 - validation_frac, 5):
        num_training_data = int(len(X) * training_frac)
        model_accuracies = {}
        for model_name, create_model in models.items():
            print("Starting model:", model_name)
            rng = np.random.default_rng(CFG.seed)
            model_accuracy = []
            for i in range(num_trials):
                # Split the data randomly.
                idxs = list(range(num_data))
                rng.shuffle(idxs)
                train_idxs = idxs[num_valid:num_valid + num_training_data]
                valid_idxs = idxs[:num_valid]
                X_train = np.array([X[i] for i in train_idxs])
                y_train = np.array([y[i] for i in train_idxs])
                X_valid = [X[i] for i in valid_idxs]
                y_valid = [y[i] for i in valid_idxs]
                # Train.
                model = create_model()
                model.fit(X_train, y_train)
                # Predict.
                y_pred = [model.classify(x) for x in X_valid]
                acc = np.mean([(y == y_hat)
                               for y, y_hat in zip(y_valid, y_pred)])
                print(f"Trial {i} accuracy: {acc}")
                model_accuracy.append(acc)
            model_accuracies[model_name] = model_accuracy
        print(f"Overall accuracies for training_frac={training_frac}")
        print("------------------")
        for model_name, model_accuracy in model_accuracies.items():
            print(f"{model_name}: {np.mean(model_accuracy)}")
        training_data_results[num_training_data] = model_accuracies

    # Make num training data versus validation accuracy plot.
    plt.figure()
    plt.title("Kitchen (Move To Kettle) Offline Sample Complexity Analysis")
    plt.xlabel("# Training Examples for Sampler")
    plt.ylabel("Validation Classification Accuracy")
    xs = sorted(training_data_results)
    for model_name in models:
        all_ys = np.array([training_data_results[x][model_name] for x in xs])
        ys = np.mean(all_ys, axis=1)
        ys_std = np.std(all_ys, axis=1)
        plt.plot(xs, ys, label=model_name)
        plt.fill_between(xs, ys - ys_std, ys + ys_std, alpha=0.2)
    plt.legend()
    plt.xlim(left=0)
    plt.ylim(-0.1, 1.1)
    outfile = "kitchen_moveto_sample_complexity.png"
    plt.savefig(outfile)
    print(f"Wrote out to {outfile}")


if __name__ == "__main__":
    _main()
