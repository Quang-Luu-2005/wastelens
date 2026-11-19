"""Binary versus multi-class classification with everything else held fixed.

Same HOG + HSV + Lab descriptors, the same 256-64-32 MLP with LayerNorm and dropout, the same
optimizer, batch size, epochs, flips and cross-validation folds as the deployed descriptor MLP.
Only the label map, the number of output units and the loss change:

  six-class softmax + cross-entropy      (the Lab03 model)
  six one-vs-rest sigmoids + BCE         (multi-class built from binary heads)
  binary 1-logit sigmoid + BCE           (the Lab02 formulation)
  binary 2-logit softmax + cross-entropy (mathematically equivalent model family)
"""

import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .cv import CONFIGS
from .mlp_experiments import fit_normalization
from .model import MLPVariant

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
TASKS = {
    # task -> (class names, mapping from the six TrashNet class ids to task labels)
    "six": (CLASSES, np.arange(6)),
    "fiber": (["other", "fiber"], np.array([1, 0, 0, 1, 0, 0])),   # cardboard + paper vs the rest
    "trash": (["other", "trash"], np.array([0, 0, 0, 0, 0, 1])),   # 117 of 2,144 images are trash
}


@dataclass(frozen=True)
class Setup:
    name: str
    task: str
    head: str                 # "softmax", "sigmoid" (one logit) or "ovr" (one sigmoid per class)
    pos_weight: bool = False  # Weight positives by negatives/positives in BCE (imbalanced binary task).

    @property
    def outputs(self) -> int:
        return 1 if self.head == "sigmoid" else len(TASKS[self.task][0])


SETUPS = {setup.name: setup for setup in (
    Setup("six_softmax", "six", "softmax"),
    Setup("six_ovr", "six", "ovr"),
    Setup("fiber_sigmoid", "fiber", "sigmoid"),
    Setup("fiber_softmax2", "fiber", "softmax"),
    Setup("trash_sigmoid", "trash", "sigmoid"),
    Setup("trash_sigmoid_weighted", "trash", "sigmoid", pos_weight=True),
)}


def task_labels(task: str, six_labels: np.ndarray) -> np.ndarray:
    return TASKS[task][1][six_labels]


def make_loss(setup: Setup, train_y: torch.Tensor) -> nn.Module:
    """Cross-entropy on C logits, or binary cross-entropy on sigmoid logits."""
    if setup.head == "softmax":
        return nn.CrossEntropyLoss()
    weight = None
    if setup.pos_weight:
        positives = train_y.float().sum()
        weight = ((len(train_y) - positives) / positives).reshape(1)
    # BCEWithLogitsLoss fuses the sigmoid with the log for numerical stability.
    return nn.BCEWithLogitsLoss(pos_weight=weight)


def targets_for(setup: Setup, labels: torch.Tensor) -> torch.Tensor:
    if setup.head == "softmax":
        return labels                                            # Integer class ids.
    if setup.head == "sigmoid":
        return labels.float().unsqueeze(1)                       # (N, 1) in {0, 1}.
    return F.one_hot(labels, setup.outputs).float()               # (N, C) one-vs-rest targets.


def probabilities(setup: Setup, logits: torch.Tensor) -> torch.Tensor:
    """Per-class probabilities (N, classes of the task)."""
    if setup.head == "softmax":
        return logits.softmax(dim=1)
    if setup.head == "sigmoid":
        positive = torch.sigmoid(logits)
        return torch.cat([1 - positive, positive], dim=1)
    return torch.sigmoid(logits)  # One-vs-rest scores; they need not sum to one.


def logit_gradient(setup: Setup, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Per-sample dLoss/dlogits before batch averaging (without pos_weight).

    softmax + CE: softmax(z) - onehot(y); sigmoid + BCE: sigmoid(z) - y (elementwise for OvR).
    """
    if setup.head == "softmax":
        return logits.softmax(dim=1) - F.one_hot(targets, setup.outputs).float()
    return torch.sigmoid(logits) - targets


def train_fold(setup: Setup, train_x, flipped_x, train_y, eval_x, eval_y, seed: int, epochs: int = 300,
               record_every: int = 10, device: str = "cpu") -> dict:
    """Train one fold with the deployed recipe; return held-out probabilities and training curves."""
    device = torch.device(device)
    train_x, flipped_x, eval_x = (torch.as_tensor(a, device=device) for a in (train_x, flipped_x, eval_x))
    train_y, eval_y = torch.as_tensor(train_y, device=device), torch.as_tensor(eval_y, device=device)
    experiment = CONFIGS["extra_lab"]
    torch.manual_seed(seed)
    model = MLPVariant(image_size=64, num_classes=setup.outputs, **experiment.model_config()).to(device)
    fit_normalization(model, train_x, experiment)  # Standardization from this fold's training images.
    loss_fn = make_loss(setup, train_y).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    order_rng = torch.Generator(device=device).manual_seed(seed + 1000)
    flip_rng = torch.Generator(device=device).manual_seed(seed + 2000)
    history = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_start = time.perf_counter()
        losses, gradient_norms, seen = 0.0, 0.0, 0
        order = torch.randperm(len(train_y), generator=order_rng, device=device)
        for indices in order.split(64):
            flip = torch.rand((len(indices), 1), generator=flip_rng, device=device) < 0.5
            images = torch.where(flip, flipped_x[indices], train_x[indices])
            targets = targets_for(setup, train_y[indices])
            logits = model(images)
            loss = loss_fn(logits, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                losses += loss.item() * len(indices)
                gradient_norms += logit_gradient(setup, logits, targets).norm(dim=1).sum().item()
                seen += len(indices)
        row = {"epoch": epoch, "train_loss": losses / seen, "logit_gradient_norm": gradient_norms / seen,
               "seconds": time.perf_counter() - epoch_start}
        if epoch % record_every == 0 or epoch == epochs:
            model.eval()
            with torch.inference_mode():
                logits = model(eval_x)
                row["eval_loss"] = F.binary_cross_entropy_with_logits(
                    logits, targets_for(setup, eval_y)).item() if setup.head != "softmax" else F.cross_entropy(
                    logits, eval_y).item()
                row["eval_accuracy"] = (probabilities(setup, logits).argmax(1) == eval_y).float().mean().item()
        history.append(row)
    model.eval()
    with torch.inference_mode():
        final = probabilities(setup, model(eval_x)).cpu().numpy()
    return {"probabilities": final, "history": history, "seconds": time.perf_counter() - started,
            "parameters": sum(p.numel() for p in model.parameters()),
            "head_parameters": sum(p.numel() for p in model.layers[-1].parameters())}


def initial_gradient_check(setup: Setup, features: np.ndarray, labels: np.ndarray, seed: int = 0) -> dict:
    """Loss and per-sample logit gradient at initialization; analytic gradient versus autograd."""
    experiment = CONFIGS["extra_lab"]
    torch.manual_seed(seed)
    model = MLPVariant(image_size=64, num_classes=setup.outputs, **experiment.model_config())
    x, y = torch.as_tensor(features[:256]), torch.as_tensor(labels[:256])
    fit_normalization(model, x, experiment)
    model.eval()  # No dropout, so the analytic and autograd gradients see the same logits.
    logits = model(x).detach().requires_grad_(True)
    targets = targets_for(setup, y)
    # Unweighted loss: the analytic formula below has no pos_weight term.
    loss = (F.cross_entropy(logits, targets, reduction="sum") if setup.head == "softmax"
            else F.binary_cross_entropy_with_logits(logits, targets, reduction="sum"))
    loss.backward()
    analytic = logit_gradient(setup, logits.detach(), targets)
    return {"initial_loss": loss.item() / len(y),
            "initial_gradient_norm": analytic.norm(dim=1).mean().item(),
            "analytic_vs_autograd_max_error": (analytic - logits.grad).abs().max().item()}
