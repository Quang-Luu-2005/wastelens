"""Render the report's loss, gradient and update formulas to PNG with matplotlib mathtext.

Document exporters print LaTeX blocks as raw code, so the report embeds these images instead.

    .venv\\Scripts\\python.exe scripts/render_formulas.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/figures/formulas"
FORMULAS = {
    "loss-binary": r"$p=\sigma(z)=\dfrac{1}{1+e^{-z}},\qquad L_{BCE}=-\left[\,y\log p+(1-y)\log(1-p)\,\right]$",
    # A side-set capital sigma keeps the sum compact inside the fraction.
    "loss-multiclass": r"$p_k=\dfrac{e^{z_k}}{\Sigma_{j=1}^{C}\,e^{z_j}},\qquad "
                       r"L_{CE}=-\log p_y=-z_y+\log\Sigma_{j=1}^{C}\,e^{z_j}$",
    "gradient": r"$\dfrac{\partial L_{BCE}}{\partial z}=\sigma(z)-y,\qquad "
                r"\dfrac{\partial L_{CE}}{\partial z_k}=p_k-\mathbf{1}[k=y]$",
    "update": r"Nhị phân:  $w\leftarrow w-\eta\,(p-y)\,h$          "
              r"Đa lớp:  $W\leftarrow W-\eta\,(p-\mathrm{onehot}(y))\,h^{T}$",
}


def main():
    matplotlib.rcParams["mathtext.fontset"] = "cm"
    OUT.mkdir(parents=True, exist_ok=True)
    for name, text in FORMULAS.items():
        figure = plt.figure(figsize=(0.01, 0.01))
        figure.text(0, 0, text, fontsize=17)
        figure.savefig(OUT / f"{name}.png", dpi=220, bbox_inches="tight", pad_inches=0.08, facecolor="white")
        plt.close(figure)
        print(OUT / f"{name}.png")


if __name__ == "__main__":
    main()
