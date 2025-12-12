import matplotlib.pyplot as plt
import torch
from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from laplace import Laplace
from laplace.likelihood import Likelihood


class StaticOccupancyObjective(torch.nn.Module):
    def __init__(self, eps=1e-12):
        super().__init__()
        self.eps = eps

    def safe_log(self, x):
        return torch.log(torch.clamp(x, self.eps, 1.0))

    def _site_log_prob(self, psi, p, y):
        log_psi = self.safe_log(psi)
        log_1_psi = self.safe_log(1.0 - psi)
        log_p = self.safe_log(p)
        log_1_p = self.safe_log(1.0 - p)

        any_det = (y > 0).any(dim=1)
        log_like_occ = log_psi + torch.sum(y * log_p + (1 - y) * log_1_p, dim=1)
        log_like_all0_occ = log_psi + torch.sum(log_1_p, dim=1)
        log_like_all0_unocc = log_1_psi

        log_site_like = torch.where(
            any_det,
            log_like_occ,
            torch.logaddexp(log_like_all0_occ, log_like_all0_unocc),
        )
        return log_site_like

    def forward(self, psi, p, y):
        log_site_like = self._site_log_prob(psi, p, y)
        return -torch.sum(log_site_like)


class OccupancyLikelihood(Likelihood):
    """Custom likelihood for static occupancy with detection replicates."""

    def __init__(self, n_visits: int):
        super().__init__(name="occupancy")
        self.n_visits = n_visits
        self.obj = StaticOccupancyObjective()

    def forward(
        self,
        f: torch.Tensor,
        y: torch.Tensor,
        *,
        reduction: str = "sum",
        temperature: float = 1.0,
    ) -> torch.Tensor:
        psi_logits = f[:, :1]
        p_logits = f[:, 1 : 1 + self.n_visits]
        psi = torch.sigmoid(psi_logits)
        p = torch.sigmoid(p_logits)
        loss = self.obj(psi, p, y) / temperature
        if reduction == "mean":
            loss = loss / f.shape[0]
        return loss

    def predict(
        self,
        f: torch.Tensor,
        *,
        temperature: float = 1.0,
        link: str = "identity",
    ) -> torch.Tensor:
        psi = torch.sigmoid(f[:, :1] / temperature)
        return psi

    def transform_function_samples(self, samples: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(samples[:, :, :1])

    def check_sigma_noise(self, sigma_noise: torch.Tensor | float) -> None:
        sigma = torch.as_tensor(sigma_noise)
        if sigma.ndim == 0:
            sigma = sigma.reshape(1)
        if not torch.allclose(sigma, torch.ones_like(sigma)):
            raise ValueError("sigma_noise must be 1 for occupancy likelihood.")

    def is_classification_like(self) -> bool:
        return True

    def validate_output_targets(
        self, output: torch.Tensor, target: torch.Tensor
    ) -> None:
        if target.ndim != 2 or output.shape[1] < target.shape[1] + 1:
            raise ValueError("Output/target shapes incompatible for occupancy loss.")


class OccupancyModel(nn.Module):
    def __init__(self, n_visits: int):
        super().__init__()
        self.psi_net = nn.Sequential(
            nn.Linear(2, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
        )
        self.p_net = nn.Sequential(
            nn.Linear(2, 16),
            nn.Tanh(),
            nn.Linear(16, n_visits),
        )

    def forward(self, x: torch.Tensor):
        # x shape: (batch, 4). First two dims for psi covariates, last two for p covariates.
        x_psi, x_p = x[:, :2], x[:, 2:]
        psi_logits = self.psi_net(x_psi)
        p_logits = self.p_net(x_p)
        return torch.cat([psi_logits, p_logits], dim=1)


def simulate_data(n_sites=400, n_visits=3, seed=0):
    torch.manual_seed(seed)
    x_psi = torch.rand(n_sites, 2) * 2 - 1  # in [-1,1]
    x_p = torch.rand(n_sites, 2) * 2 - 1
    X = torch.cat([x_psi, x_p], dim=1)
    psi_true = torch.sigmoid(4.0 * x_psi[:, :1] - 4.0 * x_psi[:, 1:])  # sharper boundary
    p_true = torch.sigmoid(3.0 * x_p[:, :1] + 2.0 * x_p[:, 1:])
    psi_sample = torch.bernoulli(psi_true)
    y = torch.zeros(n_sites, n_visits)
    for j in range(n_visits):
        detections = torch.bernoulli(psi_sample * p_true)
        y[:, j] = detections.squeeze(1)
    return X, y.long()


def train_map(model: nn.Module, loader: DataLoader, epochs: int = 400, lr: float = 1e-2):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    obj = StaticOccupancyObjective()
    for _ in range(epochs):
        for X, y in loader:
            opt.zero_grad()
            f = model(X)
            psi = torch.sigmoid(f[:, :1])
            p = torch.sigmoid(f[:, 1:])
            loss = obj(psi, p, y)
            loss.backward()
            opt.step()


def main():
    n_visits = 3
    X, y = simulate_data(n_sites=400, n_visits=n_visits)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = OccupancyModel(n_visits)
    train_map(model, loader)

    likelihood = OccupancyLikelihood(n_visits)
    variants = [
        ("full", model),
        ("diag", model),
    ]

    curves = []
    for hess, mdl in variants:
        # sweep first psi covariate; hold others at mean
        x1_grid = torch.linspace(-1.2, 1.2, 200).unsqueeze(1)
        x2_const = torch.zeros_like(x1_grid)
        x_psi_grid = torch.cat([x1_grid, x2_const], dim=1)
        x_p_grid = torch.zeros_like(x_psi_grid)  # hold detection covariates at mean
        X_grid = torch.cat([x_psi_grid, x_p_grid], dim=1)
        la = Laplace(
            mdl,
            likelihood,
            subset_of_weights="all",
            hessian_structure=hess,
            prior_precision=10.0,
        )
        la.fit(loader)
        samples = la._nn_predictive_samples(X_grid, n_samples=200)
        psi_mean = samples.mean(0).squeeze().detach().numpy()
        psi_std = samples.std(0).squeeze().detach().numpy()
        curves.append((hess, psi_mean, psi_std))

    # use the grid from the last iteration for plotting the MAP curve from the baseline model
    x_np = x1_grid.squeeze().numpy()
    with torch.no_grad():
        logits_map = model(X_grid)
        psi_map = torch.sigmoid(logits_map[:, :1]).squeeze().numpy()

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.scatter(
        X[:, 0].numpy(),
        y.sum(dim=1).clamp(max=1).numpy(),
        c="gray",
        alpha=0.2,
        s=8,
        label="data (any detection)",
    )
    ax.plot(x_np, psi_map, linestyle="--", color="#1f9d55", label="MAP psi")
    colors = {
        "full": "#16a34a",
        "diag": "#0ea5e9",
    }
    for hess, mean, std in curves:
        label = hess
        color = colors.get(hess, "#16a34a")
        ax.plot(x_np, mean, color=color, label=f"{label} mean")
        ax.fill_between(
            x_np,
            (mean - std).clip(0, 1),
            (mean + std).clip(0, 1),
            color=color,
            alpha=0.15,
        )
    ax.set_xlabel("covariate_psi_0")
    ax.set_ylabel("P(occupancy)")
    ax.set_ylim([-0.05, 1.05])
    ax.legend(loc="upper right")
    ax.set_title("Occupancy probability vs covariate with Laplace uncertainty")
    fig.tight_layout()

    out_path = Path("docs/assets/occupancy_custom_likelihood.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
