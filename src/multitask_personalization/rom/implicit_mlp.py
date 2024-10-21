"""Implicit MLP model for ROM."""

import torch
from torch import nn


class MLPROMClassifierTorch(nn.Module):
    """Implicit MLP model for ROM."""

    def __init__(
        self,
        device="cuda",
        input_size=8,
        hidden_sizes=None,
    ) -> None:
        """Initialize the model."""
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [32, 32, 32]
        layers: list[nn.Module] = []
        self.input_size = input_size
        self.device = device
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())

        self._mlp = nn.Sequential(*layers)
        self._mlp.to(self.device)

        self._classification_offset = 0.0

    def load(self, ckpt_path=None) -> None:
        """Load the model from a checkpoint."""
        if ckpt_path is None:
            # ckpt_path = "src/multitask_personalization/rom/ckpts/implicit-mlp.model"
            ckpt_path = "src/multitask_personalization/rom/ckpts/implicit-mlp_cpu.pth"
        # with open(ckpt_path, "rb") as f:
        #     _, _, state_dict = pickle.load(f)

        # Remove 'net.' prefix from the keys in the state_dict
        # new_state_dict = {}
        # for key in state_dict.keys():
        #     new_key = key.replace("net.", "")  # Remove 'net.' prefix
        #     new_state_dict[new_key] = state_dict[key]

        # new_state_dict = {key.replace("net.", ""): value.to('cpu') for \
        # key, value in state_dict.items()}
        # self._mlp.load_state_dict(new_state_dict)

        print(f"Using device: {self.device}")

        # Load the model state dictionary with map location set to the chosen device
        state_dict = torch.load(ckpt_path, weights_only=True)

        # Load the state dictionary into your model
        self._mlp.load_state_dict(state_dict)

        # Move the entire model to the chosen device
        self._mlp.to(self.device)
        print("Model loaded and moved to device.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self._mlp(x)

    def classify(self, x: torch.Tensor) -> torch.Tensor:
        """Classify the input."""
        return (self.forward(x) + self._classification_offset).round().squeeze()
