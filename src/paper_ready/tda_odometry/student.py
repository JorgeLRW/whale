import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNetLite(nn.Module):
    """Simple PointNet-like encoder -> MLP regressor to predict 51-dim TDA features.

    Input: (B, N, 3) point clouds
    Output: (B, 51)
    """

    def __init__(self, out_dim: int = 51):
        super().__init__()
        # per-point MLP
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)

        # global MLP
        self.fc1 = nn.Linear(128, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, 3) -> (B, 3, N)
        x = x.permute(0, 2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        # global max pool
        x = torch.max(x, dim=2)[0]
        x = F.relu(self.bn3(self.fc1(x)))
        x = F.relu(self.bn4(self.fc2(x)))
        x = self.fc3(x)
        return x


class PointNetLarge(nn.Module):
    """Higher-capacity PointNet-style regressor for TDA features.

    Uses a deeper per-point encoder and larger global MLP. Intended for smoke tests
    to see if capacity reduces feature MSE.
    """

    def __init__(self, out_dim: int = 51, dropout: float = 0.3):
        super().__init__()
        # per-point MLP (Conv1d)
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.bn3 = nn.BatchNorm1d(256)
        # global MLP
        self.fc1 = nn.Linear(256, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn6 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(dropout)
        self.fc4 = nn.Linear(128, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, 3) -> (B, 3, N)
        x = x.permute(0, 2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # global max pool
        x = torch.max(x, dim=2)[0]
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = F.relu(self.bn6(self.fc3(x)))
        x = self.dropout(x)
        x = self.fc4(x)
        return x


def get_student(arch: str = 'pointnet_lite', out_dim: int = 51):
    """Factory for student models.

    arch: one of 'pointnet_lite' or 'pointnet_large'
    """
    arch = (arch or 'pointnet_lite').lower()
    if arch in ('pointnet_lite', 'lite', 'pointnetlite'):
        return PointNetLite(out_dim=out_dim)
    if arch in ('pointnet_large', 'large', 'pointnetlarge'):
        return PointNetLarge(out_dim=out_dim)
    raise ValueError(f'Unknown student arch: {arch}')
