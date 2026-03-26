"""
Extended LSTM (xLSTM) Architecture for Sensor Fusion

This module implements the mLSTM (matrix memory) variant of xLSTM for processing
heterogeneous sensor data with temporal dependencies.

Reference: Beck et al. "xLSTM: Extended Long Short-Term Memory" (2024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math


class mLSTMCell(nn.Module):
    """
    Matrix LSTM (mLSTM) Cell with exponential gating and matrix memory.
    
    Key differences from standard LSTM:
    - Matrix memory instead of scalar memory
    - Exponential gating for better gradient flow
    - Covariance update rule for memory
    """
    
    def __init__(self, input_size: int, hidden_size: int, head_size: int = 32):
        """
        Initialize mLSTM cell.
        
        Args:
            input_size: Dimension of input features
            hidden_size: Dimension of hidden state
            head_size: Dimension of each attention head
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.num_heads = hidden_size // head_size
        
        # Input projections
        self.W_i = nn.Linear(input_size, hidden_size)  # Input gate
        self.W_f = nn.Linear(input_size, hidden_size)  # Forget gate
        self.W_o = nn.Linear(input_size, hidden_size)  # Output gate
        self.W_q = nn.Linear(input_size, hidden_size)  # Query
        self.W_k = nn.Linear(input_size, head_size)    # Key
        self.W_v = nn.Linear(input_size, hidden_size)  # Value
        
        # Layer normalization
        self.ln_q = nn.LayerNorm(hidden_size)
        self.ln_k = nn.LayerNorm(head_size)
        self.ln_v = nn.LayerNorm(hidden_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with appropriate scaling."""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Forward pass of mLSTM cell.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            state: Optional tuple of (h, C, n) where:
                   - h: hidden state (batch_size, hidden_size)
                   - C: matrix memory (batch_size, hidden_size, head_size)
                   - n: normalizer (batch_size, hidden_size)
        
        Returns:
            Tuple of (output, new_state)
        """
        batch_size = x.size(0)
        
        # Initialize state if not provided
        if state is None:
            h = torch.zeros(batch_size, self.hidden_size, device=x.device)
            C = torch.zeros(batch_size, self.hidden_size, self.head_size, device=x.device)
            n = torch.ones(batch_size, self.hidden_size, device=x.device)
        else:
            h, C, n = state
        
        # Compute gates and projections
        i = torch.sigmoid(self.W_i(x))  # Input gate
        f = torch.sigmoid(self.W_f(x))  # Forget gate
        o = torch.sigmoid(self.W_o(x))  # Output gate
        
        q = self.ln_q(self.W_q(x))      # Query
        k = self.ln_k(self.W_k(x))      # Key
        v = self.ln_v(self.W_v(x))      # Value
        
        # Exponential gating (key innovation of xLSTM)
        i_exp = torch.exp(i)
        f_exp = torch.exp(f)
        
        # Update matrix memory (covariance update rule)
        # C_t = f_t ⊙ C_{t-1} + i_t ⊙ (v_t ⊗ k_t^T)
        C_new = f_exp.unsqueeze(-1) * C + i_exp.unsqueeze(-1) * torch.bmm(
            v.unsqueeze(-1), k.unsqueeze(1)
        )
        
        # Update normalizer
        # n_t = f_t ⊙ n_{t-1} + i_t
        n_new = f_exp * n + i_exp
        
        # Compute output using matrix memory
        # h_t = o_t ⊙ (C_t @ k_t / n_t)
        # Use key instead of query for dimension compatibility
        h_new = o * (torch.bmm(C_new, k.unsqueeze(-1)).squeeze(-1) / (n_new + 1e-6))
        
        return h_new, (h_new, C_new, n_new)


class mLSTMLayer(nn.Module):
    """
    Multi-layer mLSTM with residual connections.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        head_size: int = 32,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.cell = mLSTMCell(input_size, hidden_size, head_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple] = None
    ) -> Tuple[torch.Tensor, Tuple]:
        """
        Process sequence through mLSTM layer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            state: Optional initial state
        
        Returns:
            Tuple of (outputs, final_state)
        """
        batch_size, seq_len, _ = x.size()
        
        outputs = []
        current_state = state
        
        for t in range(seq_len):
            h, current_state = self.cell(x[:, t, :], current_state)
            outputs.append(h)
        
        outputs = torch.stack(outputs, dim=1)  # (batch, seq_len, hidden)
        outputs = self.dropout(outputs)
        outputs = self.layer_norm(outputs)
        
        return outputs, current_state


class SensorFusionXLSTM(nn.Module):
    """
    xLSTM-based model for sensor fusion and temporal modeling.
    
    Architecture:
    1. Input projection for heterogeneous sensors
    2. Multi-layer mLSTM for temporal dependencies
    3. Multi-head attention for sensor fusion
    4. Output heads for position and activity prediction
    """
    
    def __init__(
        self,
        feature_dim: int,
        hidden_size: int = 256,
        num_layers: int = 3,
        head_size: int = 32,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        output_position_dim: int = 3,  # (x, y, floor)
        output_activity_dim: int = 5   # Number of activity classes
    ):
        """
        Initialize the sensor fusion xLSTM model.
        
        Args:
            feature_dim: Dimension of input features
            hidden_size: Hidden dimension for LSTM layers
            num_layers: Number of mLSTM layers
            head_size: Size of each attention head in mLSTM
            num_attention_heads: Number of attention heads for sensor fusion
            dropout: Dropout probability
            output_position_dim: Dimension of position output
            output_activity_dim: Number of activity classes
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # mLSTM layers
        self.lstm_layers = nn.ModuleList([
            mLSTMLayer(
                input_size=hidden_size if i == 0 else hidden_size,
                hidden_size=hidden_size,
                head_size=head_size,
                dropout=dropout
            )
            for i in range(num_layers)
        ])
        
        # Multi-head attention for sensor fusion
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.attention_norm = nn.LayerNorm(hidden_size)
        
        # Output heads
        self.position_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_position_dim)
        )
        
        self.activity_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_activity_dim)
        )
        
        # Uncertainty estimation (optional)
        self.position_uncertainty = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_position_dim),
            nn.Softplus()  # Ensure positive uncertainty
        )
    
    def forward(
        self,
        x: torch.Tensor,
        return_uncertainty: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the model.
        
        Args:
            x: Input features of shape (batch_size, seq_len, feature_dim)
            return_uncertainty: Whether to return position uncertainty
        
        Returns:
            Tuple of (position, activity_logits, uncertainty)
            - position: (batch_size, seq_len, position_dim)
            - activity_logits: (batch_size, seq_len, activity_dim)
            - uncertainty: (batch_size, seq_len, position_dim) or None
        """
        # Input projection
        x = self.input_proj(x)  # (batch, seq_len, hidden)
        
        # Pass through mLSTM layers
        for lstm_layer in self.lstm_layers:
            x_out, _ = lstm_layer(x)
            x = x + x_out  # Residual connection
        
        # Self-attention for sensor fusion
        attn_out, _ = self.attention(x, x, x)
        x = self.attention_norm(x + attn_out)  # Residual + norm
        
        # Output predictions
        position = self.position_head(x)
        activity_logits = self.activity_head(x)
        
        uncertainty = None
        if return_uncertainty:
            uncertainty = self.position_uncertainty(x)
        
        return position, activity_logits, uncertainty
    
    def predict_step(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single-step prediction (for inference).
        
        Args:
            x: Input features of shape (batch_size, 1, feature_dim)
        
        Returns:
            Tuple of (position, activity_class)
        """
        self.eval()
        with torch.no_grad():
            position, activity_logits, _ = self.forward(x)
            activity_class = torch.argmax(activity_logits, dim=-1)
            return position[:, -1, :], activity_class[:, -1]


def create_model(feature_dim: int, config: Optional[dict] = None) -> SensorFusionXLSTM:
    """
    Create a SensorFusionXLSTM model with default or custom configuration.
    
    Args:
        feature_dim: Input feature dimension
        config: Optional configuration dictionary
    
    Returns:
        Initialized model
    """
    default_config = {
        'hidden_size': 256,
        'num_layers': 3,
        'head_size': 32,
        'num_attention_heads': 8,
        'dropout': 0.1,
        'output_position_dim': 3,
        'output_activity_dim': 5
    }
    
    if config is not None:
        default_config.update(config)
    
    return SensorFusionXLSTM(feature_dim=feature_dim, **default_config)


if __name__ == "__main__":
    # Example usage
    print("=== xLSTM Sensor Fusion Model Demo ===\n")
    
    # Model configuration
    feature_dim = 70  # Example: from feature extractor
    batch_size = 4
    seq_len = 20
    
    # Create model
    model = create_model(feature_dim)
    
    print(f"Model architecture:")
    print(f"  Input dimension: {feature_dim}")
    print(f"  Hidden size: {model.hidden_size}")
    print(f"  Number of layers: {model.num_layers}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    # Create sample input
    x = torch.randn(batch_size, seq_len, feature_dim)
    
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    position, activity_logits, uncertainty = model(x, return_uncertainty=True)
    
    print(f"\nOutput shapes:")
    print(f"  Position: {position.shape}")
    print(f"  Activity logits: {activity_logits.shape}")
    print(f"  Uncertainty: {uncertainty.shape}")
    
    # Single-step prediction
    x_single = torch.randn(1, 1, feature_dim)
    pos_pred, act_pred = model.predict_step(x_single)
    
    print(f"\nSingle-step prediction:")
    print(f"  Position: {pos_pred.shape}")
    print(f"  Activity class: {act_pred.item()}")
