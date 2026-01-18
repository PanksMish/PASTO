"""
Trajectory Encoder Module for PASTO
Implements LSTM and Transformer-based encoders for temporal student data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import math


class LSTMEncoder(nn.Module):
    """
    LSTM-based trajectory encoder
    Encodes temporal student data into latent representations
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        """
        Initialize LSTM encoder
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden state dimension
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTM
        """
        super(LSTMEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Output dimension
        self.output_dim = hidden_dim * self.num_directions
        
    def forward(
        self, 
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through LSTM encoder
        
        Args:
            x: Input sequences (batch_size, seq_len, input_dim)
            lengths: Actual sequence lengths (batch_size,)
            
        Returns:
            Tuple of:
                - Sequence outputs (batch_size, seq_len, hidden_dim * num_directions)
                - Final hidden state (batch_size, hidden_dim * num_directions)
        """
        batch_size = x.size(0)
        
        # Pack sequences if lengths provided
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
        
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Unpack sequences if packed
        if lengths is not None:
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out, batch_first=True
            )
        
        # Apply dropout
        lstm_out = self.dropout(lstm_out)
        
        # Concatenate forward and backward hidden states
        if self.bidirectional:
            # h_n shape: (num_layers * 2, batch_size, hidden_dim)
            # Take last layer, concatenate forward and backward
            h_forward = h_n[-2, :, :]  # Forward direction
            h_backward = h_n[-1, :, :]  # Backward direction
            final_hidden = torch.cat([h_forward, h_backward], dim=1)
        else:
            final_hidden = h_n[-1, :, :]  # Last layer
        
        return lstm_out, final_hidden


class TransformerEncoder(nn.Module):
    """
    Transformer-based trajectory encoder
    Uses self-attention to capture long-range dependencies
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.3,
        max_seq_length: int = 100
    ):
        """
        Initialize Transformer encoder
        
        Args:
            input_dim: Input feature dimension
            d_model: Transformer model dimension
            nhead: Number of attention heads
            num_encoder_layers: Number of encoder layers
            dim_feedforward: Feedforward network dimension
            dropout: Dropout probability
            max_seq_length: Maximum sequence length
        """
        super(TransformerEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_seq_length)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output dimension
        self.output_dim = d_model
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through Transformer encoder
        
        Args:
            x: Input sequences (batch_size, seq_len, input_dim)
            mask: Attention mask (seq_len, seq_len)
            
        Returns:
            Tuple of:
                - Sequence outputs (batch_size, seq_len, d_model)
                - Pooled output (batch_size, d_model)
        """
        # Project input to d_model
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Create causal mask if not provided
        if mask is None:
            seq_len = x.size(1)
            mask = self._generate_square_subsequent_mask(seq_len).to(x.device)
        
        # Transformer encoding
        encoder_output = self.transformer_encoder(x, mask=mask)
        
        # Apply dropout
        encoder_output = self.dropout(encoder_output)
        
        # Pool over sequence dimension (mean pooling)
        pooled_output = encoder_output.mean(dim=1)
        
        return encoder_output, pooled_output
    
    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate causal mask for autoregressive modeling"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask


class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer
    Adds position information to input embeddings
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """
        Initialize positional encoding
        
        Args:
            d_model: Model dimension
            dropout: Dropout probability
            max_len: Maximum sequence length
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension
        
        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            
        Returns:
            Input with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TrajectoryEncoder(nn.Module):
    """
    Unified trajectory encoder supporting both LSTM and Transformer
    """
    
    def __init__(self, config: Dict):
        """
        Initialize trajectory encoder based on config
        
        Args:
            config: Configuration dictionary with encoder settings
        """
        super(TrajectoryEncoder, self).__init__()
        
        self.config = config
        encoder_config = config['model']['encoder']
        encoder_type = encoder_config['type'].lower()
        
        # Get input dimension
        self.input_dim = encoder_config.get('input_dim', 64)
        
        # Initialize appropriate encoder
        if encoder_type == 'lstm':
            self.encoder = LSTMEncoder(
                input_dim=self.input_dim,
                hidden_dim=encoder_config['hidden_dim'],
                num_layers=encoder_config['num_layers'],
                dropout=encoder_config['dropout'],
                bidirectional=encoder_config.get('bidirectional', True)
            )
        elif encoder_type == 'transformer':
            transformer_config = config['model']['transformer']
            self.encoder = TransformerEncoder(
                input_dim=self.input_dim,
                d_model=transformer_config['d_model'],
                nhead=transformer_config['nhead'],
                num_encoder_layers=transformer_config['num_encoder_layers'],
                dim_feedforward=transformer_config['dim_feedforward'],
                dropout=transformer_config['dropout']
            )
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
        
        self.encoder_type = encoder_type
        self.output_dim = self.encoder.output_dim
        
    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through trajectory encoder
        
        Args:
            x: Input sequences (batch_size, seq_len, input_dim)
            lengths: Sequence lengths for LSTM (batch_size,)
            mask: Attention mask for Transformer
            
        Returns:
            Tuple of (sequence_outputs, final_representation)
        """
        if self.encoder_type == 'lstm':
            return self.encoder(x, lengths)
        else:
            return self.encoder(x, mask)
    
    def get_output_dim(self) -> int:
        """Get output dimension of encoder"""
        return self.output_dim


class StateDiscretizer(nn.Module):
    """
    Discretizes continuous trajectory representations into discrete policy states
    Maps latent representations to a finite state space for RMAB
    """
    
    def __init__(
        self,
        input_dim: int,
        num_risk_bins: int = 4,
        num_engagement_bins: int = 4,
        num_transient_states: int = 7,
        dropout_state_id: int = 24
    ):
        """
        Initialize state discretizer
        
        Args:
            input_dim: Input representation dimension
            num_risk_bins: Number of risk level bins
            num_engagement_bins: Number of engagement level bins
            num_transient_states: Number of additional transient states
            dropout_state_id: ID for absorbing dropout state
        """
        super(StateDiscretizer, self).__init__()
        
        self.input_dim = input_dim
        self.num_risk_bins = num_risk_bins
        self.num_engagement_bins = num_engagement_bins
        self.num_transient_states = num_transient_states
        self.dropout_state_id = dropout_state_id
        
        # Total number of states
        self.num_base_states = num_risk_bins * num_engagement_bins
        self.num_states = self.num_base_states + num_transient_states + 1  # +1 for dropout
        
        # Project representation to risk and engagement scores
        self.risk_projection = nn.Linear(input_dim, 1)
        self.engagement_projection = nn.Linear(input_dim, 1)
        
        # Transient state classifier (for high-risk unstable states)
        self.transient_classifier = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_transient_states)
        )
        
    def forward(
        self,
        h: torch.Tensor,
        dropout_labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Discretize continuous representations into states
        
        Args:
            h: Latent representations (batch_size, hidden_dim)
            dropout_labels: Known dropout labels if available (batch_size,)
            
        Returns:
            Discrete state IDs (batch_size,)
        """
        batch_size = h.size(0)
        
        # Compute risk and engagement scores
        risk_scores = torch.sigmoid(self.risk_projection(h)).squeeze(-1)  # [0, 1]
        engagement_scores = torch.sigmoid(self.engagement_projection(h)).squeeze(-1)  # [0, 1]
        
        # Discretize into bins
        risk_bins = (risk_scores * self.num_risk_bins).long()
        risk_bins = torch.clamp(risk_bins, 0, self.num_risk_bins - 1)
        
        engagement_bins = (engagement_scores * self.num_engagement_bins).long()
        engagement_bins = torch.clamp(engagement_bins, 0, self.num_engagement_bins - 1)
        
        # Compute base state IDs
        base_states = risk_bins * self.num_engagement_bins + engagement_bins
        
        # Classify transient states for high-risk students
        transient_logits = self.transient_classifier(h)
        transient_probs = F.softmax(transient_logits, dim=1)
        transient_states = torch.argmax(transient_probs, dim=1)
        
        # High risk threshold
        high_risk_mask = risk_scores > 0.75
        
        # Assign transient states to high-risk students
        final_states = base_states.clone()
        final_states[high_risk_mask] = self.num_base_states + transient_states[high_risk_mask]
        
        # Assign dropout state if known
        if dropout_labels is not None:
            dropout_mask = dropout_labels == 1
            final_states[dropout_mask] = self.dropout_state_id
        
        return final_states
    
    def get_continuous_scores(
        self,
        h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get continuous risk and engagement scores
        
        Args:
            h: Latent representations (batch_size, hidden_dim)
            
        Returns:
            Tuple of (risk_scores, engagement_scores)
        """
        risk_scores = torch.sigmoid(self.risk_projection(h)).squeeze(-1)
        engagement_scores = torch.sigmoid(self.engagement_projection(h)).squeeze(-1)
        
        return risk_scores, engagement_scores


if __name__ == "__main__":
    # Test LSTM encoder
    print("Testing LSTM Encoder...")
    lstm_encoder = LSTMEncoder(
        input_dim=64,
        hidden_dim=128,
        num_layers=2,
        dropout=0.3,
        bidirectional=True
    )
    
    x = torch.randn(32, 30, 64)  # (batch, seq_len, features)
    seq_out, final_hidden = lstm_encoder(x)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Sequence output shape: {seq_out.shape}")
    print(f"  Final hidden shape: {final_hidden.shape}")
    
    # Test Transformer encoder
    print("\nTesting Transformer Encoder...")
    transformer_encoder = TransformerEncoder(
        input_dim=64,
        d_model=128,
        nhead=8,
        num_encoder_layers=3,
        dim_feedforward=512,
        dropout=0.3
    )
    
    seq_out, pooled = transformer_encoder(x)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Sequence output shape: {seq_out.shape}")
    print(f"  Pooled output shape: {pooled.shape}")
    
    # Test State Discretizer
    print("\nTesting State Discretizer...")
    discretizer = StateDiscretizer(
        input_dim=256,
        num_risk_bins=4,
        num_engagement_bins=4,
        num_transient_states=7
    )
    
    h = torch.randn(32, 256)
    states = discretizer(h)
    risk, engagement = discretizer.get_continuous_scores(h)
    
    print(f"  Input shape: {h.shape}")
    print(f"  Discrete states shape: {states.shape}")
    print(f"  State range: [{states.min()}, {states.max()}]")
    print(f"  Risk scores shape: {risk.shape}")
    print(f"  Engagement scores shape: {engagement.shape}")
