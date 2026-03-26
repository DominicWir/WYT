"""
Training Script for xLSTM Sensor Fusion Model

This script handles model training with multi-task learning for position
estimation and activity classification.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Optional, Tuple
import os
import json
from datetime import datetime
from tqdm import tqdm

from xlstm_model import SensorFusionXLSTM, create_model
from data_preprocessing import create_dataloaders, create_train_val_split
from feature_extraction import SensorFeatureExtractor


class MultiTaskLoss(nn.Module):
    """
    Combined loss for multi-task learning (position + activity).
    """
    
    def __init__(
        self,
        position_weight: float = 1.0,
        activity_weight: float = 1.0,
        use_uncertainty: bool = False
    ):
        super().__init__()
        self.position_weight = position_weight
        self.activity_weight = activity_weight
        self.use_uncertainty = use_uncertainty
        
        self.position_loss = nn.MSELoss()
        self.activity_loss = nn.CrossEntropyLoss()
    
    def forward(
        self,
        position_pred: torch.Tensor,
        position_true: torch.Tensor,
        activity_pred: torch.Tensor,
        activity_true: torch.Tensor,
        uncertainty: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss.
        
        Args:
            position_pred: Predicted positions (batch, seq, 3)
            position_true: True positions (batch, seq, 3)
            activity_pred: Activity logits (batch, seq, n_classes)
            activity_true: True activity labels (batch, seq)
            uncertainty: Optional uncertainty estimates (batch, seq, 3)
        
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Position loss (MSE)
        if self.use_uncertainty and uncertainty is not None:
            # Uncertainty-weighted loss
            pos_loss = torch.mean(
                ((position_pred - position_true) ** 2) / (2 * uncertainty ** 2) +
                torch.log(uncertainty)
            )
        else:
            pos_loss = self.position_loss(position_pred, position_true)
        
        # Activity loss (Cross-entropy)
        # Reshape for cross-entropy: (batch * seq, n_classes)
        batch_size, seq_len, n_classes = activity_pred.shape
        activity_pred_flat = activity_pred.reshape(-1, n_classes)
        activity_true_flat = activity_true.reshape(-1)
        
        act_loss = self.activity_loss(activity_pred_flat, activity_true_flat)
        
        # Combined loss
        total_loss = (
            self.position_weight * pos_loss +
            self.activity_weight * act_loss
        )
        
        loss_dict = {
            'total': total_loss.item(),
            'position': pos_loss.item(),
            'activity': act_loss.item()
        }
        
        return total_loss, loss_dict


class Trainer:
    """
    Trainer for xLSTM sensor fusion model.
    """
    
    def __init__(
        self,
        model: SensorFusionXLSTM,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: str = None
    ):
        # Auto-detect best available device
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        
        print(f"Training on device: {device}")
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Loss function
        self.criterion = MultiTaskLoss(
            position_weight=config.get('position_weight', 1.0),
            activity_weight=config.get('activity_weight', 1.0),
            use_uncertainty=config.get('use_uncertainty', False)
        )
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.get('learning_rate', 1e-3),
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_position_loss': [],
            'val_position_loss': [],
            'train_activity_loss': [],
            'val_activity_loss': []
        }
        
        # Create checkpoint directory
        self.checkpoint_dir = config.get('checkpoint_dir', 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_losses = {'total': 0, 'position': 0, 'activity': 0}
        n_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.epoch + 1} [Train]')
        
        for batch in pbar:
            features = batch['features'].to(self.device)
            positions = batch['positions'].to(self.device)
            activities = batch['activities'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            pos_pred, act_pred, uncertainty = self.model(
                features,
                return_uncertainty=self.config.get('use_uncertainty', False)
            )
            
            # Compute loss
            loss, loss_dict = self.criterion(
                pos_pred, positions, act_pred, activities, uncertainty
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['grad_clip']
                )
            
            self.optimizer.step()
            
            # Accumulate losses
            for key in total_losses:
                total_losses[key] += loss_dict[key]
            n_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': f"{loss_dict['total']:.4f}"})
        
        # Average losses
        avg_losses = {k: v / n_batches for k, v in total_losses.items()}
        return avg_losses
    
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        
        total_losses = {'total': 0, 'position': 0, 'activity': 0}
        n_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {self.epoch + 1} [Val]')
            
            for batch in pbar:
                features = batch['features'].to(self.device)
                positions = batch['positions'].to(self.device)
                activities = batch['activities'].to(self.device)
                
                # Forward pass
                pos_pred, act_pred, uncertainty = self.model(
                    features,
                    return_uncertainty=self.config.get('use_uncertainty', False)
                )
                
                # Compute loss
                loss, loss_dict = self.criterion(
                    pos_pred, positions, act_pred, activities, uncertainty
                )
                
                # Accumulate losses
                for key in total_losses:
                    total_losses[key] += loss_dict[key]
                n_batches += 1
                
                pbar.set_postfix({'loss': f"{loss_dict['total']:.4f}"})
        
        # Average losses
        avg_losses = {k: v / n_batches for k, v in total_losses.items()}
        return avg_losses
    
    def train(self, num_epochs: int):
        """Train the model for multiple epochs."""
        print(f"\nTraining on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}\n")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Train
            train_losses = self.train_epoch()
            
            # Validate
            val_losses = self.validate()
            
            # Update history
            self.history['train_loss'].append(train_losses['total'])
            self.history['val_loss'].append(val_losses['total'])
            self.history['train_position_loss'].append(train_losses['position'])
            self.history['val_position_loss'].append(val_losses['position'])
            self.history['train_activity_loss'].append(train_losses['activity'])
            self.history['val_activity_loss'].append(val_losses['activity'])
            
            # Learning rate scheduling
            self.scheduler.step(val_losses['total'])
            
            # Print epoch summary
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"  Train Loss: {train_losses['total']:.4f} "
                  f"(Pos: {train_losses['position']:.4f}, "
                  f"Act: {train_losses['activity']:.4f})")
            print(f"  Val Loss:   {val_losses['total']:.4f} "
                  f"(Pos: {val_losses['position']:.4f}, "
                  f"Act: {val_losses['activity']:.4f})")
            
            # Save best model
            if val_losses['total'] < self.best_val_loss:
                self.best_val_loss = val_losses['total']
                self.save_checkpoint('best_model.pt')
                print(f"  ✓ Saved best model (val_loss: {self.best_val_loss:.4f})")
            
            # Save periodic checkpoint
            if (epoch + 1) % self.config.get('save_every', 10) == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pt')
        
        print("\n✓ Training completed!")
        self.save_history()
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history,
            'config': self.config
        }
        
        filepath = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        filepath = os.path.join(self.checkpoint_dir, filename)
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']
    
    def save_history(self):
        """Save training history."""
        filepath = os.path.join(self.checkpoint_dir, 'training_history.json')
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)


if __name__ == "__main__":
    print("=== xLSTM Training Demo ===\n")
    
    # Configuration
    config = {
        'feature_dim': 70,
        'hidden_size': 128,
        'num_layers': 2,
        'batch_size': 16,
        'sequence_length': 20,
        'learning_rate': 1e-3,
        'weight_decay': 1e-5,
        'num_epochs': 5,
        'position_weight': 1.0,
        'activity_weight': 0.5,
        'use_uncertainty': False,
        'grad_clip': 1.0,
        'save_every': 2,
        'checkpoint_dir': 'checkpoints'
    }
    
    # Create dummy data
    n_samples = 500
    features = np.random.randn(n_samples, config['feature_dim'])
    positions = np.random.randn(n_samples, 3)
    activities = np.random.randint(0, 5, n_samples)
    
    # Split data
    train_data, val_data = create_train_val_split(
        features, positions, activities, val_ratio=0.2
    )
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_data, val_data,
        batch_size=config['batch_size'],
        sequence_length=config['sequence_length']
    )
    
    # Create model
    model = create_model(
        feature_dim=config['feature_dim'],
        config={
            'hidden_size': config['hidden_size'],
            'num_layers': config['num_layers']
        }
    )
    
    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, config)
    
    # Train
    trainer.train(num_epochs=config['num_epochs'])
    
    print("\n✓ Demo completed!")
