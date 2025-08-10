import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MyResidualBlock(nn.Module):
    def __init__(self, downsample):
        super(MyResidualBlock, self).__init__()
        self.downsample = downsample
        self.stride = 2 if self.downsample else 1
        K = 9
        P = (K - 1) // 2

        self.conv1 = nn.Conv2d(256, 256, kernel_size=(1, K), stride=(1, self.stride), padding=(0, P), bias=False)
        self.bn1 = nn.BatchNorm2d(256)

        self.conv2 = nn.Conv2d(256, 256, kernel_size=(1, K), padding=(0, P), bias=False)
        self.bn2 = nn.BatchNorm2d(256)

        if self.downsample:
            self.idfunc_0 = nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2))
            self.idfunc_1 = nn.Conv2d(256, 256, kernel_size=(1, 1), bias=False)

    def forward(self, x):
        identity = x
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        if self.downsample:
            identity = self.idfunc_0(identity)
            identity = self.idfunc_1(identity)
        x += identity
        return x


class EnhancedNN(nn.Module):
    def __init__(self, nOUT):
        super(EnhancedNN, self).__init__()
        # Main branch - using all leads (similar to original architecture)
        self.conv = nn.Conv2d(in_channels=12, out_channels=256, kernel_size=(1, 15), padding=(0, 7), stride=(1, 2), bias=False)
        self.bn = nn.BatchNorm2d(256)

        # Keep original 5 residual blocks from the original architecture
        self.rb_0 = MyResidualBlock(downsample=True)
        self.rb_1 = MyResidualBlock(downsample=True)
        self.rb_2 = MyResidualBlock(downsample=True) 
        self.rb_3 = MyResidualBlock(downsample=True)
        self.rb_4 = MyResidualBlock(downsample=True)
        
        self.mha = nn.MultiheadAttention(256, 8)
        self.pool = nn.AdaptiveMaxPool1d(output_size=1)
        
        # AFlutter-specific branch - we'll keep this from the original model
        # but focus it on AFlutter identification
        self.flutter_conv = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=15, padding=7, stride=2)
        self.flutter_bn = nn.BatchNorm1d(64)
        self.flutter_pool = nn.AdaptiveMaxPool1d(output_size=64)  # Reduced from 128 to 64
        
        # PVC-specific branch - adding a dedicated path for PVC detection
        self.pvc_conv = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=9, padding=4, stride=2)
        self.pvc_bn = nn.BatchNorm1d(64)
        self.pvc_pool = nn.AdaptiveMaxPool1d(output_size=32)  # Smaller size for beat-focused features
        
        # Frequency domain processing for AFlutter
        self.freq_fc = nn.Linear(256, 32)
        
        # Combined features processing
        self.fc_1 = nn.Linear(256 + 12 + 32 + 64 + 32, nOUT)  # Main + leads + freq + flutter + pvc features

    def compute_fft_features(self, lead_data, device):
        """Compute frequency domain features from lead data with broader frequency range (3-12 Hz)."""
        batch_size = lead_data.size(0)
        fft_features = torch.zeros(batch_size, 256, device=device)
        
        for i in range(batch_size):
            # Get the lead data for this sample
            sample_data = lead_data[i, 0, :].cpu().detach().numpy()
            
            # Apply FFT
            fft_result = np.abs(np.fft.rfft(sample_data))
            
            # Adjust for broader frequency range (3-12 Hz)
            # For 500 Hz sampling rate and 8192 samples:
            # 3 Hz corresponds to bin ~50
            # 12 Hz corresponds to bin ~197
            # Use normalized values to make learning more stable
            relevant_fft = fft_result[50:306] if len(fft_result) >= 306 else np.pad(fft_result, (0, max(0, 306-len(fft_result))))[50:306]
            
            # Normalize FFT features for better learning
            if np.max(relevant_fft) > 0:
                relevant_fft = relevant_fft / np.max(relevant_fft)
                
            fft_features[i, :len(relevant_fft)] = torch.tensor(relevant_fft, device=device)
            
        return fft_features

    def forward(self, x, l, beat_locs=None):
        device = x.device
        
        # Main branch processing (all leads)
        x_main = F.leaky_relu(self.bn(self.conv(x)))
        x_main = self.rb_0(x_main)
        x_main = self.rb_1(x_main)
        x_main = self.rb_2(x_main)
        x_main = self.rb_3(x_main) 
        x_main = self.rb_4(x_main)
        
        x_main = F.dropout(x_main, p=0.5, training=self.training)
        x_main = x_main.squeeze(2).permute(2, 0, 1)  # [seq, batch, features]
        x_main, _ = self.mha(x_main, x_main, x_main)
        x_main = x_main.permute(1, 2, 0)
        x_main = self.pool(x_main).squeeze(2)
        
        # Extract Lead II for rhythm analysis
        lead_II = x[:, 1:2, 0, :]  # Extract Lead II [B, 1, 8192]
        
        # AFlutter-specific branch
        x_flutter = F.leaky_relu(self.flutter_bn(self.flutter_conv(lead_II)))
        x_flutter = self.flutter_pool(x_flutter)  # [B, 64, 64]
        x_flutter = x_flutter.reshape(x_flutter.size(0), -1)  # Flatten to [B, 4096]
        x_flutter = F.dropout(x_flutter, p=0.3, training=self.training)
        
        # PVC-specific branch - focus on beat morphology
        x_pvc = F.leaky_relu(self.pvc_bn(self.pvc_conv(lead_II)))
        x_pvc = self.pvc_pool(x_pvc)  # [B, 64, 32]
        x_pvc = x_pvc.reshape(x_pvc.size(0), -1)  # Flatten to [B, 2048]
        x_pvc = F.dropout(x_pvc, p=0.3, training=self.training)
        
        # Apply dimensionality reduction for flutter and PVC features
        x_flutter = F.leaky_relu(torch.nn.functional.linear(x_flutter, 
                                                         torch.randn(64, x_flutter.size(1), device=device) / np.sqrt(x_flutter.size(1))))
        x_pvc = F.leaky_relu(torch.nn.functional.linear(x_pvc, 
                                                    torch.randn(32, x_pvc.size(1), device=device) / np.sqrt(x_pvc.size(1))))
        
        # Frequency domain features for AFlutter
        fft_features = self.compute_fft_features(lead_II, device)
        freq_features = F.leaky_relu(self.freq_fc(fft_features))
        
        # Combine features from all branches
        combined_features = torch.cat((x_main, l, freq_features, x_flutter, x_pvc), dim=1)
        
        # Final classification
        logits = self.fc_1(combined_features)
        return logits, torch.sigmoid(logits)