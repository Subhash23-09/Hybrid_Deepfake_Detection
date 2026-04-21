import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class SpatialAnalysis(nn.Module):
    def __init__(self):
        super(SpatialAnalysis, self).__init__()
        # Load EfficientNet as the spatial backbone
        self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        # We extract features before the final classification head
        self.feature_extractor = nn.Sequential(*list(self.backbone.children())[:-1])
        # To strictly follow the flowchart, we have an intermediate "CNN Fake Probability" 
        self.cnn_prob_head = nn.Linear(1280, 2)

    def forward(self, x):
        # Extract deep spatial features
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1) # Flatten -> (Batch, 1280)
        # Calculate intermediate CNN Fake Probability branch score
        cnn_logits = self.cnn_prob_head(features)
        
        return features, cnn_logits

class FrequencyAnalysis(nn.Module):
    def __init__(self):
        super(FrequencyAnalysis, self).__init__()

    def forward(self, x):
        """
        Calculates Frequency Anomaly Scores using PyTorch FFT on GPU.
        Input x: (Batch, Channels, Height, Width)
        Output: 1D Frequency Anomaly Vector
        """
        # Grayscale approximation: Y = 0.2989 R + 0.5870 G + 0.1140 B
        gray = 0.2989 * x[:, 0, :, :] + 0.5870 * x[:, 1, :, :] + 0.1140 * x[:, 2, :, :]
        
        # Apply 2D FFT
        fft_2d = torch.fft.fft2(gray)
        fft_shift = torch.fft.fftshift(fft_2d)
        
        # Calculate magnitude spectrum logically
        magnitude = torch.log(torch.abs(fft_shift) + 1e-8)
        
        # We pool frequency bands
        batch_size, h, w = magnitude.shape
        center_h, center_w = h // 2, w // 2
        
        # Frequency Anomaly Score vector extraction
        # Focus on a specific band of central frequencies representing high-freq anomalies
        freq_features_x = torch.mean(magnitude[:, center_h - 50 : center_h + 50, :], dim=1)
        freq_features_y = torch.mean(magnitude[:, :, center_w - 50 : center_w + 50], dim=2)
        
        # Frequency Anomaly Score vector (combining structural X and Y frequency anomalies)
        freq_anomaly_score = torch.cat([freq_features_x, freq_features_y], dim=1)
        return freq_anomaly_score

class MultiDomainFusion(nn.Module):
    def __init__(self):
        super(MultiDomainFusion, self).__init__()
        
        self.spatial = SpatialAnalysis()
        self.frequency = FrequencyAnalysis()
        
        # Frequency vector length: (W + H) -> 224 + 224 = 448
        # Spatial vector length: 1280
        # Total concatenated features: 1280 + 448 = 1728
        
        self.fusion_network = nn.Sequential(
            nn.Linear(1728, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 2) # Final Classification (Real / Fake)
        )

    def forward(self, x):
        """
        Input x: images (Batch, 3, 224, 224)
        """
        # 1. Spatial Analysis -> CNN Probability & Features
        spatial_features, cnn_logits = self.spatial(x)
        
        # 2. Frequency Analysis -> Frequency Anomaly Score
        freq_anomaly_score = self.frequency(x)
        
        # 3. Multi-Domain Fusion
        combined_features = torch.cat([spatial_features, freq_anomaly_score], dim=1)
        final_classification = self.fusion_network(combined_features)
        
        # We return both the final prediction and the intermediate cnn logits 
        # because the flowchart maps CNN Fake Probability explicitly 
        # and it can be useful for regularized loss calculation
        return cnn_logits, final_classification

if __name__ == "__main__":
    dummy_input = torch.randn(8, 3, 224, 224)
    model = MultiDomainFusion()
    cnn_l, output = model(dummy_input)
    print("Multi-Domain Fusion initialized. Spatial + Frequency merged!")
    print(f"CNN Logits: {cnn_l.shape}, Final Output: {output.shape}")
