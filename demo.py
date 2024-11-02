import torch
import cv2
import numpy as np
from lib.config import cfg, update_config
from lib.models.pose_hrnet import get_pose_net
from lib.utils.transforms import get_affine_transform
import argparse

# Load configuration
config_file = 'experiments/deepfashion2/hrnet/w48_384x288_adam_lr1e-3.yaml'
cfg.merge_from_file(config_file)

# Modify the configuration for 600x600 output
cfg.defrost()
cfg.MODEL.IMAGE_SIZE = [600, 600]
cfg.MODEL.HEATMAP_SIZE = [600, 600]
cfg.freeze()

def modify_model_for_size(model):
    """Modify the model's final layers to output 600x600"""
    # Add an upsampling layer after the final layer
    class ModifiedHRNet(torch.nn.Module):
        def __init__(self, base_model):
            super(ModifiedHRNet, self).__init__()
            self.base_model = base_model
            self.upsample = torch.nn.Upsample(
                size=(600, 600),
                mode='bilinear',
                align_corners=True
            )

        def forward(self, x):
            x = self.base_model(x)
            x = self.upsample(x)
            return x

    return ModifiedHRNet(model)

# Load and modify the model
model = get_pose_net(cfg, is_train=False)
model = modify_model_for_size(model)

# Load pretrained weights
model.base_model.load_state_dict(torch.load("model.pth", map_location='cpu'))
model = model.double()
model.eval()

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or could not be loaded.")
    
    # Resize to 600x600
    image_resized = cv2.resize(image, (600, 600))
    
    # Convert to tensor and normalize
    image_resized = image_resized.astype(np.float32) / 255.0
    image_resized = (image_resized - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    image_resized = image_resized.transpose(2, 0, 1)
    image_tensor = torch.from_numpy(image_resized).unsqueeze(0).double()
    return image_tensor

def predict_landmarks(image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        heatmaps = outputs.squeeze(0).cpu().numpy()
        print("Output shape:", heatmaps.shape)  # Should be (294, 600, 600)
        
        landmarks = []
        for i in range(heatmaps.shape[0]):
            heatmap = heatmaps[i]
            y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            landmarks.append((x, y))
        
        return landmarks

def main(image_path):
    image_tensor = preprocess_image(image_path)
    landmarks = predict_landmarks(image_tensor)
    print("Predicted landmarks:", landmarks)
    return landmarks

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='Path to the input image')
    args = parser.parse_args()
    main(args.image)