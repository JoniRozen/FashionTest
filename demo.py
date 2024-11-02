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
cfg.defrost()  # Unfreeze config to modify it
cfg.MODEL.IMAGE_SIZE = [600, 600]  # Set new image size
cfg.MODEL.EXTRA.STAGE4.NUM_CHANNELS = [48, 96, 192, 384]  # Adjust channels if needed
cfg.MODEL.EXTRA.FINAL_CONV_KERNEL = 1  # Ensure final conv kernel is 1x1
cfg.MODEL.HEATMAP_SIZE = [600, 600]  # Set heatmap size to match desired output
cfg.freeze()

# Load the HRNet model
model_path = 'model.pth'  # Path to the pretrained model

def modify_model_for_size(model):
    """Modify the model's final layers to output 600x600"""
    # Get the last deconv layer
    last_layer = list(model.final_layer.children())[-1]
    
    # Create new final layer with same channels but different output size
    new_final_layer = torch.nn.Sequential(
        # Upsampling to reach 600x600
        torch.nn.ConvTranspose2d(
            in_channels=last_layer.in_channels,
            out_channels=last_layer.out_channels,
            kernel_size=4,  # Adjust kernel size for upsampling
            stride=2,
            padding=1,
            output_padding=0,
            bias=False
        ),
        torch.nn.BatchNorm2d(last_layer.out_channels),
        torch.nn.ReLU(inplace=True),
        # Final 1x1 conv to maintain channel dimension
        torch.nn.Conv2d(
            in_channels=last_layer.out_channels,
            out_channels=last_layer.out_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
    )
    
    # Replace the final layer
    model.final_layer = new_final_layer
    return model

# Load and modify the model
model = get_pose_net(cfg, is_train=False)
model = modify_model_for_size(model)

# Load pretrained weights and adjust for modified architecture
pretrained_dict = torch.load(model_path, map_location='cpu')
model_dict = model.state_dict()

# Filter out incompatible keys
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict, strict=False)

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
            # Scale coordinates to original image size if needed
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