import torch
import cv2
import numpy as np
from lib.config import cfg, update_config
from lib.models.pose_hrnet import get_pose_net
from lib.utils.transforms import get_affine_transform
import argparse
import torch.nn.functional as F

# Load configuration
config_file = 'experiments/deepfashion2/hrnet/w48_384x288_adam_lr1e-3.yaml'
cfg.merge_from_file(config_file)

# Keep original image size in config
cfg.defrost()
cfg.MODEL.IMAGE_SIZE = [384, 288]  # Keep original size
cfg.freeze()

# Load the HRNet model
model_path = 'model.pth'
model = get_pose_net(cfg, is_train=False)
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model = model.double()
model.eval()

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or could not be loaded.")
    
    # First resize to model's expected input size
    image_resized = cv2.resize(image, (cfg.MODEL.IMAGE_SIZE[1], cfg.MODEL.IMAGE_SIZE[0]))
    
    # Convert to tensor and normalize
    image_resized = image_resized.astype(np.float32) / 255.0
    image_resized = (image_resized - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    image_resized = image_resized.transpose(2, 0, 1)
    image_tensor = torch.from_numpy(image_resized).unsqueeze(0).double()
    return image_tensor, image.shape

def predict_landmarks(image_tensor, original_shape):
    with torch.no_grad():
        # Get model output at original size
        outputs = model(image_tensor)
        
        # Resize output to 600x600
        outputs_resized = F.interpolate(
            outputs,
            size=(600, 600),
            mode='bilinear',
            align_corners=True
        )
        
        heatmaps = outputs_resized.squeeze(0).cpu().numpy()
        print("Output shape:", heatmaps.shape)  # Should be (294, 600, 600)
        
        landmarks = []
        for i in range(heatmaps.shape[0]):
            heatmap = heatmaps[i]
            y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            
            # Scale coordinates to original image size if needed
            orig_h, orig_w = original_shape[:2]
            x = int(x * (orig_w / 600))
            y = int(y * (orig_h / 600))
            
            landmarks.append((x, y))
        
        return landmarks, heatmaps

def save_heatmaps(heatmaps, save_path='heatmaps'):
    """Save heatmaps for visualization"""
    import os
    os.makedirs(save_path, exist_ok=True)
    
    for i in range(min(10, heatmaps.shape[0])):  # Save first 10 heatmaps
        heatmap = heatmaps[i]
        heatmap = ((heatmap - heatmap.min()) * 255 / 
                   (heatmap.max() - heatmap.min())).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(save_path, f'heatmap_{i}.jpg'), heatmap_color)

def main(image_path):
    # Preprocess the input image
    image_tensor, original_shape = preprocess_image(image_path)
    
    # Get landmark predictions and heatmaps
    landmarks, heatmaps = predict_landmarks(image_tensor, original_shape)
    
    # Save some heatmaps for visualization
    save_heatmaps(heatmaps)
    
    # Load original image and draw landmarks
    image = cv2.imread(image_path)
    for x, y in landmarks:
        cv2.circle(image, (x, y), 3, (0, 255, 0), -1)
    
    # Save result
    cv2.imwrite('result.jpg', image)
    
    print("Predicted landmarks:", landmarks)
    print("Results saved: result.jpg and heatmaps/*.jpg")
    return landmarks

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='Path to the input image')
    args = parser.parse_args()
    main(args.image)