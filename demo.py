import torch
import cv2
import numpy as np
from lib.config import cfg, update_config
from lib.models.pose_hrnet import get_pose_net
from lib.utils.transforms import get_affine_transform
import argparse

# Load configuration
config_file = 'experiments/deepfashion2/hrnet/w48_384x288_adam_lr1e-3.yaml'  # Path to config file
cfg.merge_from_file(config_file)
cfg.freeze()

# Update the model path and parameters if needed
model_path = 'model.pth'  # Path to the pretrained model

# Load the HRNet model
model = get_pose_net(cfg, is_train=False)
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model = model.double() 
model.eval()

def preprocess_image(image_path):
    # Load and preprocess the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or could not be loaded.")
    
    # Resize image according to model input size, which may vary (check config)
    input_size = cfg.MODEL.IMAGE_SIZE  # for example (384, 288)
    image_resized = cv2.resize(image, (input_size[1], input_size[0]))
    
    # Convert to tensor and normalize
    image_resized = image_resized.astype(np.float32) / 255.0
    image_resized = (image_resized - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    image_resized = image_resized.transpose(2, 0, 1)  # Change to (C, H, W)
    image_tensor = torch.from_numpy(image_resized).unsqueeze(0).double()  # Add batch dimension
    return image_tensor

def predict_landmarks(image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        # The output is usually heatmaps for each landmark point; postprocess to get coordinates
        heatmaps = outputs.squeeze(0).cpu().numpy()
        print(heatmaps.shape)
        
        # Find the landmark points from the heatmaps
        landmarks = []

        for i in range(heatmaps.shape[0]):
            heatmap = heatmaps[i]
            y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            landmarks.append((x, y))
        
        return landmarks

def main(image_path):
    # Preprocess the input image
    image_tensor = preprocess_image(image_path)
    
    # Get landmark predictions
    landmarks = predict_landmarks(image_tensor)
    
    print("Predicted landmarks:", landmarks)
    return landmarks

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='Path to the input image')
    args = parser.parse_args()
    main(args.image)
