import os
import io
import base64
import torch
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from typing import Optional
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI
app = FastAPI(title="Flood Detection API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for local testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model globally
print("Loading model...")
model = smp.Unet(
    encoder_name="efficientnet-b3", 
    encoder_weights=None, # We load our own weights
    in_channels=3,                  
    classes=1,                      
    activation=None
)

# Load weights
weights_path = 'best_efficientnet_b3.pth'
if os.path.exists(weights_path):
    model.load_state_dict(torch.load(weights_path, map_location=device))
    print(f"Model weights loaded from {weights_path}")
else:
    print(f"WARNING: Model weights not found at {weights_path}. Running with uninitialized weights.")

model.to(device)
model.eval()

# Preprocessing transforms
preprocess = A.Compose([
    A.Resize(512, 512),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

def calculate_metrics(pred, target):
    """Calculates Dice, IoU, Precision, and Recall manually for the batch of 1."""
    pred = pred.view(-1)
    target = target.view(-1)
    
    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    fn = ((1 - pred) * target).sum()
    
    smooth = 1e-6
    precision = tp / (tp + fp + smooth)
    recall = tp / (tp + fn + smooth)
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return {
        "dice": dice.item(),
        "iou": iou.item(),
        "precision": precision.item(),
        "recall": recall.item()
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...), mask_file: Optional[UploadFile] = File(None)):
    try:
        # Read the image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Original size for resizing the mask back later
        original_height, original_width = img.shape[:2]
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Preprocess Image
        augmented = preprocess(image=img_rgb)
        input_tensor = augmented['image'].unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            output = model(input_tensor)
            
            # Postprocessing (Sigmoid + Threshold)
            pred_tensor = (torch.sigmoid(output) > 0.5).float()
            
            # If ground truth mask is provided, calculate real-time metrics
            metrics = None
            if mask_file is not None:
                mask_contents = await mask_file.read()
                if mask_contents: # Check if file is not empty
                    mask_nparr = np.frombuffer(mask_contents, np.uint8)
                    gt_mask = cv2.imdecode(mask_nparr, cv2.IMREAD_GRAYSCALE)
                    
                    # Resize mask to exactly match the model's 512x512 output shape
                    gt_mask_resized = cv2.resize(gt_mask, (512, 512), interpolation=cv2.INTER_NEAREST)
                    gt_mask_binary = (gt_mask_resized > 0).astype(np.float32)
                    
                    # Convert to tensor to compare
                    gt_tensor = torch.from_numpy(gt_mask_binary).unsqueeze(0).unsqueeze(0).to(device)
                    
                    # Calculate
                    metrics = calculate_metrics(pred_tensor, gt_tensor)
            
            # Convert prediction back to numpy array [H, W]
            mask_np = pred_tensor.squeeze().cpu().numpy()
            
            # Resize mask back to original image size for display
            mask_resized = cv2.resize(mask_np, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
            
            # Create a blue overlay for the mask
            overlay = np.zeros((original_height, original_width, 4), dtype=np.uint8)
            overlay[mask_resized == 1] = [255, 150, 0, 150] # Blue color, semi-transparent
            
            # Encode overlay to base64 PNG
            _, buffer = cv2.imencode('.png', overlay)
            mask_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Encode raw predicted mask to base64
            raw_mask = (mask_resized * 255).astype(np.uint8)
            _, raw_buffer = cv2.imencode('.png', raw_mask)
            raw_mask_base64 = base64.b64encode(raw_buffer).decode('utf-8')
            
            response_data = {
                "mask": mask_base64,
                "raw_mask": raw_mask_base64
            }
            
            if metrics:
                response_data["metrics"] = metrics
                # Also encode the ground truth mask for comparison
                gt_mask_display = cv2.resize(gt_mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
                gt_mask_display = (gt_mask_display > 0).astype(np.uint8) * 255
                _, gt_buffer = cv2.imencode('.png', gt_mask_display)
                response_data["gt_mask"] = base64.b64encode(gt_buffer).decode('utf-8')
                
            return JSONResponse(content=response_data)
            
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# Serve static files for the frontend
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
