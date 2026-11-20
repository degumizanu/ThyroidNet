import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os
import glob
import cv2
from tqdm import tqdm
from model import ThyroidNet
import numpy as np
from scipy.spatial import cKDTree
import os
import argparse
import sys
# ==========================================
# 1. METRIC FUNCTIONS
# ==========================================

def calculate_dice_iou(pred_mask, gt_mask):
    """
    Input: Boolean or Binary Numpy Arrays (0s and 1s)
    Returns: Dice Score, IoU Score
    """
    # Flatten
    pred_flat = pred_mask.flatten()
    gt_flat = gt_mask.flatten()
    
    intersection = np.sum(pred_flat * gt_flat)
    sum_pred = np.sum(pred_flat)
    sum_gt = np.sum(gt_flat)
    
    # Dice: 2*Intersection / (Sum A + Sum B)
    dice = (2.0 * intersection) / (sum_pred + sum_gt + 1e-6)
    
    # IoU: Intersection / Union
    union = sum_pred + sum_gt - intersection
    iou = intersection / (union + 1e-6)
    
    return dice, iou
def calculate_sens_spec(pred_mask, gt_mask):
    """
    Input: Binary Numpy Arrays (0s and 1s)
    Returns: Sensitivity, Specificity
    """
    # Flatten to ensure 1D arrays
    pred_flat = pred_mask.flatten()
    gt_flat = gt_mask.flatten()
    
    # 1. True Positives (TP): Pred=1 AND GT=1
    tp = np.sum(pred_flat * gt_flat)
    
    # 2. True Negatives (TN): Pred=0 AND GT=0
    # Calculated by inverting the masks (1 - mask)
    tn = np.sum((1.0 - pred_flat) * (1.0 - gt_flat))
    
    # 3. False Positives (FP): Pred=1 BUT GT=0
    fp = np.sum(pred_flat * (1.0 - gt_flat))
    
    # 4. False Negatives (FN): Pred=0 BUT GT=1
    fn = np.sum((1.0 - pred_flat) * gt_flat)
    
    # Sensitivity (Recall): TP / (TP + FN)
    sensitivity = tp / (tp + fn + 1e-6)
    
    # Specificity: TN / (TN + FP)
    specificity = tn / (tn + fp + 1e-6)
    
    return sensitivity, specificity




def calculate_hd95(pred_mask, gt_mask):
    if np.sum(pred_mask) == 0 or np.sum(gt_mask) == 0:
        return np.nan

    pred_coords = np.argwhere(pred_mask)
    gt_coords = np.argwhere(gt_mask)

    # Nearest-neighbor distances
    tree_gt = cKDTree(gt_coords)
    tree_pred = cKDTree(pred_coords)

    d_pred_to_gt, _ = tree_gt.query(pred_coords, k=1)
    d_gt_to_pred, _ = tree_pred.query(gt_coords, k=1)

    all_distances = np.concatenate([d_pred_to_gt, d_gt_to_pred])

    return np.percentile(all_distances, 95)


# ==========================================
# 2. EVALUATION LOOP
# ==========================================

def evaluate_test_set(image_dir, mask_dir, model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on {device}...")
    
    # 1. Load Model
    # Ensure 'pretrained=False' because we are loading your custom weights anyway
    model = ThyroidNet(256, 256, pretrained=False).to(device)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded successfully.")
    else:
        print(f"Error: Model file {model_path} not found.")
        return

    model.eval()
    
    # 2. Get Files
    img_files = sorted(glob.glob(os.path.join(image_dir, "*.png")) + glob.glob(os.path.join(image_dir, "*.PNG")) + glob.glob(os.path.join(image_dir, "*.jpg")))
    mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.png")) + glob.glob(os.path.join(mask_dir, "*.PNG")) + glob.glob(os.path.join(mask_dir, "*.jpg")))
    
    if len(img_files) == 0:
        print("Error: No images found in directory.")
        return
    if len(img_files) != len(mask_files):
        print(f"Warning: Mismatch in file counts! Images: {len(img_files)}, Masks: {len(mask_files)}")
        # We will truncate to the shorter list for safety
        limit = min(len(img_files), len(mask_files))
        img_files = img_files[:limit]
        mask_files = mask_files[:limit]

    # 3. Storage for Metrics
    dice_scores = []
    iou_scores = []
    hd_scores = []
    spec_score = []
    sens_score = []
    
    print("-" * 60)
    print(f"{'Image Name':<30} | {'Dice':<8} | {'IoU':<8} | {'HD (px)':<8}")
    print("-" * 60)

    # 4. Loop
    with torch.no_grad():
        for i in tqdm(range(len(img_files)), desc="Processing"):
            img_path = img_files[i]
            mask_path = mask_files[i]
            fname = os.path.basename(img_path)
            
            # --- Preprocessing ---
            # Image
            img_raw = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img_resized = cv2.resize(img_raw, (256, 256))
            img_tensor = torch.from_numpy(img_resized / 255.0).float().unsqueeze(0).unsqueeze(0).to(device)
            
            # Mask (Ground Truth)
            mask_raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            # mask_raw = np.array(Image.open(mask_path).convert('L'))
            mask_resized = cv2.resize(mask_raw, (256, 256), interpolation=cv2.INTER_NEAREST)
            # Ensure binary (0 or 1)
            gt_mask = (mask_resized > 127).astype(np.uint8)
            # gt_mask = mask_resized
            
            # --- Inference ---
            # Forward pass: S_map, alpha_map, gain, mask_logits
            _, _, _, mask_logits, _ = model(img_tensor)
            
            probs = torch.sigmoid(mask_logits)
            pred_mask = (probs.squeeze().cpu().numpy() > 0.5).astype(np.uint8)

            #saving predicted mask
            
            
            cv2.imwrite(f"predicted_masks{fname}", pred_mask)


            
            
            # --- Calculate Metrics ---
            dice, iou = calculate_dice_iou(pred_mask, gt_mask)
            hd = calculate_hd95(pred_mask, gt_mask)
            sens,spec = calculate_sens_spec(pred_mask, gt_mask)
            
            # Store
            dice_scores.append(dice)
            iou_scores.append(iou)
            sens_score.append(sens)
            spec_score.append(spec)
            if not np.isnan(hd):
                hd_scores.append(hd)
            
            # Print row (optional, comment out if too much spam)
            # tqdm.write(f"{fname:<30} | {dice:.4f}   | {iou:.4f}   | {hd if not np.isnan(hd) else 'NaN':.2f}")

    # ==========================================
    # 3. FINAL SUMMARY
    # ==========================================
    avg_dice = np.mean(dice_scores)
    std_dice = np.std(dice_scores)
    
    avg_iou = np.mean(iou_scores)
    std_iou = np.std(iou_scores)

    
    avg_hd = np.mean(hd_scores) if len(hd_scores) > 0 else 0.0
    std_hd = np.std(hd_scores) if len(hd_scores) > 0 else 0.0

    avg_spec = np.mean(spec_score)
    std_spec = np.std(spec_score)

    avg_sens = np.mean(sens_score)
    std_sens = np.std(sens_score)

    print("\n" + "="*60)
    print("FINAL TEST RESULTS")
    print("="*60)
    print(f"Dataset Size:      {len(img_files)} images")
    print(f"Dice Score:        {avg_dice:.4f} ± {std_dice:.4f}")
    print(f"IoU Score:         {avg_iou:.4f} ± {std_iou:.4f}")
    print(f"Hausdorff Dist:    {avg_hd:.2f} px ± {std_hd:.2f}")
    print(f"Specificity: {avg_spec:.2f} ± {std_spec:.2f}")
    print(f"Sensetivity: {avg_sens:.2f} ±{std_sens:.2f}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to test dataset")
    args = parser.parse_args()

    data_dir = args.data

    TEST_IMAGES = os.path.join(data_dir, "images")
    TEST_MASKS = os.path.join(data_dir,"masks")
    MODEL_FILE = "best_model.pth"

    if not os.path.exists(TEST_IMAGES) or not os.path.exists(TEST_MASKS):
        print("Please provide valid data directory with images and masks folder in it.")
        sys.exit()
        

    evaluate_test_set(TEST_IMAGES, TEST_MASKS, MODEL_FILE)

if __name__ == "__main__":
    main()   