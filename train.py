import torch
from torch.utils.data import DataLoader, random_split
import glob
from tqdm import tqdm
from model import ThyroidNet
from loss import Loss
import torch.nn as nn
import sys
from dataset import NoduleDataset
import argparse
import os
# --- 2. Metric Helper ---
def calculate_metrics(pred_logits, target_mask, threshold=0.5):
    probs = torch.sigmoid(pred_logits)
    pred_mask = (probs > threshold).float()
    
    pred_flat = pred_mask.view(-1)
    target_flat = target_mask.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    dice = (2. * intersection) / (pred_flat.sum() + target_flat.sum() + 1e-6)
    
    union = pred_flat.sum() + target_flat.sum() - intersection
    iou = intersection / (union + 1e-6)
    
    return dice.item(), iou.item()

# --- 3. Evaluation Function (With Progress Bar) ---
def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0
    total_dice = 0
    total_iou = 0
    
    # Wrap loader with tqdm for a progress bar
    # leave=False means the bar disappears after it's done to keep the terminal clean
    loop = tqdm(loader, desc="Validating", leave=False)
    
    with torch.no_grad():
        for imgs, masks in loop:
            imgs, masks = imgs.to(device), masks.to(device)
            
            S_map, alpha_map, gain_vec, mask_logits, gland_map = model(imgs)
            
            loss= criterion(S_map, alpha_map, gain_vec, mask_logits, imgs, masks, gland_map)
            total_loss += loss.item()
            
            dice, iou = calculate_metrics(mask_logits, masks)
            total_dice += dice
            total_iou += iou
            
    avg_loss = total_loss / len(loader)
    avg_dice = total_dice / len(loader)
    avg_iou = total_iou / len(loader)
    
    return avg_loss, avg_dice, avg_iou

# --- 4. Main Training Loop (With Progress Bar) ---
def train(data_dir):

    BATCH_SIZE = 16
    LR = 1e-4
    EPOCHS = 200
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Starting training on {DEVICE}...")

    # Data Loaders
    
    dataset = NoduleDataset(data_dir)
    train_dataset, val_dataset = random_split(dataset, (0.8, 0.2))

    print(f"LENGTH OF THE DATASET:{len(dataset)}")
    print(f"TRAIN SPLIT:{len(train_dataset)}")
    print(f"VALIDATION SPLIT:{len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Model Setup
    model = ThyroidNet(256, 256, pretrained=True).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)


    criterion = Loss().to(DEVICE)
  
    
    best_val_dice = 0.0

    # EPOCH LOOP
    for epoch in range(EPOCHS):
        model.train()
        train_loss_accum = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for imgs, masks in loop:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            
            # Forward
            S_map, alpha_map, gain_vec, mask_logits, gland_map = model(imgs)
            loss = criterion(S_map, alpha_map, gain_vec, mask_logits, imgs, masks, gland_map)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss_accum += loss.item()
            
            # Update the progress bar with the current loss
            loop.set_postfix(loss=loss.item())
            
        avg_train_loss = train_loss_accum / len(train_loader)
        
        # --- Evaluation ---
        val_loss, val_dice, val_iou = evaluate(model, val_loader, DEVICE, criterion)
        
        tqdm.write(f"End of Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Dice: {val_dice:.4f}")
        
        # --- Checkpointing ---
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), "best_model.pth")
            tqdm.write(f"   >>> New Best Model Saved! (Dice: {val_dice:.4f})")
            
        torch.save(model.state_dict(), "latest_model.pth")

    print(f"Training Complete. Best Validation Dice: {best_val_dice:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to dataset")
    args = parser.parse_args()

    data_dir = args.data

    if data_dir=="" or data_dir==None:
        print("Please provide the training data root directory: --data root_dir")
        sys.exit()
    elif not os.path.exists(data_dir):
        print("Please provide the valid data directory")
        sys.exit()
    elif not os.path.exists(os.path.join(data_dir,"images")) or not os.path.exists(os.path.join(data_dir,"masks")):
        print("Please provide valid data directory with images and masks folder in it.")
        sys.exit()
        


    train(data_dir)

if __name__ == "__main__":
    main()   