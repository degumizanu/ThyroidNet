import torch
import torch.nn as nn
import torch.nn.functional as F

class Loss(nn.Module):
    def __init__(self, height=256):
        super().__init__()
        self.height = height
        self.bce = nn.BCEWithLogitsLoss(reduction='none') 
        
        # Spatial weights: Penalize Skin (Top 15%)
        y = torch.linspace(0, 1, height).view(1, 1, height, 1)
        self.spatial_weights = torch.ones_like(y)
        self.spatial_weights[y < 0.20] = 10.0 

    def get_surrounding_ring(self, mask):
        # Create the "Peritumoral Ring"
        dilated = F.max_pool2d(mask, kernel_size=21, stride=1, padding=10)
        ring = dilated - mask
        return ring
    def get_approximate_gland_area(self, mask):
        dilated = F.max_pool2d(mask.float(), kernel_size=27, stride=1, padding=13)
        return dilated

    def forward(self, S_map, alpha_map, gain_vec, pred_logits, target_img, target_mask, gland_map):
        
        # --- 1. SEGMENTATION (Weighted) ---
        pixel_loss = self.bce(pred_logits, target_mask)
        weights = self.spatial_weights.to(pixel_loss.device)
        loss_seg = (pixel_loss * weights).mean()
        
        pred_prob = torch.sigmoid(pred_logits)
        inter = (pred_prob * target_mask).sum()
        union = pred_prob.sum() + target_mask.sum()
        dice_loss = 1 - (2. * inter + 1e-6) / (union + 1e-6)
        
        # --- 2. PHYSICS RECONSTRUCTION ---
        z_step = 1.0 / self.height
        integral_alpha = torch.cumsum(alpha_map, dim=2) * z_step
        attenuation = torch.exp(-2.0 * integral_alpha)
        gain_map = gain_vec.unsqueeze(1).unsqueeze(3).expand_as(S_map)
        I_rec = S_map * attenuation * gain_map
        loss_rec = F.l1_loss(I_rec, target_img)
        
        # --- 3. LOCAL HOMOGENEITY ---
        # "The Ring around the nodule should be smooth (low variance)"
        ring_mask = self.get_surrounding_ring(target_mask)
        sum_ring = ring_mask.sum() + 1e-6
        mean_ring_S = (S_map * ring_mask).sum() / sum_ring
        
        variance_ring = torch.abs(S_map - mean_ring_S) * ring_mask
        loss_homogeneity = variance_ring.sum() / sum_ring
        
        # --- 4. CONTRAST ---
        # "The Nodule should look different from the Ring"
        mean_nodule_S = (S_map * target_mask).sum() / (target_mask.sum() + 1e-6)
        loss_contrast = F.relu(0.1 - torch.abs(mean_nodule_S - mean_ring_S))

        # appx_gland_area = self.get_approximate_gland_area(target_mask)

        # l_gland = F.binary_cross_entropy(gland_map, appx_gland_area)
        L_energy = -gland_map.mean()


        
        # Weights
        # return (loss_seg + dice_loss) + 0.3*loss_rec + 0.3 * loss_homogeneity + 0.3 * loss_contrast
        return (loss_seg+dice_loss) + 0.5*loss_rec