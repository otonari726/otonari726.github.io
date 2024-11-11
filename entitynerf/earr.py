import torch
import torch.nn.functional as F

def earr_loss(rgb_gt, rgb_outputs, mask, patch_size=64, residual_threshold=0.8):
    """
    Compute robust RGB loss.
    
    Args:
        rgb_gt (torch.Tensor): Ground truth RGB image
        rgb_outputs (torch.Tensor): Predicted RGB image
        mask (torch.Tensor): Mask tensor with 0 for stuff and otherwise instance ID (things)
        patch_size (int): Size of patches to use
        residual_threshold (float): Threshold for residual
    
    Returns:
        torch.Tensor: Computed loss
    """
    # Reshape inputs into patches
    out_patches = rgb_outputs.view(-1, patch_size, patch_size, 3)
    gt_patches = rgb_gt.view(-1, patch_size, patch_size, 3)
    
    # Calculate residuals
    residuals = torch.mean((out_patches - gt_patches) ** 2, dim=-1)
    
    with torch.no_grad():
        # Get unique mask IDs
        unique_mask_ids = [torch.unique(mask[i]) for i in range(mask.shape[0])]
        
        # Calculate median residual
        med_residual = torch.quantile(residuals, residual_threshold)
        
        # Calculate initial weights (Equation 8)
        weight = (residuals <= med_residual).float()
        weight[mask == 0] = 1.0
        
        # Calculate average weights for each unique mask ID
        refined_weights = torch.ones_like(weight)
        for i, unique_ids in enumerate(unique_mask_ids):
            for j in range(1, unique_ids.shape[0]):
                mask_region = mask[i] == unique_ids[j]
                refined_weights[i][mask_region] = torch.mean(weight[i][mask_region])
        
        # Threshold refined weights
        refined_weights = (refined_weights > residual_threshold).float()
        print(f"Refined weights shape: {refined_weights.shape}, max: {refined_weights.max()}, min: {refined_weights.min()}, mean: {refined_weights.mean()}")
        
        # Apply dilation
        refined_weights = apply_dilation(refined_weights)
        print(f"After dilation - shape: {refined_weights.shape}, max: {refined_weights.max()}, min: {refined_weights.min()}, mean: {refined_weights.mean()}")

    return residuals * refined_weights

def apply_dilation(weights, kernel_size=3):
    # Define dilation kernel
    kernel = torch.ones(1, 1, kernel_size, kernel_size).to(weights.device)
    
    # Apply dilation
    dilated = F.conv2d(
        1 - weights.unsqueeze(1),
        kernel,
        padding=kernel_size // 2,
    )
    dilated = F.conv2d(
        dilated,
        kernel,
        padding=kernel_size // 2,
    )
    
    # Threshold dilated weights
    dilated = torch.where(
        dilated > 0,
        torch.tensor(1.0).to(weights.device),
        torch.tensor(0.0).to(weights.device),
    )
    
    # Remove extra dimension and invert
    return 1 - dilated[:, 0]
