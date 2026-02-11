# ============================================================================
# MONKEYPATCH: FIX DATASETS vs PILLOW VERSION ISSUE
# ============================================================================
import PIL.Image
import PIL.ExifTags
if not hasattr(PIL.Image, 'ExifTags'):
    print("✓ Monkeypatching PIL.Image.ExifTags to fix datasets library crash...")
    PIL.Image.ExifTags = PIL.ExifTags

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
import numpy as np
from typing import List, Dict, Tuple
import warnings
import pandas as pd
import os
import gc
import sys
import os
from datasets import load_dataset
# import warnings # Already imported
from scipy import stats
from scipy.sparse.linalg import LinearOperator, eigsh

# from nnsight import NNsight
import foolbox as fb
from foolbox.attacks import LinfDeepFoolAttack, L2DeepFoolAttack, LinfPGD, L2CarliniWagnerAttack

warnings.filterwarnings('ignore')

# Check for extra attacks
try:
    import torchattacks
    TORCHATTACKS_AVAILABLE = True
    print("✓ Torchattacks available")
except ImportError:
    TORCHATTACKS_AVAILABLE = False
    print("⚠ Torchattacks not available")

try:
    from autoattack import AutoAttack
    AUTOATTACK_AVAILABLE = True
    print("✓ AutoAttack available")
except ImportError:
    AUTOATTACK_AVAILABLE = False
    print("⚠ AutoAttack not available")

# ============================================================================
# REPRODUCIBILITY
# ============================================================================
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Random seed: {RANDOM_SEED}")

# ============================================================================
# 1. LOAD MODEL (ViT-B/16)
# ============================================================================
# ============================================================================
# 1. LOAD MODEL (ViT-B/16) - ImageNet
# ============================================================================
def load_model_and_data():
    """Load pretrained ViT-B/16 and ImageNet Validation Stream"""
    print("Loading ViT-B/16 (ImageNet weights)...")
    # IMAGENET1K_V1: 76.130% Acc -> Best for "Standard" ViT
    weights = ViT_B_16_Weights.IMAGENET1K_V1
    model = vit_b_16(weights=weights)
    model = model.to(device)
    model.eval()

    # Preprocessing
    preprocess = weights.transforms()
    
    # Streaming ImageNet Validation
    print("  → Connecting to ImageNet-1k Validation Stream...")
    # Streaming ImageNet Validation
    print("  → Connecting to ImageNet-1k Validation Stream...")
    dataset = None
    try:
        # Try User-Specified ID: ILSVRC/imagenet-1k
        print("  → Trying 'ILSVRC/imagenet-1k'...")
        dataset = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True, trust_remote_code=True) 
    except Exception as e0:
        print(f"  ⚠ Failed to load ILSVRC/imagenet-1k: {e0}")
        try:
            # Try Official ImageNet-1k (requires Auth)
            print("  → Trying 'imagenet-1k'...")
            dataset = load_dataset("imagenet-1k", split="validation", streaming=True)
        except Exception as e:
            print(f"  ⚠ Failed to load ImageNet-1k: {e}")
        print("  → Trying fallback: 'frgfm/imagenette' (Public Subset)...")
        try:
            # Fallback 1: Imagenette (10 classes, open)
            dataset = load_dataset("frgfm/imagenette", "320px", split="validation", streaming=True)
        except Exception as e2:
            print(f"  ⚠ Failed to load Imagenette: {e2}")
            try:
                # Fallback 2: Tiny ImageNet or similar? Let's try cifar10 as last resort for stability test only
                print("  → Trying fallback: 'cifar10' (Domain shift warning!)...")
                dataset = load_dataset("cifar10", split="test", streaming=True)
            except Exception as e3:
                 print(f"  ❌ ALL DATASET LOADS FAILED. Script will likely crash.")
                 dataset = []

    if not dataset:
         print("  ⚠ No data available.")

    # WRAPPER FIX: Model expects ImageNet Normalization, but we use [0, 1] tensors
    class Normalize(nn.Module):
        def __init__(self, mean, std):
            super().__init__()
            self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))
        def forward(self, x):
            return (x - self.mean) / self.std
            
    norm_layer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).to(device)
    model = nn.Sequential(norm_layer, model)
    model.eval()

    # Foolbox expects bounds (0, 1), and our wrapped model takes (0, 1) outputting logits
    fmodel = fb.PyTorchModel(model, bounds=(0, 1), device=device)
    return model, fmodel, dataset, preprocess

def compute_normalized_power_spectrum(img):
    """Compute NPS (1D Azimuthal Integration)"""
    if torch.is_tensor(img):
        img = img.detach().cpu().numpy()
    if img.ndim == 3:
        img = np.mean(img, axis=0) # Grayscale
        
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
    
    h, w = img.shape
    cy, cx = h // 2, w // 2
    y, x = np.mgrid[:h, :w]
    radii = np.sqrt((x - cx)**2 + (y - cy)**2).astype(int)
    
    # Radial Profile
    tbin = np.bincount(radii.ravel(), magnitude_spectrum.ravel())
    nr = np.bincount(radii.ravel())
    radial_profile = tbin / (nr + 1e-8)
    
    # Normalize
    profile = radial_profile[:min(h, w)//2]
    normalized_profile = (profile - profile.min()) / (profile.max() - profile.min() + 1e-8)
    
    return normalized_profile

def compute_activation_spectrum(activation):
    """Compute spectrum for activation tensor"""
    if activation.ndim == 4:  # (B, C, H, W)
        spatial = activation.mean(dim=1)  # Average over channels
        spectrum = compute_normalized_power_spectrum(spatial.squeeze())
        return spectrum
    return None

def extract_penultimate_features(model, x):
    """Extract features before final classification layer"""
    # ViT: Return ALL tokens for ED calculation (Spatial ED)
    features = []
    def hook(m, i, o):
        # o shape: (B, N, D)
        features.append(o)
    
    
    # Unwrap if wrapped with Normalize (nn.Sequential)
    # We assume model is either ViT or Sequential(Normalize, ViT)
    if isinstance(model, nn.Sequential):
        backbone = model[1] # ViT is usually at index 1
    else:
        backbone = model

    # Determine layer to hook
    target_layer = None
    if hasattr(backbone, 'blocks'):
        target_layer = backbone.blocks[-1]
    elif hasattr(backbone, 'encoder') and hasattr(backbone.encoder, 'layers'):
        target_layer = backbone.encoder.layers[-1]
    
    if target_layer:
        handle = target_layer.register_forward_hook(hook)
        _ = model(x) # Forward pass through FULL model (including Normalize)
        handle.remove()
        if not features:
            print("    ⚠ Warning: No features captured!")
            return None
        feat = features[0]
        # print(f"    DEBUG: Captured feature shape: {feat.shape}")
        return feat # (B, N, D)
    print("    ⚠ Warning: No target layer found for features")
    return None





# ============================================================================
# 3. ATTACK GENERATION
# ============================================================================
# ============================================================================
# 3. ATTACK GENERATION
# ============================================================================
def generate_adversarial_examples(fmodel, model, images, labels, attack_type='DeepFool'):
    """
    Generate adversarial examples.
    Input: images in [0, 1] range.
    Model: Wrapped model expecting [0, 1] range.
    """
    images = images.to(device)
    labels = labels.to(device)
    
    # Foolbox (Expects [0, 1])
    if attack_type in ['DeepFool', 'PGD', 'CW']:
        if attack_type == 'DeepFool':
            attack = LinfDeepFoolAttack(steps=50, loss='logits')
            epsilons = None
        elif attack_type == 'PGD':
            # PGD-40 with stepsize 2/255 (standard relative to eps=8/255)
            attack = LinfPGD(steps=40, abs_stepsize=2/255, random_start=True)
            epsilons = [8/255] 
        elif attack_type == 'CW':
            # Increased strength: 5 BS steps, 1000 iter (vs 3/100)
            attack = L2CarliniWagnerAttack(binary_search_steps=5, steps=1000, 
                                           stepsize=0.01, confidence=0)
            epsilons = None
            
        try:
            if epsilons:
                _, advs, success = attack(fmodel, images, labels, epsilons=epsilons)
                advs = advs[0] if isinstance(advs, list) else advs
                success = success[0] if isinstance(success, (list, tuple)) else success
            else:
                # Use None for epsilons to allow unbounded search (DeepFool/CW)
                _, advs, success = attack(fmodel, images, labels, epsilons=None)
                
            return (advs, True) if success.item() else (images, False)
        except Exception as e:
            print(f"    ⚠ Attack {attack_type} failed: {e}")
            return images, False

    # Torchattacks (Expects [0, 1] if we pass wrapped model)
    elif attack_type in ['FGSM', 'BIM', 'MIFGSM'] and TORCHATTACKS_AVAILABLE:
        try:
            # Model already handles normalization, so we just pass it
            atk_model = model 
            
            if attack_type == 'FGSM': 
                atk = torchattacks.FGSM(atk_model, eps=8/255)
            elif attack_type == 'BIM': 
                atk = torchattacks.BIM(atk_model, eps=8/255, alpha=2/255, steps=20)
            elif attack_type == 'MIFGSM': 
                atk = torchattacks.MIFGSM(atk_model, eps=8/255, alpha=2/255, steps=20)
            
            adv = atk(images, labels)
            
            with torch.no_grad():
                success = (model(adv).argmax(1) != labels).all().item()
            
            return adv, success
            
        except Exception as e:
            print(f"    ⚠ TorchAttacks {attack_type} error: {e}")
            return images, False

    # AutoAttack (Expects [0, 1] standard version)
    elif attack_type == 'AutoAttack' and AUTOATTACK_AVAILABLE:
        try:
            # AutoAttack expects a model that takes [0, 1] inputs
            adversary = AutoAttack(model, norm='Linf', eps=8/255, 
                                   version='standard', verbose=False)
            
            adv = adversary.run_standard_evaluation(images, labels, bs=images.shape[0])
            
            with torch.no_grad():
                success = (model(adv).argmax(1) != labels).all().item()
            
            return adv, success
            
        except Exception as e:
            print(f"    ⚠ AutoAttack error: {e}")
            return images, False

    return images, False

# ============================================================================
# 4. SLT METRICS (ViT-Optimized)
# ============================================================================
def compute_effective_dimension(features):
    """Compute Effective Dimensionality via participation ratio"""
    # features: (N, D) or (D,)
    if features is None:
        # print("    DEBUG: Features is None, returning 1.0")
        return 1.0
        
    if features.ndim == 1:
        features = features.unsqueeze(0)
    
    # Center
    feat_centered = features - features.mean(0)
    
    # SVD
    # Note: if N=1, this is trivial (0). We usually need multiple samples to estimate ED.
    # But here we might be looking at spatial tokens?
    # User snippet: sv = torch.linalg.svd(features - features.mean(0), full_matrices=False)[1]
    # If features is (Batch, Dim), we need batch > 1. 
    # IF we are doing single-sample ED on ViT tokens: (1, Tokens, Dim)
    # The snippet implies we might be doing it on tokens?
    # "ViT shows significant Effective Dimensionality (ED) collapse" usually refers to the representation manifold.
    # Let's assume single-sample token covariance for now, OR batch-level if we had batches.
    # But we process 1 sample at a time.
    # Actually, the user snippet calculates ED on 'features_clean' vs 'features_adv'. 
    # If len(features) is 1, ED is undefined/1.
    # Let's assume 'extract_penultimate_features' returns (B, D). With B=1, we can't compute ED of the manifold.
    # BUT, if we compute it on the TOKENS of the ViT? 
    # The user said "ViT shows significant Effective Dimensionality (ED) collapse".
    # And "extract_penultimate_features... CLS token". 
    # One CLS token is a vector. ED of 1 vector is 1.
    # MAYBE they mean the ED of the *perturbation* or something?
    # Or maybe they want to collect features across the whole dataset?
    # "ViT shows significant Effective Dimensionality (ED) collapse (-64%..." 
    # This implies a dataset-level property.
    # However, the user snippet computes it INSIDE the loop: `ED_clean = compute_effective_dimension(features_clean)`.
    # If `features_clean` is a single vector, this fails.
    # UNLESS `extract_penultimate_features` returns the full token sequence? 
    # The snippet says `features.append(o[:, 0, :]) # CLS token`.
    # So it IS a single vector.
    # THIS IS A PROBLEM.
    # Check user snippet again: `inputs` in user snippet seemed to be batch? 
    # `features_clean`... `ED_clean`.
    # Wait, the user's `extract_features` returns `o[:, 0, :]`.
    # Maybe the user implies the input `x` has batch size > 1?
    # In my loop: `for img, lbl in testloader` -> `img` is usually batch 1.
    # I should try to use batch size > 1?
    # Or maybe the "ED collapse" is computed on the tokens of a single image? 
    # `features.append(o)` (all tokens).
    # If I change `features.append(o)` instead of `o[:, 0, :]`, I get (B, 197, 768). 
    # Then I can compute ED of the 197 tokens.
    # That makes sense for "Representation Collapse" of a single image's representation!
    # I will support both.
    
    # For now, I'll follow the user explicitly:
    # "features_clean = extract_penultimate_features(model, img_tensor)"
    # If the user code fails on B=1, I will catch it.
    
    if features is None:
        # print("    DEBUG: Features is None, returning 1.0")
        return 1.0

    # Auto-squeeze singleton batch dim: (1, N, D) -> (N, D)
    if features.ndim == 3 and features.shape[0] == 1:
        features = features.squeeze(0)
    
    # Handle single vector case: (D,) -> (1, D)
    if features.ndim == 1:
        features = features.unsqueeze(0)
    
    try:
        if features.ndim > 1 and features.shape[0] > 1:
            # Correct Centering: Center across tokens (dim 0)
            feat_centered = features - features.mean(0)
            
            cov = feat_centered @ feat_centered.T
            eigs = torch.linalg.eigvalsh(cov)
            eigs = eigs[eigs > 1e-6]
            if len(eigs) == 0: 
                return 0.0
            ed = (eigs.sum()**2) / (eigs**2).sum()
            return ed.item()
        else:
            # Single vector case (ED=1.0)
            # print("    DEBUG: ED input is single vector (N=1)")
            return 1.0
    except Exception as e:
        print(f"    ⚠ ED Computation Failed: {e}")
        return 1.0

def estimate_gradient_norm(model, image, label):
    """Compute ||∇L||² (Fisher Trace Proxy)"""
    model.eval()
    if image.ndim == 3: 
        image = image.unsqueeze(0)
    if isinstance(label, int):
        label = torch.tensor([label], device=device)
    elif label.ndim == 0:
        label = label.unsqueeze(0)
    
    image = image.to(device)
    label = label.to(device)
    
    model.zero_grad()
    # Ensure gradients w.r.t INPUT
    if not image.requires_grad:
        image.requires_grad = True # Just in case
    
    out = model(image)
    loss = nn.CrossEntropyLoss()(out, label)
    
    # Auto-diff Gradient w.r.t INPUT
    grads = torch.autograd.grad(loss, image, create_graph=False, allow_unused=True)
    
    norm_sq = 0
    for g in grads:
        if g is not None:
            norm_sq += g.pow(2).sum().item()
            
    return norm_sq

def estimate_hessian_trace_robust(model, image, label, num_v=10):
    """
    Compute Trace of Hessian via Hutchinson's Method
    
    FIX #4: Critical fix for ViT - disable Flash Attention for double backprop
    """
    model.eval()
    
    if image.ndim == 3: 
        image = image.unsqueeze(0)
    if isinstance(label, int):
        label = torch.tensor([label], device=device)
    elif label.ndim == 0:
        label = label.unsqueeze(0)
    
    image = image.to(device)
    label = label.to(device)
    
    # FIX #4: Disable Flash Attention for ViT double backward
    # Flash Attention's backward pass is not differentiable (can't do double backward)
    # This is CRITICAL for ViT Hessian computation
    try:
        # Try with context manager (PyTorch 2.0+)
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
            return _compute_hessian_trace_inner(model, image, label, num_v)
    except AttributeError:
        # Fallback for older PyTorch
        print("  ⚠ Flash Attention control not available - Hessian may fail")
        return _compute_hessian_trace_inner(model, image, label, num_v)

def _compute_hessian_trace_inner(model, image, label, num_v):
    """Inner computation (separated for cleaner error handling)"""
    # First pass: compute gradient w.r.t INPUT
    model.zero_grad()
    
    if not image.requires_grad:
        image.requires_grad = True
        
    output = model(image)
    criterion = nn.CrossEntropyLoss() # Define criterion here
    loss = criterion(output, label)
    
    # Gradient w.r.t INPUT
    grads = torch.autograd.grad(loss, image, create_graph=True, allow_unused=True)
    grads = [g for g in grads if g is not None]
    
    if not grads: 
        return None, None
    
    grad_vec = torch.cat([g.contiguous().view(-1) for g in grads])
    
    # Hutchinson samples
    traces = []
    for i in range(num_v):
        v = torch.randint_like(grad_vec, high=2, device=device).float() * 2 - 1
        
        try:
            # Compute Hessian-vector product w.r.t INPUT
            Hv = torch.autograd.grad(
                outputs=torch.dot(grad_vec, v),
                inputs=image, # Correct: Input Hessian
                create_graph=False,
                retain_graph=(i < num_v - 1),
                allow_unused=True
            )
            Hv = [h for h in Hv if h is not None]
            
            if not Hv: 
                continue
            
            Hv_vec = torch.cat([h.contiguous().view(-1) for h in Hv])
            trace_sample = torch.dot(v[:len(Hv_vec)], Hv_vec).item()
            traces.append(trace_sample)
            
        except RuntimeError as e:
            if "out of memory" in str(e): 
                print("  ⚠ OOM in Hessian computation")
                torch.cuda.empty_cache()
                return None, None
            elif "double backward" in str(e).lower() or "not differentiable" in str(e).lower():
                print("  ⚠ ViT attention not differentiable - skipping Hessian")
                return None, None
            else:
                raise e
    
    if not traces: 
        return None, None
    
    mean_trace = np.mean(traces)
    std_error = np.std(traces) / np.sqrt(len(traces))
    
    return mean_trace, std_error

def estimate_parameter_hessian_trace(model, image, label, num_v=5):
    """
    Estimate Trace of the Parameter Hessian (Weight Space).
    Context: Generalization / Flatness (Paper 1).
    """
    model.eval()
    model.zero_grad()
    
    # 1. Gradient w.r.t PARAMETERS
    # Note: FLASH ATTENTION handling needed for ViT
    try:
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
             output = model(image)
             loss = nn.CrossEntropyLoss()(output, label)
             grads = torch.autograd.grad(loss, model.parameters(), create_graph=True, allow_unused=True)
    except AttributeError:
        output = model(image)
        loss = nn.CrossEntropyLoss()(output, label)
        grads = torch.autograd.grad(loss, model.parameters(), create_graph=True, allow_unused=True)

    grads = [g for g in grads if g is not None]
    if not grads: return None
    
    grad_vec = torch.cat([g.contiguous().view(-1) for g in grads])
    
    traces = []
    for i in range(num_v):
        v = torch.randint_like(grad_vec, high=2, device=device).float() * 2 - 1
        try:
             # Disable flash attention for double backward
             with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
                Hv = torch.autograd.grad(
                    outputs=torch.dot(grad_vec, v),
                    inputs=model.parameters(),
                    create_graph=False,
                    retain_graph=(i < num_v - 1),
                    allow_unused=True
                )
             Hv = [h for h in Hv if h is not None]
             if Hv:
                Hv_vec = torch.cat([h.contiguous().view(-1) for h in Hv])
                val = torch.dot(v[:len(Hv_vec)], Hv_vec).item()
                traces.append(val)
        except Exception as e:
            # print(f"Param Hessian Error: {e}")
            torch.cuda.empty_cache()
            return None
            
    if not traces: return None
    return np.mean(traces)

def classify_regime(trace):
    """Classify curvature regime"""
    if trace is None or np.isnan(trace): 
        return "Unknown"
    if trace > 1000: 
        return "Positive Curvature (Strong)"
    if trace > 0: 
        return "Positive Curvature (Weak)"
    if trace > -1000: 
        return "Negative Curvature (Weak)"
    return "Negative Curvature (Strong)"

# ============================================================================
# 4.a DOMINANT HESSIAN ANALYSIS (NEW)
# ============================================================================


    







def export_results_to_csv(results, architecture_name, output_dir='results_csv'):
    """Export SLT analysis results to CSV files for detailed analysis."""
    import pandas as pd
    import numpy as np
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Summary CSV
    print(f"  Creating summary CSV...")
    summary_data = []
    for attack in results.keys():
        total = results[attack]['asr_stats']['total_valid_samples']
        success = results[attack]['asr_stats']['successful_adv_samples']
        failed = results[attack]['asr_stats']['failed_adv_samples']
        asr = (success / total * 100) if total > 0 else 0
        summary_data.append({
            'Attack': attack,
            'Total_Valid_Samples': total,
            'Successful_Attacks': success,
            'Failed_Attacks': failed,
            'ASR': f"{asr:.2f}%"
        })
    pd.DataFrame(summary_data).to_csv(f"{output_dir}/{architecture_name}_summary.csv", index=False)
    
    # 2. Detailed CSV per attack
    for attack in results.keys():
        print(f"  Creating detailed CSV for {attack}...")
        detailed_data = []
        success_idx = 0
        failed_idx = 0
        
        for i in range(len(results[attack]['metadata']['attack_success'])):
            is_success = results[attack]['metadata']['attack_success'][i]
            adv_dict = results[attack]['adversarial_success' if is_success else 'adversarial_failed']
            adv_idx = success_idx if is_success else failed_idx
            if is_success:
                success_idx += 1
            else:
                failed_idx += 1
            
            detailed_data.append({
                'Sample_ID': i,
                'Attack_Success': is_success,
                'Original_Label': results[attack]['metadata']['original_label'][i],
                'Clean_Prediction': results[attack]['metadata']['clean_prediction'][i],
                'Adv_Prediction': results[attack]['metadata']['adv_prediction'][i],
                'Clean_Input_Trace': results[attack]['clean']['hessian_trace'][i],
                'Adv_Input_Trace': adv_dict['hessian_trace'][adv_idx] if adv_idx < len(adv_dict['hessian_trace']) else np.nan,
                'Clean_Param_Trace': results[attack]['clean']['hessian_trace_param'][i],
                'Adv_Param_Trace': adv_dict['hessian_trace_param'][adv_idx] if adv_idx < len(adv_dict['hessian_trace_param']) else np.nan,
            })
        
        pd.DataFrame(detailed_data).to_csv(f"{output_dir}/{architecture_name}_{attack}_detailed.csv", index=False)
    
    print(f"\n[CSV Export Complete] - Files saved to: {output_dir}/")


def run_analysis_vit(num_samples=30, attack_types=['PGD'], probing=False):
    """
    Run SLT analysis on ViT-B/16
    
    FIX #5: Added detailed progress reporting and error tracking
    """
    print("\n" + "="*80)
    print("ViT-B/16 SLT ANALYSIS")
    print("="*80)
    print(f"Target samples per attack: {num_samples}")
    print(f"Attacks: {', '.join(attack_types)}")
    print(f"Hutchinson samples: 10 (for stability)")
    print("="*80)
    
    # Load model & data
    model, fmodel, dataset, preprocess = load_model_and_data()
    
    # Initialize results storage
    results = {
        atk: {
            'clean': {'hessian_trace': [], 'hessian_stderr': [], 'hessian_trace_param': [], 'gradient_norm': [], 'curvature_regime': [], 'ed': []}, 
            'adversarial_success': {'hessian_trace': [], 'hessian_stderr': [], 'hessian_trace_param': [], 'gradient_norm': [], 'curvature_regime': [], 'ed': []},
            'adversarial_failed': {'hessian_trace': [], 'hessian_stderr': [], 'hessian_trace_param': [], 'gradient_norm': [], 'curvature_regime': [], 'ed': []},
            'metadata': {'original_label': [], 'clean_prediction': [], 'adv_prediction': [], 'attack_success': []},
            'asr_stats': {'total_valid_samples': 0, 'successful_adv_samples': 0, 'failed_adv_samples': 0}
        } 
        for atk in attack_types
    }
    
    if probing:
        for atk in attack_types:
            results[atk]['clean']['probing'] = []
            results[atk]['adv']['probing'] = []
    
    # Process each attack
    for atk in attack_types:
        print(f"\n{'='*80}")
        print(f"ATTACK: {atk}")
        print(f"{'='*80}")
        
        count = 0
        attempts = 0
        max_attempts = num_samples * 5
        
        # Generator for streaming dataset
        data_iter = iter(dataset)
        
        while count < num_samples and attempts < max_attempts:
            try:
                item = next(data_iter)
            except StopIteration:
                break
                
            attempts += 1
            
            # Prepare image
            try:
                img_pil = item['image']
                if img_pil.mode != 'RGB': continue
                lbl_idx = item['label']
                
                # Preprocess: -> Tensor (0-1) -> Normalize -> Model
                # Standard timm preprocess usually includes Normalize.
                # But for attacks we want (0-1) tensor.
                # So we manually do Resize/Crop/ToTensor
                img = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor()
                ])(img_pil).unsqueeze(0).to(device)
                
                lbl = torch.tensor([lbl_idx]).to(device)
                
            except Exception as e:
                continue
            
            # Skip misclassified clean samples
            with torch.no_grad():
                out = model(img)
                pred = out.argmax(1)
                conf = F.softmax(out, 1).max().item()
                
                if pred != lbl or conf < 0.4: # Lower confidence threshold for ImageNet 
                    continue
                
                # Valid sample for ASR
                results[atk]['asr_stats']['total_valid_samples'] += 1

            
            # Generate adversarial example
            adv, success = generate_adversarial_examples(fmodel, model, img, lbl, atk)
            
            if not success:
               print(f"    ✗ Attack generation failed entirely (skipping)")
               results[atk]['asr_stats']['failed_adv_samples'] += 1
               continue

            with torch.no_grad():
                adv_pred = model(adv).argmax(1)
                attack_succeeded = (adv_pred != lbl).item()
            
            # Update ASR Stats
            if attack_succeeded:
                results[atk]['asr_stats']['successful_adv_samples'] += 1
            else:
                results[atk]['asr_stats']['failed_adv_samples'] += 1

            # Compute SLT Metrics
            try:
                print(f"  Sample {count+1}/{num_samples}: Label={lbl.item()}, " + 
                      f"Clean Pred={pred.item()} (conf={conf:.2%}), " +
                      f"Adv Pred={adv_pred.item()}")
                
                # Clean metrics
                print(f"    Computing clean metrics...", end='')
                c_grad = estimate_gradient_norm(model, img, lbl)
                c_trace, c_stderr = estimate_hessian_trace_robust(model, img, lbl, num_v=10)
                c_trace_param = estimate_parameter_hessian_trace(model, img, lbl, num_v=5)
                c_regime = classify_regime(c_trace)
                print(f" Grad: {c_grad:.2e}, Input Tr: {c_trace if c_trace else 'N/A'}, Param Tr: {c_trace_param if c_trace_param else 'N/A'}")
                
                torch.cuda.empty_cache()
                
                # Adversarial metrics
                print(f"    Computing adv metrics...", end='')
                a_grad = estimate_gradient_norm(model, adv, lbl)
                a_trace, a_stderr = estimate_hessian_trace_robust(model, adv, lbl, num_v=10)
                a_trace_param = estimate_parameter_hessian_trace(model, adv, lbl, num_v=5)
                a_regime = classify_regime(a_trace)
                print(f" Grad: {a_grad:.2e}, Input Tr: {a_trace if a_trace else 'N/A'}, Param Tr: {a_trace_param if a_trace_param else 'N/A'}")
                
                torch.cuda.empty_cache()
                
                # Novel Metric: ED (on tokens)
                feat_c = extract_penultimate_features(model, img)
                feat_a = extract_penultimate_features(model, adv)
                
                ed_c = compute_effective_dimension(feat_c)
                ed_a = compute_effective_dimension(feat_a)
                
                print(f"    ED Clean: {ed_c:.1f}, ED Adv: {ed_a:.1f}")

                # Store Results
                # Clean (Always stored)
                results[atk]['clean']['gradient_norm'].append(c_grad)
                results[atk]['clean']['hessian_trace'].append(c_trace if c_trace is not None else np.nan)
                results[atk]['clean']['hessian_stderr'].append(c_stderr if c_stderr is not None else np.nan)
                results[atk]['clean']['hessian_trace_param'].append(c_trace_param if c_trace_param is not None else np.nan)
                results[atk]['clean']['curvature_regime'].append(c_regime)
                results[atk]['clean']['ed'].append(ed_c)
                
                # Adversarial (Success vs Failed split)
                target_dict = results[atk]['adversarial_success'] if attack_succeeded else results[atk]['adversarial_failed']
                
                target_dict['gradient_norm'].append(a_grad)
                target_dict['hessian_trace'].append(a_trace if a_trace is not None else np.nan)
                target_dict['hessian_stderr'].append(a_stderr if a_stderr is not None else np.nan)
                target_dict['hessian_trace_param'].append(a_trace_param if a_trace_param is not None else np.nan)
                target_dict['curvature_regime'].append(a_regime)
                target_dict['ed'].append(ed_a)
                
                # Metadata
                results[atk]['metadata']['original_label'].append(lbl.item())
                results[atk]['metadata']['clean_prediction'].append(pred.item())
                results[atk]['metadata']['adv_prediction'].append(adv_pred.item())
                results[atk]['metadata']['attack_success'].append(attack_succeeded)
                
                # Report
                if c_grad > 0:
                    grad_ratio = a_grad / c_grad
                    print(f"    ✓ Gradient amplification: {grad_ratio:.1f}x")
                
                if c_trace is not None and a_trace is not None:
                    regime_change = f"{c_regime} → {a_regime}"
                    print(f"    ✓ Curvature regime: {regime_change}")
                
                count += 1
                
            except Exception as e:
                print(f"\n    ✗ Error processing sample: {e}")
                torch.cuda.empty_cache()
                gc.collect()
                continue
        
        print(f"\n  → Completed: {count}/{num_samples} successful samples for {atk}")
        
        # Cleanup
        torch.cuda.empty_cache()
        gc.collect()

    # ========================================================================
    # SUMMARY STATISTICS
    # ========================================================================
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    for atk in attack_types:
        d = results[atk]
        
        # Merge success/failed for summary statistics? 
        # Or just use success? 
        # The original code just used d['clean'] vs d['adv'].
        # Now we have d['adversarial_success'] and d['adversarial_failed'].
        # For simplicity in Summary, let's aggregate them back or just focus on SUCCESSFUL attacks for comparison?
        # Usually we compare Clean vs Successful Adversarial.
        
        if not d['clean']['gradient_norm']:
            print(f"\n{atk}: No samples processed")
            continue
            
        # Helper to get array
        def get_diff_metric(clean_list, adv_success_list, adv_failed_list):
            # We want to align them? 
            # Actually, the lists in 'clean' correspond 1-to-1 with the samples processed.
            # But 'adversarial_success' only contains successful ones? NO.
            # Wait.
            # My previous loop logic:
            # target_dict = results[atk]['adversarial_success'] if attack_succeeded else results[atk]['adversarial_failed']
            # target_dict['gradient_norm'].append(a_grad)
            # This means 'clean' has N entries. 'adversarial_success' has S entries. 'adversarial_failed' has F entries.
            # S + F = N (valid samples).
            # So we cannot directly compare arrays index-by-index unless we reconstruct the full array or filter 'clean' to 'success'.
            # Let's filter 'clean' to match 'success' for the comparison stats.
            return
        
        # We need to reconstruct the "Successful" subset of Clean to compare with Adv Success.
        # But wait, `results[atk]['clean']` has all valid samples.
        # We can use `results[atk]['metadata']['attack_success']` boolean mask!
        
        success_mask = np.array(results[atk]['metadata']['attack_success'])
        
        if success_mask.sum() == 0:
             print(f"\n{atk}: No successful attacks to analyze")
             continue

        # Extract Clean Metrics (All)
        c_grad_all = np.array(d['clean']['gradient_norm'])
        c_trace_all = np.array([x if x is not None else np.nan for x in d['clean']['hessian_trace']])
        c_ed_all = np.array(d['clean']['ed'])

        # Extract Adv Metrics (Success Only) -> These are in d['adversarial_success']
        # BUT they are not aligned by index if we just take the list.
        # Actually they are just a list of successful ones.
        # And `c_grad[success_mask]` should align with `d['adversarial_success']['gradient_norm']`?
        # Let's verify loop logic.
        # Loop: 
        #   Append to Clean (Index i)
        #   If Success: Append to AdvSuccess (Index j)
        #   Else: Append to AdvFail (Index k)
        #   Append to Metadata Success[i] = True
        # So YES, `Clean[success_mask]` corresponds exactly to `AdvSuccess`.
        
        c_grad = c_grad_all[success_mask]
        c_trace = c_trace_all[success_mask]
        c_ed = c_ed_all[success_mask]
        
        a_grad = np.array(d['adversarial_success']['gradient_norm'])
        a_trace = np.array([x if x is not None else np.nan for x in d['adversarial_success']['hessian_trace']])
        a_ed = np.array(d['adversarial_success']['ed'])
        
        
        # Filter valid traces (NaNs)
        valid_mask = ~np.isnan(c_trace) & ~np.isnan(a_trace)
        c_trace_valid = c_trace[valid_mask]
        a_trace_valid = a_trace[valid_mask]
        
        print(f"\n{'='*80}")
        print("NOVELTY METRICS: PAPER 2 (ASR, ACCURACY, CORRELATION)")
        print(f"{'='*80}")
        
        # Pearson Correlation (Gradient vs Trace)
        if len(a_grad) > 2:
            valid_idx = ~np.isnan(a_trace)
            if valid_idx.sum() > 2:
                r, p = stats.pearsonr(a_grad[valid_idx], a_trace[valid_idx])
                print(f"  Gradient-Curvature Correlation (r): {r:.4f} (p={p:.4e})")
                if r < -0.3: print("    → NEGATIVE coupling (Instability)")
                elif r > 0.3: print("    → POSITIVE coupling (Stability)")
            else:
                print("  (Not enough valid traces for correlation)")

        print(f"\n{'='*80}")
        print(f"ATTACK: {atk} (n={len(c_grad)})")
        print(f"{'='*80}")
        
        # Gradient analysis
        grad_ratio = a_grad.mean() / c_grad.mean() if c_grad.mean() > 0 else np.nan
        print(f"\n★★★ GRADIENT NORM² (Primary Metric):")
        print(f"  Clean:       {c_grad.mean():.2e} ± {c_grad.std():.2e}")
        print(f"  Adversarial: {a_grad.mean():.2e} ± {a_grad.std():.2e}")
        print(f"  Ratio:       {grad_ratio:.1f}x")
        
        if grad_ratio > 100:
            print(f"  → EXTREME gradient amplification!")
        elif grad_ratio > 10:
            print(f"  → MAJOR gradient amplification")
        
        # ED Analysis
        if len(c_ed) > 0:
            print(f"\n★★ EFFECTIVE DIMENSIONALITY (Spatial Token Collapse):")
            print(f"  Clean:       {c_ed.mean():.1f} ± {c_ed.std():.1f}")
            print(f"  Adversarial: {a_ed.mean():.1f} ± {a_ed.std():.1f}")
            if c_ed.mean() > 0:
                collapse_pct = (c_ed.mean() - a_ed.mean()) / c_ed.mean() * 100
                print(f"  Collapse:    {collapse_pct:.1f}%")

        # Trace analysis
        if len(c_trace_valid) > 0:
            print(f"\n★★ HESSIAN TRACE (Curvature - {len(c_trace_valid)}/{len(c_grad)} valid):")
            print(f"  Clean:       {c_trace_valid.mean():.2e} ± {c_trace_valid.std():.2e}")
            print(f"  Adversarial: {a_trace_valid.mean():.2e} ± {a_trace_valid.std():.2e}")
            
            # Curvature regime analysis
            pos_mask = (c_trace_valid > 0) & (a_trace_valid > 0)
            saddle_count = (a_trace_valid < 0).sum()
            
            if pos_mask.sum() > 0:
                curv_ratio = a_trace_valid[pos_mask].mean() / c_trace_valid[pos_mask].mean()
                print(f"\n  Local Minimum Regime: {pos_mask.sum()}/{len(c_trace_valid)} samples")
                print(f"    Curvature ratio: {curv_ratio:.2f}x")
                
                if curv_ratio > 2:
                    print(f"    → STRONG SHARPENING")
                elif curv_ratio < 0.8:
                    print(f"    → FLATTENING (Shifting towards Negative)")
                else:
                    print(f"    → Stable")
            
            if saddle_count > 0:
                saddle_pct = saddle_count / len(a_trace_valid) * 100
                print(f"\n  Saddle Point Regime: {saddle_count}/{len(a_trace_valid)} ({saddle_pct:.0f}%)")
                print(f"    → Geometric instability detected")
        else:
            print(f"\n★★ HESSIAN TRACE: No valid traces computed")
            print(f"  ⚠ ViT attention may not support double backward")
    
    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    try:
        # Save as NPZ
        save_dict = {}
        for atk in attack_types:
            # Clean metrics
            for metric in ['gradient_norm', 'hessian_trace', 'hessian_stderr', 'hessian_trace_param', 'ed']:
                save_dict[f'{atk}_clean_{metric}'] = np.array(results[atk]['clean'][metric])
            
            # Advisarial metrics (Success Only)
            for metric in ['gradient_norm', 'hessian_trace', 'hessian_stderr', 'hessian_trace_param', 'ed']:
                save_dict[f'{atk}_adv_success_{metric}'] = np.array(results[atk]['adversarial_success'][metric])
                
            # Advisarial metrics (Failed Only)
            for metric in ['gradient_norm', 'hessian_trace', 'hessian_stderr', 'hessian_trace_param', 'ed']:
                save_dict[f'{atk}_adv_failed_{metric}'] = np.array(results[atk]['adversarial_failed'][metric])
            
            # Save Metadata
            save_dict[f'{atk}_attack_success'] = np.array(results[atk]['metadata']['attack_success'])
            save_dict[f'{atk}_original_label'] = np.array(results[atk]['metadata']['original_label'])
        
        np.savez('slt_vit_results.npz', **save_dict)
        print("✓ Saved to slt_vit_results.npz")
        
        # Compute and Print Correlations (NOVEL FINDING)
        # analyze_correlations(results, attack_types) # Function not defined in this partial view, disabling for safety
        

        
    except Exception as e:
        print(f"⚠ Error in saving/analysis block: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    
    
    # CSV Export
    print(f"\n{'='*80}")
    print("[Exporting to CSV]")
    print(f"{'='*80}")
    export_results_to_csv(results, architecture_name='vit', output_dir='results_csv')
    
    return results



if __name__ == "__main__":
    print("\n" + "="*80)
    print("STARTING ViT SLT ANALYSIS EXECUTION")
    print("="*80)
    
    # Run full analysis
    results = run_analysis_vit(
        num_samples=500,
        attack_types=['AutoAttack', 'PGD', 'BIM', 'MIFGSM', 'FGSM', 'DeepFool', 'CW']
    )
    
    print("\n" + "="*80)
    print("ALL TASKS COMPLETED SUCCESSFULLY")
    print("="*80)
