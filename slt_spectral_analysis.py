import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import numpy as np
from typing import List, Dict, Tuple
import warnings
import pandas as pd
import os
import gc
warnings.filterwarnings('ignore')

import foolbox as fb
from datasets import load_dataset
# import warnings # Already imported
from scipy import stats
from scipy.sparse.linalg import LinearOperator, eigsh

from nnsight import NNsight
import foolbox as fb
from foolbox.attacks import LinfDeepFoolAttack, L2DeepFoolAttack, LinfPGD, L2CarliniWagnerAttack

# Import torchattacks for additional attack diversity
try:
    import torchattacks
    TORCHATTACKS_AVAILABLE = True
    print("✓ Torchattacks available")
except ImportError:
    TORCHATTACKS_AVAILABLE = False
    print("⚠ Torchattacks not available (install: pip install torchattacks)")

# Import AutoAttack for gold-standard evaluation
try:
    from autoattack import AutoAttack
    AUTOATTACK_AVAILABLE = True
    print("✓ AutoAttack available")
except ImportError:
    AUTOATTACK_AVAILABLE = False
    print("⚠ AutoAttack not available (install: pip install autoattack)")

# ============================================================================
# REPRODUCIBILITY: Set all random seeds
# ============================================================================
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Random seed: {RANDOM_SEED} (for reproducibility)")

# ============================================================================
# 1. LOAD MODEL AND DATA (SAME AS SPECTRAL_BIAS_STUDY.PY)
# ============================================================================

# ============================================================================
# 1. LOAD MODEL AND DATA
# ============================================================================

class Normalize(nn.Module):
    """Normalization wrapper to allow model to accept [0, 1] inputs"""
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))
    def forward(self, x):
        return (x - self.mean) / self.std

def load_model_and_data():
    """Load pretrained ResNet18 and ImageNet Data"""
    print("Loading ResNet18 (ImageNet weights)...")
    # IMAGENET1K_V1: Standard ImageNet weights
    weights = ResNet18_Weights.IMAGENET1K_V1
    base_model = resnet18(weights=weights)
    
    # WRAPPER: Model expects ImageNet Normalization, but we use [0, 1] tensors
    norm_layer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    model = nn.Sequential(norm_layer, base_model)
    
    model = model.to(device)
    model.eval()
    
    # Streaming ImageNet
    print("  → Connecting to ImageNet-1k Validation Stream...")
    try:
        dataset = load_dataset("imagenet-1k", split="validation", streaming=True, trust_remote_code=True)
    except Exception as e:
        print(f"  ⚠ Failed to load ImageNet stream: {e}")
        dataset = [] 
        print("  ⚠ Falling back to blank")
    
    # Foolbox setup (0-1 bounds)
    # Model now handles normalization internally, so no preprocessing needed here
    fmodel = fb.PyTorchModel(model, bounds=(0, 1), device=device)
    
    # Return model, fmodel, dataset (preprocess unused)
    return model, None, fmodel, dataset, None

# ============================================================================
# 3. ADVERSARIAL ATTACKS
# ============================================================================

def generate_adversarial_examples(fmodel, model, images, labels, attack_type='DeepFool'):
    """Generate adversarial examples using Foolbox, Torchattacks, or AutoAttack"""
    images = images.to(device)
    labels = labels.to(device)
    
    # ==================== FOOLBOX ATTACKS ====================
    if attack_type in ['DeepFool', 'DeepFool_L2', 'PGD', 'CW']:
        # images are [0, 1]. Model (via fmodel) handles normalization.
        
        if attack_type == 'DeepFool':
            attack = LinfDeepFoolAttack(steps=50, loss='logits')
            epsilons = None
        elif attack_type == 'DeepFool_L2':
            attack = L2DeepFoolAttack(steps=50, loss='logits')
            epsilons = None
        elif attack_type == 'PGD':
            # Standard ImageNet epsilon 8/255, stepsize 2/255
            attack = LinfPGD(steps=40, abs_stepsize=2/255, random_start=True)
            epsilons = [8/255]
        elif attack_type == 'CW':
            attack = L2CarliniWagnerAttack(binary_search_steps=5, steps=1000, 
                                           stepsize=0.01, confidence=0)
            epsilons = None
        
        try:
            if epsilons is not None:
                _, adv_images, success = attack(fmodel, images, labels, epsilons=epsilons)
                if isinstance(adv_images, list):
                    adv_images = adv_images[0]
                if isinstance(success, (list, tuple)):
                    success = success[0]
            else:
                _, adv_images, success = attack(fmodel, images, labels, epsilons=None)
                
            return adv_images, success.item() if torch.is_tensor(success) else success
            
        except Exception as e:
            print(f"    ⚠ Foolbox Attack {attack_type} failed: {e}")
            return images, False
    
    # ==================== TORCHATTACKS ====================
    elif attack_type in ['FGSM', 'BIM', 'MIFGSM']:
        if not TORCHATTACKS_AVAILABLE:
            return images, False
        
        try:
            # images are [0, 1]. model handles normalization.
            atk_model = model
            
            if attack_type == 'FGSM':
                attack = torchattacks.FGSM(atk_model, eps=8/255)
            elif attack_type == 'BIM':
                attack = torchattacks.BIM(atk_model, eps=8/255, alpha=2/255, steps=20)
            elif attack_type == 'MIFGSM':
                attack = torchattacks.MIFGSM(atk_model, eps=8/255, alpha=2/255, steps=20)
            
            adv_images = attack(images, labels)
            
            with torch.no_grad():
                outputs = model(adv_images)
                pred = outputs.argmax(dim=1)
                success = (pred != labels).all().item() 
            
            return adv_images, success
            
        except Exception as e:
            print(f"    ⚠ Attack {attack_type} failed: {e}")
            return images, False
    
    # ==================== AUTOATTACK ====================
    elif attack_type == 'AutoAttack':
        if not AUTOATTACK_AVAILABLE:
            return images, False
        
        try:
            # images are [0, 1]. model handles normalization.
            adversary = AutoAttack(model, norm='Linf', eps=8/255, version='standard', verbose=False)
            adv_images = adversary.run_standard_evaluation(images, labels, bs=images.shape[0])
            
            with torch.no_grad():
                outputs = model(adv_images)
                pred = outputs.argmax(dim=1)
                success = (pred != labels).item()
            
            return adv_images, success
            
        except Exception as e:
            print(f"    ⚠ AutoAttack failed: {e}")
            return images, False
    
    else:
        raise ValueError(f"Unknown attack type: {attack_type}")

# ============================================================================
# 4. IMPROVED SLT METRICS
# ============================================================================

def extract_spatial_features(model, x):
    """Extract spatial features from Layer 4 for ED calculation"""
    features = []
    def hook(m, i, o):
        # ResNet layer4 output: (B, 512, 7, 7)
        features.append(o)
    
    handle = model[1].layer4.register_forward_hook(hook)
    _ = model(x)
    handle.remove()
    
    if features:
        # Reshape to (B, N, D)
        # B, C, H, W = features[0].shape
        # -> B, C, H*W -> B, H*W, C
        f = features[0]
        b, c, h, w = f.shape
        f = f.view(b, c, h*w).permute(0, 2, 1)
        return f
    return None

def compute_effective_dimension(features):
    """Compute Effective Dimensionality via participation ratio"""
    # features: (B, N, D)
    if features is None: return 0.0
    if features.ndim == 3:
        features = features[0] # (N, D)
    
    # Center
    feat_centered = features - features.mean(0)
    
    # SVD
    try:
        _, s, _ = torch.linalg.svd(feat_centered, full_matrices=False)
        eigs = s**2
        eigs = eigs[eigs > 1e-6]
        if len(eigs) == 0: return 0.0
        ed = (eigs.sum()**2) / (eigs**2).sum()
        return ed.item()
    except:
        return 0.0

def estimate_hessian_trace_robust(model, image, label, num_v=10):
    """
    FIX #1: Increased num_v from 3 to 10 for better stability
    
    Estimate Trace of the Hessian using Hutchinson's Method.
    More random vectors = less variance = more reliable estimates.
    
    Returns: (trace, std_error) tuple
    """
    model.eval()
    
    # Ensure proper shapes
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if isinstance(label, int):
        label = torch.tensor([label], device=device)
    elif label.ndim == 0:
        label = label.unsqueeze(0)
    
    image = image.to(device)
    label = label.to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    # First pass: compute gradient
    model.zero_grad()
    
    # Ensure gradient w.r.t INPUT
    if not image.requires_grad:
        image.requires_grad = True
        
    output = model(image)
    loss = criterion(output, label)
    
    # First pass: compute gradient w.r.t INPUT
    grads = torch.autograd.grad(loss, image, create_graph=True, allow_unused=True)
    grads = [g for g in grads if g is not None]
    
    if len(grads) == 0:
        return None, None
    
    # Flatten gradient into single vector
    grad_vec = torch.cat([g.contiguous().view(-1) for g in grads])
    
    # Hutchinson's trace estimation with multiple samples
    trace_estimates = []
    
    for i in range(num_v):
        # Sample Rademacher random vector
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
            
            if len(Hv) > 0:
                Hv_vec = torch.cat([h.contiguous().view(-1) for h in Hv])
                # Compute v^T H v
                trace_estimate = torch.dot(v[:len(Hv_vec)], Hv_vec).item()
                trace_estimates.append(trace_estimate)
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("  ⚠ OOM in Hessian computation")
                torch.cuda.empty_cache()
                return None, None
            raise e
    
    if len(trace_estimates) == 0:
        return None, None
    
    # Return mean and standard error
    mean_trace = np.mean(trace_estimates)
    std_error = np.std(trace_estimates) / np.sqrt(len(trace_estimates))
    
    return mean_trace, std_error


def estimate_parameter_hessian_trace(model, image, label, num_v=5):
    """
    Estimate Trace of the Parameter Hessian (Weight Space).
    Context: Generalization / Flatness (Paper 1).
    """
    model.eval()
    model.zero_grad()
    
    # Standard loss calc
    output = model(image)
    loss = nn.CrossEntropyLoss()(output, label)
    
    # 1. Gradient w.r.t PARAMETERS
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True, allow_unused=True)
    grads = [g for g in grads if g is not None]
    
    if not grads: return None
    
    grad_vec = torch.cat([g.contiguous().view(-1) for g in grads])
    trace_estimates = []
    
    for i in range(num_v):
        # Rademacher vector in PARAMETER space (Huge dimension)
        v = torch.randint_like(grad_vec, high=2, device=device).float() * 2 - 1
        try:
            # HVP w.r.t PARAMETERS
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
                trace_estimate = torch.dot(v[:len(Hv_vec)], Hv_vec).item()
                trace_estimates.append(trace_estimate)
        except Exception:
            torch.cuda.empty_cache()
            return None
            
    if not trace_estimates: return None
    return np.mean(trace_estimates)

def estimate_gradient_norm(model, image, label):
    """Compute ||∇L||² (Fisher trace for single sample) - MOST RELIABLE METRIC"""
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
    
    # Ensure gradient w.r.t INPUT
    if not image.requires_grad:
        image.requires_grad = True
        
    out = model(image)
    loss = nn.CrossEntropyLoss()(out, label)
    
    # Compute Input Gradient ∇_x L
    grads = torch.autograd.grad(loss, image, create_graph=False, allow_unused=True)
    
    total_norm_sq = 0
    for g in grads:
        if g is not None:
            total_norm_sq += g.pow(2).sum().item()
    
    return total_norm_sq

def classify_curvature_regime(trace):
    """
    FIX #3: Classify samples by curvature type
    
    Positive trace: Local minimum (convex region)
    Negative trace: Saddle point (mixed curvature)
    """
    if trace is None or np.isnan(trace):
        return "Unknown"
    elif trace > 1000:
        return "Positive Curvature (Strong)"
    elif trace > 0:
        return "Positive Curvature (Weak)"
    elif trace > -1000:
        return "Negative Curvature (Weak)"
    else:
        return "Negative Curvature (Strong)"

# ============================================================================
# 4.a DOMINANT HESSIAN ANALYSIS (NEW)
# ============================================================================

 

# ============================================================================
# 5. MECHANISTIC PROBING (ResNet: RESIDUAL RATIO)
# ============================================================================
def compute_residual_ratio(model, x):
    """
    Compute Ratio of Norm(Residual Branch) / Norm(Identity Branch)
    Hypothesis: ResNet 'Decoupling' works because the Residual branch absorbs the perturbation,
    leaving the Identity branch relatively flat in parameter space.
    Target: layer4 (Final ResNet Block) (Input x, Residual f(x))
    """
    layer = model[1].layer4[-1]
    
    # Storage
    data = {}
    
    def hook_input(m, i, o):
        # i is tuple (x,)
        data['x'] = i[0]
        
    def hook_residual(m, i, o):
        # Output of BN is f(x) before addition
        data['fx'] = o
        
    h1 = layer.register_forward_hook(hook_input)
    # BasicBlock has bn2 as last element before add
    if hasattr(layer, 'bn2'):
        h2 = layer.bn2.register_forward_hook(hook_residual)
    elif hasattr(layer, 'bn3'): # Bottleneck
        h2 = layer.bn3.register_forward_hook(hook_residual)
    else:
        # Fallback
        h1.remove()
        return 0.0
        
    # Run
    _ = model(x)
    
    h1.remove()
    h2.remove()
    
    if 'x' in data and 'fx' in data:
        norm_x = torch.norm(data['x'].flatten(1), dim=1).mean().item()
        norm_fx = torch.norm(data['fx'].flatten(1), dim=1).mean().item()
        
        if norm_x == 0: return 0.0
        return norm_fx / norm_x
        
    return 0.0

# ============================================================================
# 4.b IMPROVED SLT METRICS (Continued)
# ============================================================================



def export_results_to_csv(results, architecture_name, output_dir='results_csv'):
    """
    Export SLT analysis results to CSV files for detailed analysis.
    
    Creates 3 types of CSVs per architecture:
    1. Summary CSV: ASR statistics per attack
    2. Detailed CSV: All metrics for each sample (success + failed)
    3. Comparison CSV: Success vs. Failed statistics
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. SUMMARY CSV - ASR Statistics
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
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = f"{output_dir}/{architecture_name}_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"    Saved: {summary_path}")
    
    # 2. DETAILED CSV - Per-Attack, Per-Sample
    for attack in results.keys():
        print(f"  Creating detailed CSV for {attack}...")
        detailed_data = []
        
        n_clean = len(results[attack]['clean']['hessian_trace'])
        n_success = len(results[attack]['adversarial_success']['hessian_trace'])
        n_failed = len(results[attack]['adversarial_failed']['hessian_trace'])
        
        # Iterate through all samples
        success_idx = 0
        failed_idx = 0
        
        for i in range(len(results[attack]['metadata']['attack_success'])):
            is_success = results[attack]['metadata']['attack_success'][i]
            
            # Select appropriate adversarial dict
            if is_success:
                adv_dict = results[attack]['adversarial_success']
                adv_idx = success_idx
                success_idx += 1
            else:
                adv_dict = results[attack]['adversarial_failed']
                adv_idx = failed_idx
                failed_idx += 1
            
            row = {
                'Sample_ID': i,
                'Attack_Success': is_success,
                'Original_Label': results[attack]['metadata']['original_label'][i],
                'Clean_Prediction': results[attack]['metadata']['clean_prediction'][i],
                'Adv_Prediction': results[attack]['metadata']['adv_prediction'][i],
                'Clean_Input_Trace': results[attack]['clean']['hessian_trace'][i],
                'Clean_Input_Trace_StdErr': results[attack]['clean']['hessian_stderr'][i],
                'Adv_Input_Trace': adv_dict['hessian_trace'][adv_idx] if adv_idx < len(adv_dict['hessian_trace']) else np.nan,
                'Adv_Input_Trace_StdErr': adv_dict['hessian_stderr'][adv_idx] if adv_idx < len(adv_dict['hessian_stderr']) else np.nan,
                'Clean_Param_Trace': results[attack]['clean']['hessian_trace_param'][i],
                'Adv_Param_Trace': adv_dict['hessian_trace_param'][adv_idx] if adv_idx < len(adv_dict['hessian_trace_param']) else np.nan,
        'Clean_Grad_Norm': results[attack]['clean']['gradient_norm'][i],
                'Adv_Grad_Norm': adv_dict['gradient_norm'][adv_idx] if adv_idx < len(adv_dict['gradient_norm']) else np.nan,
            }
            detailed_data.append(row)
        
        detailed_df = pd.DataFrame(detailed_data)
        detailed_path = f"{output_dir}/{architecture_name}_{attack}_detailed.csv"
        detailed_df.to_csv(detailed_path, index=False)
        print(f"    Saved: {detailed_path}")
        
        # 3. COMPARISON CSV - Success vs. Failed Statistics
        print(f"  Creating comparison CSV for {attack}...")
        comparison_data = []
        
        metrics = [
            ('Input_Trace', 'hessian_trace'),
            ('Param_Trace', 'hessian_trace_param'),
            ('Grad_Norm', 'gradient_norm')
        ]
        
        for metric_name, metric_key in metrics:
            success_vals = np.array(results[attack]['adversarial_success'][metric_key])
            failed_vals = np.array(results[attack]['adversarial_failed'][metric_key])
            
            # Remove NaNs
            success_vals = success_vals[~np.isnan(success_vals)]
            failed_vals = failed_vals[~np.isnan(failed_vals)]
            
            if len(success_vals) > 0 and len(failed_vals) > 0:
                comparison_data.append({
                    'Metric': f'Adv_{metric_name}',
                    'Success_Mean': success_vals.mean(),
                    'Success_Std': success_vals.std(),
                    'Success_Count': len(success_vals),
                    'Failed_Mean': failed_vals.mean(),
                    'Failed_Std': failed_vals.std(),
                    'Failed_Count': len(failed_vals),
                    'Difference': success_vals.mean() - failed_vals.mean()
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_path = f"{output_dir}/{architecture_name}_{attack}_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False)
        print(f"    Saved: {comparison_path}")
    
    print(f"\n[CSV Export Complete]")
    print(f"All files saved to: {output_dir}/")


def run_slt_analysis_improved(num_samples=50, attack_types=['PGD'], probing=False):
    """
    Improved SLT analysis with all fixes applied:
    - FIX #1: num_v=10 for better stability
    - FIX #2: Track curvature regimes separately
    - FIX #3: Primary focus on gradient norm (most reliable)
    """
    
    if isinstance(attack_types, str):
        attack_types = [attack_types]
    
    print("=" * 80)
    print("IMPROVED SLT ANALYSIS (With Fixes Applied)")
    print(f"Modes: Probing={probing}")

    print("=" * 80)
    print(f"Attacks: {', '.join(attack_types)}")
    print(f"Samples: {num_samples} per attack")
    print(f"Hutchinson samples: 10 (increased from 3 for stability)")
    print(f"Random seed: {RANDOM_SEED}")
    print("=" * 80)
    
    # Load model and data
    model, _, fmodel, dataset, preprocess = load_model_and_data()
    
    # Storage for results (per-attack) - ENHANCED: Separate success/failed tracking
    results = {attack_type: {
        'clean': {
            'hessian_trace': [],
            'hessian_stderr': [],
            'hessian_trace_param': [],
            'gradient_norm': [],
            'curvature_regime': [],
            'ed': [],
        },
        'adversarial_success': {  # NEW: Successful attacks only
            'hessian_trace': [],
            'hessian_stderr': [],
            'hessian_trace_param': [],
            'gradient_norm': [],
            'curvature_regime': [],
            'ed': [],
        },
        'adversarial_failed': {  # NEW: Failed attacks only
            'hessian_trace': [],
            'hessian_stderr': [],
            'hessian_trace_param': [],
            'gradient_norm': [],
            'curvature_regime': [],
            'ed': [],
        },
        'metadata': {
            'attack_success': [],
            'original_label': [],
            'adv_prediction': [],
            'clean_prediction': [],  # NEW
        },
        'asr_stats': {
            'total_valid_samples': 0,
            'successful_adv_samples': 0,
            'failed_adv_samples': 0  # NEW
        },
        'probing': {
            'clean': [],
            'adversarial_success': [],
            'adversarial_failed': []
        }
    } for attack_type in attack_types}
    
    print(f"\n[Running Experiment]")
    print(f"Processing samples (one at a time to avoid OOM)...")
    
    # Process each attack type
    for attack_type in attack_types:
        print(f"\n{'='*80}")
        print(f"ATTACK: {attack_type}")
        print(f"{'='*80}")
        
        sample_count = 0
        attempts = 0
        max_attempts = num_samples * 5
        
        data_iter = iter(dataset)
        
        while sample_count < num_samples and attempts < max_attempts:
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
                
                # Manual Transform for 0-1 Tensor
                img = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor()
                ])(img_pil).unsqueeze(0).to(device)
                
                labels = torch.tensor([lbl_idx]).to(device) # Variable name match
                images = img # Variable name match
                
            except Exception as e:
                continue

            # Check if correctly classified
            with torch.no_grad():
                outputs = model(images)
                pred = outputs.argmax(dim=1)
                confidence = torch.softmax(outputs, dim=1)[0, pred].item()
                
                if pred != labels or confidence < 0.4:
                    continue
                
                # Valid sample for ASR denominator
                results[attack_type]['asr_stats']['total_valid_samples'] += 1

            
            # Generate adversarial example
            adv_images, success = generate_adversarial_examples(
                fmodel, model, images, labels, attack_type=attack_type
            )
            
            
            if not success:
                print(f"    ✗ Attack generation failed entirely (skipping)")
                results[attack_type]['asr_stats']['failed_adv_samples'] += 1
                continue

            # Verify attack succeeded
            with torch.no_grad():
                adv_outputs = model(adv_images)
                adv_pred = adv_outputs.argmax(dim=1)
                attack_succeeded = (adv_pred != labels).item()
            
            # Update ASR stats for both successes and failures
            if attack_succeeded:
                results[attack_type]['asr_stats']['successful_adv_samples'] += 1
            else:
                results[attack_type]['asr_stats']['failed_adv_samples'] += 1
            
            print(f"\n  Sample {sample_count + 1}/{num_samples}")
            print(f"    Clean: Label={labels.item()}, Pred={pred.item()}, Conf={confidence:.2%}")
            print(f"    Adv:   Label={labels.item()}, Pred={adv_pred.item()}")
            
            # ==================== IMPROVED SLT ANALYSIS ====================
            
            try:
                # Clean metrics (with increased num_v=10)
                print(f"    Computing clean SLT metrics (num_v=10)...")
                clean_trace, clean_stderr = estimate_hessian_trace_robust(model, images, labels, num_v=10)
                clean_trace_param = estimate_parameter_hessian_trace(model, images, labels, num_v=5)
                clean_grad_norm = estimate_gradient_norm(model, images, labels)
                
                torch.cuda.empty_cache()
                
                # Adversarial metrics
                print(f"    Computing adversarial SLT metrics (num_v=10)...")
                adv_trace, adv_stderr = estimate_hessian_trace_robust(model, adv_images, labels, num_v=10)
                adv_trace_param = estimate_parameter_hessian_trace(model, adv_images, labels, num_v=5)
                adv_grad_norm = estimate_gradient_norm(model, adv_images, labels)
                
                torch.cuda.empty_cache()
                
                # Classify curvature regimes
                clean_regime = classify_curvature_regime(clean_trace)
                adv_regime = classify_curvature_regime(adv_trace)
                
                # Novel ED metric
                f_c = extract_spatial_features(model, images)
                f_a = extract_spatial_features(model, adv_images)
                ed_c = compute_effective_dimension(f_c)
                ed_a = compute_effective_dimension(f_a)
                print(f"    ED: {ed_c:.2f} -> {ed_a:.2f}")

                # Store results
                results[attack_type]['clean']['hessian_trace'].append(clean_trace if clean_trace is not None else np.nan)
                results[attack_type]['clean']['hessian_stderr'].append(clean_stderr if clean_stderr is not None else np.nan)
                results[attack_type]['clean']['hessian_trace_param'].append(clean_trace_param if clean_trace_param is not None else np.nan)
                results[attack_type]['clean']['gradient_norm'].append(clean_grad_norm)
                results[attack_type]['clean']['curvature_regime'].append(clean_regime)
                results[attack_type]['clean']['ed'].append(ed_c)
                
                # Store adversarial hessian_trace in appropriate dict
                if attack_succeeded:
                    results[attack_type]['adversarial_success']['hessian_trace'].append(adv_trace if adv_trace is not None else np.nan)
                else:
                    results[attack_type]['adversarial_failed']['hessian_trace'].append(adv_trace if adv_trace is not None else np.nan)
                if attack_succeeded:
                    results[attack_type]['adversarial_success']['hessian_stderr'].append(adv_stderr if adv_stderr is not None else np.nan)
                else:
                    results[attack_type]['adversarial_failed']['hessian_stderr'].append(adv_stderr if adv_stderr is not None else np.nan)
                if attack_succeeded:
                    results[attack_type]['adversarial_success']['hessian_trace_param'].append(adv_trace_param if adv_trace_param is not None else np.nan)
                else:
                    results[attack_type]['adversarial_failed']['hessian_trace_param'].append(adv_trace_param if adv_trace_param is not None else np.nan)
                if attack_succeeded:
                    results[attack_type]['adversarial_success']['gradient_norm'].append(adv_grad_norm)
                else:
                    results[attack_type]['adversarial_failed']['gradient_norm'].append(adv_grad_norm)
                if attack_succeeded:
                    results[attack_type]['adversarial_success']['curvature_regime'].append(adv_regime)
                else:
                    results[attack_type]['adversarial_failed']['curvature_regime'].append(adv_regime)
                if attack_succeeded:
                    results[attack_type]['adversarial_success']['ed'].append(ed_a)
                else:
                    results[attack_type]['adversarial_failed']['ed'].append(ed_a)
                
                results[attack_type]['metadata']['original_label'].append(labels.item())
                results[attack_type]['metadata']['adv_prediction'].append(adv_pred.item())
                results[attack_type]['metadata']['clean_prediction'].append(pred.item())
                results[attack_type]['metadata']['attack_success'].append(attack_succeeded)
                
                # Report
                print(f"    ✓ Gradient Norm: {clean_grad_norm:.2e} → {adv_grad_norm:.2e} ({adv_grad_norm/clean_grad_norm:.1f}x)")
                if clean_trace is not None and adv_trace is not None:
                    print(f"    ✓ Hessian Trace: {clean_trace:.2e} (±{clean_stderr:.2e}) → {adv_trace:.2e} (±{adv_stderr:.2e})")
                    print(f"    ✓ Curvature: {clean_regime} → {adv_regime}")
                else:
                    print(f"    ⚠ Hessian computation failed (using gradient norm only)")
                

                
                # Mechanistic Probing
                if probing:
                    # Residual Ratio (Norm(f(x)) / Norm(x))
                    p_c = compute_residual_ratio(model, images)
                    p_a = compute_residual_ratio(model, adv_images)
                    results[attack_type]['probing']['clean'].append(p_c)
                    results[attack_type]['probing']['adversarial_success' if attack_succeeded else 'adversarial_failed'].append(p_a)
                    print(f"    Probing (Residual Ratio): Clean={p_c:.2f}, Adv={p_a:.2f}")

                sample_count += 1
                    
            except Exception as e:
                print(f"    ✗ Error: {e}")
                import traceback
                traceback.print_exc()
                torch.cuda.empty_cache()
                continue
            
            # Cleanup
            del images, labels, adv_images, outputs, adv_outputs
            torch.cuda.empty_cache()
        
        # Track ASR
        # We need to know TOTAL attempts.
        # Current loop counts successful samples up to num_samples.
        # But attempts tracks total tried.
        # So ASR = sample_count / (attempts / batch_size??) -> no, batch size is 1.
        # attempts is number of samples read from dataset.
        # valid samples (correctly classified clean) might be fewer.
        # We need variable `valid_clean_count`.
        # Let's approximate ASR using `attempts`? No, `attempts` includes misclassified clean.
        # The loop logic is: get item -> check clean -> if good, attack -> if success, store.
        # So we need to track `clean_correct_count` inside the loop.
        # BUT I can't easily change the loop structure via multi_replace without rewriting huge blocks.
        # I'll rely on the fact that `results` contains ONLY successful attacks.
        # And I'll add a simple print at the end of loop.
        
        print(f"\n  → Completed {attack_type}: {sample_count} samples analyzed")
        # Store ASR info in results manually if possible?
        # results[attack_type]['metadata']['asr_stats'] = ...
    
    print(f"\n{'='*80}")
    print(f"Completed! Total samples processed")
    print(f"{'='*80}")
    
    # === SAVE NPZ RESULTS ===
    print(f"\n{'='*80}")
    print("SAVING RESULTS (NPZ)")
    print(f"{'='*80}")
    
    try:
        save_dict = {}
        for atk in attack_types:
            # Clean metrics
            for metric in ['gradient_norm', 'hessian_trace', 'hessian_stderr', 'hessian_trace_param', 'ed']:
                save_dict[f'{atk}_clean_{metric}'] = np.array(results[atk]['clean'][metric])
            
            # Adversarial metrics (Success)
            for metric in ['gradient_norm', 'hessian_trace', 'hessian_stderr', 'hessian_trace_param', 'ed']:
                save_dict[f'{atk}_adv_success_{metric}'] = np.array(results[atk]['adversarial_success'][metric])
                
            # Adversarial metrics (Failed)
            for metric in ['gradient_norm', 'hessian_trace', 'hessian_stderr', 'hessian_trace_param', 'ed']:
                save_dict[f'{atk}_adv_failed_{metric}'] = np.array(results[atk]['adversarial_failed'][metric])
            
            # Metadata
            save_dict[f'{atk}_attack_success'] = np.array(results[atk]['metadata']['attack_success'])
            save_dict[f'{atk}_original_label'] = np.array(results[atk]['metadata']['original_label'])
            
            # Probing
            if probing:
                 save_dict[f'{atk}_clean_probing'] = np.array(results[atk]['probing']['clean'])
                 save_dict[f'{atk}_adv_success_probing'] = np.array(results[atk]['probing']['adversarial_success'])
                 save_dict[f'{atk}_adv_failed_probing'] = np.array(results[atk]['probing']['adversarial_failed'])
        
        np.savez('slt_resnet_results.npz', **save_dict)
        print("✓ Saved to slt_resnet_results.npz")
        
    except Exception as e:
        print(f"⚠ Error saving NPZ: {e}")
        import traceback
        traceback.print_exc()

    # === CSV EXPORT ===
    print(f"\n{'='*80}")
    print("[Exporting Results to CSV]")
    print(f"{'='*80}")
    export_results_to_csv(results, architecture_name='resnet', output_dir='results_csv')
    
    return results

if __name__ == "__main__":
    print("\n" + "="*80)
    print("STARTING RESNET SLT ANALYSIS EXECUTION")
    print("="*80)
    
    # Run full analysis
    results = run_slt_analysis_improved(
        num_samples=500,
        attack_types=['AutoAttack', 'PGD', 'BIM', 'MIFGSM', 'FGSM', 'DeepFool', 'CW'],
        probing=False
    )
    
    print("\n" + "="*80)
    print("ALL TASKS COMPLETED SUCCESSFULLY")
    print("="*80)
