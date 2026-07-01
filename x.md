
## PHẦN IV: DIAGNOSTIC POWER — FRAMEWORK HeTRoM

### 4.1. Tổng Quan Framework HeTRoM

HeTRoM (Heterogeneous Task Robustness Metrics) là framework đánh giá khả năng **chẩn đoán** của một XAI method dưới các điều kiện task khác nhau. Không giống như các bài test faithfulness (đo tương quan với model mechanics), HeTRoM đo **tính hữu dụng chẩn đoán** — liệu saliency map có giúp người dùng hiểu **TẠI SAO** adaptation thất bại hay thành công?

Ba kịch bản được thiết kế để mô phỏng các tình huống thực tế trong few-shot learning:

| Kịch bản | Mô tả | Expectation FAMA |
|----------|-------|-----------------|
| **Hard Task** | $\delta M$ nhỏ hoặc dương — adaptation thất bại | FAMA chỉ ra nhiều vùng harmful |
| **Noisy Task** | Một số shots bị sai nhãn — support set bị nhiễu | FAMA xếp noisy shots ở vị trí thấp (attribution âm) |
| **OOD Task** | Support set từ phân phối khác training — domain shift | FAMA chỉ ra high-magnitude attribution toàn bộ (high uncertainty) |

#### Tại sao ba kịch bản này quan trọng?

**Hard Task**: Trong deployment, không phải mọi task đều dễ thích nghi. Reviewer sẽ hỏi: "FAMA có thể giúp người dùng hiểu tại sao adaptation thất bại không?" Nếu FAMA chỉ ra vùng gây hại nhiều trong hard tasks, nó đang hoạt động như một diagnostic tool thực sự.

**Noisy Task**: Label noise trong few-shot learning rất phổ biến trong thực tế (ảnh từ web không được kiểm duyệt, labeling errors). Nếu FAMA có thể phát hiện shots bị sai nhãn thông qua negative attribution, đây là ứng dụng **practical** cực kỳ thuyết phục với reviewer.

**OOD Task**: Domain shift là vấn đề lớn trong meta-learning deployment. Một diagnostic tool tốt phải cảnh báo khi support set khác quá nhiều với training distribution.

---

### 4.2. Kịch Bản 1: Hard Task Analysis

#### Định nghĩa Hard Task

Task được phân loại là "Hard" khi:
$$\delta M(\theta_0, S) > \tau_{hard}$$

với $\tau_{hard}$ là ngưỡng — adaptation không cải thiện (loss không giảm đáng kể hoặc tăng). Thực tiễn: $\tau_{hard} = -0.05$ (loss giảm < 5% là "hard").

Task "Easy": $\delta M < \tau_{easy} = -0.3$ (loss giảm > 30%).

#### Metrics cho Hard Task

**Negative Attribution Ratio (NAR)**:
$$\text{NAR} = \frac{\sum_p \max(-\mathbf{A}(p), 0)}{\sum_p |\mathbf{A}(p)| + \epsilon} = \frac{\|\mathbf{A}^-\|_1}{\|\mathbf{A}\|_1}$$

Kỳ vọng: NAR(Hard) > NAR(Easy) — hard tasks có nhiều vùng harmful hơn.

**Attribution Magnitude (AM)**:
$$\text{AM} = \|\mathbf{A}\|_1 / (K \cdot N \cdot H \cdot W)$$

Kỳ vọng: AM(Hard) > AM(Easy) — attribution cao hơn khi adaptation struggle (high gradient magnitude).

**Statistical Test**: Mann-Whitney U test giữa NAR distribution của Hard tasks và Easy tasks. p-value < 0.05 → có sự khác biệt có ý nghĩa thống kê.

---

### 4.3. Kịch Bản 2: Noisy Task Analysis

#### Thiết Kế Thí Nghiệm Noisy Task

**Label Flip**: Chọn ngẫu nhiên $n_{noisy}$ shots trong support set và đổi nhãn của chúng sang nhãn sai (nhãn của class khác, chọn ngẫu nhiên trong $N$ classes).

**Label Noise Rate** $\rho = n_{noisy} / (K \cdot N)$: Thử nghiệm với $\rho \in \{0.1, 0.2, 0.4\}$.

#### Metric Chính: Label Noise Detection Rate (LNDR)

$$\text{LNDR}(\rho) = \Pr\left[\text{shot với nhãn sai được xếp thấp nhất trong class của nó}\right]$$

Cụ thể, với mỗi task có $n_{noisy}$ noisy shots:
1. Tính shot-level attribution: $a_k = \text{mean}_{p}[\mathbf{A}(x_k, p)]$
2. Kiểm tra: noisy shot có $a_k$ thấp nhất (âm nhất) trong tất cả shots không?

$\text{LNDR} = \frac{\text{số task noisy shot được rank thấp nhất}}{n_{tasks}}$

**Baseline LNDR (chance level)**: $1/(K \cdot N)$ — e.g., với 5-way 5-shot: 1/25 = 0.04.

**Acceptance**: LNDR > 0.3 (7.5× better than chance) → FAMA có diagnostic power thực sự.

#### Metric Phụ: Shot Attribution Rank Correlation (SARC)

Với mỗi task, tính Spearman correlation giữa:
- **FAMA shot attribution** $a_k = \text{mean}_p[\mathbf{A}(x_k, p)]$
- **Shot correctness indicator**: $c_k = 1$ nếu shot $k$ có nhãn đúng, $c_k = 0$ nếu noisy

$\text{SARC} = \rho_s(\{a_k\}, \{c_k\})$ — dương → shots có nhãn đúng có attribution cao hơn.

---

### 4.4. Kịch Bản 3: OOD Task Analysis

#### Thiết Kế Thí Nghiệm OOD Task

**OOD Support Set Creation**:
- **Method 1 – Style Transfer OOD**: Áp dụng artistic style transfer lên support images để tạo domain shift (natural → artistic)
- **Method 2 – Dataset OOD**: Dùng support set từ dataset khác (e.g., support từ CUB-200, query từ miniImageNet)
- **Method 3 – Texture/Color OOD**: Áp dụng random color jitter mạnh (saturation=5×, hue=0.5) lên support set

Trong thực tế: Method 2 và Method 3 dễ implement hơn. Sử dụng **Method 3** (strong color jitter) làm thí nghiệm chính.

#### Metric: OOD Attribution Shift (OAS)

So sánh distribution của attribution magnitude giữa In-Distribution (ID) và OOD support:

$$\text{OAS} = \frac{\mathbb{E}_{task \sim OOD}[\|\mathbf{A}\|_1]}{\mathbb{E}_{task \sim ID}[\|\mathbf{A}\|_1]}$$

Kỳ vọng: OAS > 1 — FAMA attribution có magnitude cao hơn với OOD support set (model "confused" hơn → gradient lớn hơn).

#### Metric: OOD Detection Accuracy (ODA)

Nếu dùng FAMA attribution magnitude như một OOD detector:
$$\text{score}(S) = \|\mathbf{A}(S)\|_1$$

Áp dụng ngưỡng $\tau$ và tính AUROC của việc phân biệt ID vs. OOD:

$$\text{ODA} = \text{AUROC}\left(\{(\|\mathbf{A}(S_{ID})\|_1, 0)\} \cup \{(\|\mathbf{A}(S_{OOD})\|_1, 1)\}\right)$$

AUROC > 0.7 → FAMA attribution có thể được dùng như OOD signal.

---

### 4.5. Source Code: HeTRoMDiagnostic

```python
# fama/evaluation/hetrom_diagnostic.py

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
import copy
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from sklearn.metrics import roc_auc_score
import scipy.stats


@dataclass
class HeTRoMConfig:
    """Configuration cho HeTRoM Diagnostic."""
    # Hard/Easy Task
    tau_hard: float = -0.05      # δM > tau_hard → Hard task
    tau_easy: float = -0.30      # δM < tau_easy → Easy task
    n_tasks_classification: int = 500  # Tasks để phân loại Hard/Easy
    
    # Noisy Task
    noise_rates: List[float] = None  # None → [0.1, 0.2, 0.4]
    n_noisy_tasks: int = 100
    
    # OOD Task
    n_ood_tasks: int = 100
    ood_color_jitter_strength: float = 3.0  # Magnitude của color jitter
    
    def __post_init__(self):
        if self.noise_rates is None:
            self.noise_rates = [0.1, 0.2, 0.4]


class HardEasyTaskAnalyzer:
    """Phân tích FAMA attribution trên Hard vs. Easy tasks."""
    
    def __init__(self, adapter: MAMLAdapter, config: HeTRoMConfig):
        self.adapter = adapter
        self.config = config
    
    def classify_tasks(
        self,
        theta0: Dict[str, torch.Tensor],
        tasks: List[Dict],
    ) -> Tuple[List[Dict], List[Dict], List[float]]:
        """
        Phân loại tasks thành Hard, Easy, và Medium.
        
        Returns:
            (hard_tasks, easy_tasks, all_delta_M)
        """
        hard_tasks, easy_tasks, all_delta_M = [], [], []
        
        print("Classifying tasks by adaptation difficulty...")
        for i, task in enumerate(tasks):
            delta_M = self.adapter.compute_adaptation_gain(
                theta0,
                task['support_x'], task['support_y'],
                task['query_x'], task['query_y']
            ).item()
            all_delta_M.append(delta_M)
            
            if delta_M > self.config.tau_hard:
                hard_tasks.append(task)
            elif delta_M < self.config.tau_easy:
                easy_tasks.append(task)
            # Medium tasks bỏ qua để phân tích rõ ràng hơn
        
        print(f"Classified {len(tasks)} tasks:")
        print(f"  Hard (δM > {self.config.tau_hard}): {len(hard_tasks)}")
        print(f"  Easy (δM < {self.config.tau_easy}): {len(easy_tasks)}")
        print(f"  Medium: {len(tasks) - len(hard_tasks) - len(easy_tasks)}")
        
        return hard_tasks, easy_tasks, all_delta_M
    
    def compute_attribution_stats(
        self,
        theta0: Dict[str, torch.Tensor],
        tasks: List[Dict],
        fama_fn: callable
    ) -> Dict:
        """Tính NAR và AM cho một list of tasks."""
        nars, ams, pos_ratios, neg_ratios = [], [], [], []
        
        for task in tasks:
            saliency = fama_fn(theta0, task)  # [K*N, H, W]
            
            total = saliency.numel()
            pos_mass = saliency[saliency > 0].sum().item()
            neg_mass = (-saliency[saliency < 0]).sum().item()
            total_abs = pos_mass + neg_mass + 1e-8
            
            nar = neg_mass / total_abs
            am  = total_abs / total
            pos_ratio = (saliency > 0).float().mean().item()
            neg_ratio = (saliency < 0).float().mean().item()
            
            nars.append(nar)
            ams.append(am)
            pos_ratios.append(pos_ratio)
            neg_ratios.append(neg_ratio)
        
        return {
            'nar_mean': float(np.mean(nars)),
            'nar_std':  float(np.std(nars)),
            'am_mean':  float(np.mean(ams)),
            'am_std':   float(np.std(ams)),
            'pos_ratio_mean': float(np.mean(pos_ratios)),
            'neg_ratio_mean': float(np.mean(neg_ratios)),
            'raw_nars': nars,
            'raw_ams': ams
        }
    
    def run(
        self,
        theta0: Dict[str, torch.Tensor],
        tasks: List[Dict],
        fama_fn: callable
    ) -> Dict:
        """Chạy Hard/Easy task analysis."""
        
        # 1. Phân loại tasks
        hard_tasks, easy_tasks, all_delta_M = self.classify_tasks(theta0, tasks)
        
        if len(hard_tasks) < 5 or len(easy_tasks) < 5:
            print("Warning: Not enough hard or easy tasks. Adjust tau thresholds.")
        
        # 2. Tính attribution stats cho mỗi nhóm
        print("\nComputing attribution stats for Hard tasks...")
        hard_stats = self.compute_attribution_stats(theta0, hard_tasks[:50], fama_fn)
        
        print("Computing attribution stats for Easy tasks...")
        easy_stats = self.compute_attribution_stats(theta0, easy_tasks[:50], fama_fn)
        
        # 3. Statistical test (Mann-Whitney U)
        if hard_stats['raw_nars'] and easy_stats['raw_nars']:
            stat, p_value = scipy.stats.mannwhitneyu(
                hard_stats['raw_nars'],
                easy_stats['raw_nars'],
                alternative='greater'  # H1: Hard tasks have higher NAR
            )
        else:
            stat, p_value = None, None
        
        # 4. Correlation giữa δM và NAR
        # Lấy tasks ở cả hai nhóm, tính correlation
        mixed_tasks = hard_tasks[:25] + easy_tasks[:25]
        mixed_delta_M = []
        mixed_nar = []
        
        for task in mixed_tasks:
            dm = self.adapter.compute_adaptation_gain(
                theta0, task['support_x'], task['support_y'],
                task['query_x'], task['query_y']
            ).item()
            saliency = fama_fn(theta0, task)
            pos_mass = saliency[saliency > 0].sum().item()
            neg_mass = (-saliency[saliency < 0]).sum().item()
            nar = neg_mass / (pos_mass + neg_mass + 1e-8)
            
            mixed_delta_M.append(dm)
            mixed_nar.append(nar)
        
        corr_r, corr_p = scipy.stats.spearmanr(mixed_delta_M, mixed_nar)
        
        results = {
            'hard_stats': hard_stats,
            'easy_stats': easy_stats,
            'n_hard_tasks': len(hard_tasks),
            'n_easy_tasks': len(easy_tasks),
            'mannwhitney_pvalue': float(p_value) if p_value is not None else None,
            'nar_significant': (p_value is not None and p_value < 0.05),
            'delta_M_vs_NAR_correlation': float(corr_r),
            'delta_M_vs_NAR_pvalue': float(corr_p),
            'hard_nar_vs_easy_nar_ratio': (
                hard_stats['nar_mean'] / (easy_stats['nar_mean'] + 1e-8)
            )
        }
        
        # Print summary
        print(f"\n=== Hard vs. Easy Task Analysis ===")
        print(f"Hard tasks NAR: {hard_stats['nar_mean']:.3f} ± {hard_stats['nar_std']:.3f}")
        print(f"Easy tasks NAR: {easy_stats['nar_mean']:.3f} ± {easy_stats['nar_std']:.3f}")
        print(f"NAR ratio (Hard/Easy): {results['hard_nar_vs_easy_nar_ratio']:.2f}x")
        print(f"Mann-Whitney p-value: {p_value:.4f}" if p_value else "N/A")
        print(f"δM vs NAR Spearman: r={corr_r:.3f}, p={corr_p:.4f}")
        
        return results


class NoisyTaskAnalyzer:
    """Phân tích khả năng phát hiện shots bị sai nhãn của FAMA."""
    
    def __init__(self, adapter: MAMLAdapter, config: HeTRoMConfig):
        self.adapter = adapter
        self.config = config
    
    def _create_noisy_task(self, task: Dict, noise_rate: float) -> Tuple[Dict, List[int]]:
        """
        Tạo task với một số shots bị sai nhãn.
        
        Returns:
            (noisy_task, noisy_indices) — indices của shots bị corrupt
        """
        support_x = task['support_x'].clone()
        support_y = task['support_y'].clone()
        
        K_N = len(support_x)
        n_classes = support_y.shape[-1] if support_y.dim() > 1 else len(support_y.unique())
        
        # Chọn indices để corrupt
        n_noisy = max(1, int(K_N * noise_rate))
        noisy_indices = torch.randperm(K_N)[:n_noisy].tolist()
        
        for idx in noisy_indices:
            if support_y.dim() > 1:
                # One-hot labels: chuyển sang nhãn sai
                original_class = support_y[idx].argmax().item()
                wrong_classes = [c for c in range(n_classes) if c != original_class]
                wrong_class = np.random.choice(wrong_classes)
                
                # Reset label
                support_y[idx] = torch.zeros(n_classes)
                support_y[idx, wrong_class] = 1.0
            else:
                # Integer labels
                original_class = support_y[idx].item()
                wrong_classes = [c for c in range(n_classes) if c != original_class]
                support_y[idx] = torch.tensor(np.random.choice(wrong_classes))
        
        noisy_task = {
            'support_x': support_x,
            'support_y': support_y,
            'query_x': task['query_x'],
            'query_y': task['query_y']
        }
        
        return noisy_task, noisy_indices
    
    def compute_shot_attributions(
        self,
        saliency: torch.Tensor  # [K*N, H, W] hoặc [K*N, C, H, W]
    ) -> torch.Tensor:
        """Tính shot-level attribution từ pixel-level saliency."""
        if saliency.dim() == 4:
            saliency = saliency.mean(dim=1)  # [K*N, H, W]
        return saliency.mean(dim=(-1, -2))   # [K*N]
    
    def run_single_noise_rate(
        self,
        theta0: Dict[str, torch.Tensor],
        tasks: List[Dict],
        fama_fn: callable,
        noise_rate: float,
        n_tasks: int
    ) -> Dict:
        """Chạy Noisy Task analysis cho một noise rate."""
        
        detection_successes = 0     # LNDR (strict: minimum attribution)
        rank_correlations = []       # SARC: correlation rank attribution vs correctness
        all_noisy_attrs = []
        all_clean_attrs = []
        
        for task in tasks[:n_tasks]:
            # Tạo noisy task
            noisy_task, noisy_indices = self._create_noisy_task(task, noise_rate)
            clean_indices = [i for i in range(len(task['support_x'])) if i not in noisy_indices]
            
            # Tính FAMA trên noisy task
            saliency = fama_fn(theta0, noisy_task)
            shot_attrs = self.compute_shot_attributions(saliency)  # [K*N]
            
            # LNDR: noisy shot có attribution thấp nhất không?
            # Attribution thấp nhất = âm nhất = harmful nhất
            min_attr_shot = shot_attrs.argmin().item()
            if min_attr_shot in noisy_indices:
                detection_successes += 1
            
            # SARC: correlation between attribution và correctness label
            K_N = len(shot_attrs)
            correctness = torch.ones(K_N)
            for idx in noisy_indices:
                correctness[idx] = 0.0
            
            corr, _ = scipy.stats.spearmanr(
                shot_attrs.cpu().numpy(),
                correctness.numpy()
            )
            rank_correlations.append(corr)
            
            # Attribution của noisy vs clean shots
            for idx in noisy_indices:
                all_noisy_attrs.append(shot_attrs[idx].item())
            for idx in clean_indices:
                all_clean_attrs.append(shot_attrs[idx].item())
        
        lndr = detection_successes / n_tasks
        
        # Statistical test: noisy shots có attribution thấp hơn clean shots?
        if all_noisy_attrs and all_clean_attrs:
            stat, p_value = scipy.stats.mannwhitneyu(
                all_noisy_attrs, all_clean_attrs,
                alternative='less'  # H1: noisy có attribution thấp hơn
            )
        else:
            p_value = None
        
        return {
            'noise_rate': noise_rate,
            'lndr': lndr,
            'chance_level': 1.0 / len(task['support_x']),
            'lndr_vs_chance': lndr / (1.0 / len(task['support_x'])),
            'sarc_mean': float(np.mean(rank_correlations)),
            'sarc_std': float(np.std(rank_correlations)),
            'noisy_attr_mean': float(np.mean(all_noisy_attrs)) if all_noisy_attrs else None,
            'clean_attr_mean': float(np.mean(all_clean_attrs)) if all_clean_attrs else None,
            'mannwhitney_pvalue': float(p_value) if p_value is not None else None,
            'significant': (p_value is not None and p_value < 0.05)
        }
    
    def run(
        self,
        theta0: Dict[str, torch.Tensor],
        tasks: List[Dict],
        fama_fn: callable
    ) -> Dict:
        """Chạy Noisy Task analysis trên tất cả noise rates."""
        all_results = {}
        
        for noise_rate in self.config.noise_rates:
            print(f"\nRunning Noisy Task Analysis (noise_rate={noise_rate:.0%})...")
            result = self.run_single_noise_rate(
                theta0, tasks, fama_fn, noise_rate,
                n_tasks=self.config.n_noisy_tasks
            )
            all_results[noise_rate] = result
            
            print(f"  LNDR: {result['lndr']:.3f} ({result['lndr_vs_chance']:.1f}× chance)")
            print(f"  SARC: {result['sarc_mean']:.3f} ± {result['sarc_std']:.3f}")
            print(f"  Mann-Whitney p: {result['mannwhitney_pvalue']:.4f}" 
                  if result['mannwhitney_pvalue'] else "  N/A")
        
        return all_results


class OODTaskAnalyzer:
    """Phân tích FAMA attribution khi support set bị OOD (domain shift)."""
    
    def __init__(self, adapter: MAMLAdapter, config: HeTRoMConfig):
        self.adapter = adapter
        self.config = config
    
    def _apply_ood_transform(self, images: torch.Tensor) -> torch.Tensor:
        """
        Áp dụng strong color jitter để tạo OOD support set.
        
        Thực hiện: 
        - Brightness multiplied by random factor in [0.2, 1.8]
        - Contrast extreme (0.2 to 1.8)
        - Saturation boost
        - Random grayscale
        """
        s = self.config.ood_color_jitter_strength
        
        transform = T.Compose([
            T.ColorJitter(
                brightness=s * 0.5,
                contrast=s * 0.5,
                saturation=s * 0.5,
                hue=min(0.5, s * 0.1)
            )
        ])
        
        # Áp dụng per-image (ColorJitter cần PIL hoặc [C,H,W] tensor)
        ood_images = []
        for img in images:
            # Clamp về [0,1] trước khi transform
            img_clamped = img.clamp(0, 1)
            ood_img = transform(img_clamped)
            ood_images.append(ood_img)
        
        return torch.stack(ood_images)
    
    def run(
        self,
        theta0: Dict[str, torch.Tensor],
        tasks: List[Dict],
        fama_fn: callable
    ) -> Dict:
        """Chạy OOD Task analysis."""
        
        id_magnitudes = []   # In-distribution attribution magnitudes
        ood_magnitudes = []  # OOD attribution magnitudes
        id_delta_Ms = []
        ood_delta_Ms = []
        
        n_tasks = min(len(tasks), self.config.n_ood_tasks)
        
        for task in tasks[:n_tasks]:
            # ID: original support
            saliency_id = fama_fn(theta0, task)
            am_id = saliency_id.abs().mean().item()
            dm_id = self.adapter.compute_adaptation_gain(
                theta0, task['support_x'], task['support_y'],
                task['query_x'], task['query_y']
            ).item()
            
            # OOD: perturbed support
            ood_support_x = self._apply_ood_transform(task['support_x'].cpu()).to(task['support_x'].device)
            ood_task = {
                'support_x': ood_support_x,
                'support_y': task['support_y'],
                'query_x': task['query_x'],
                'query_y': task['query_y']
            }
            saliency_ood = fama_fn(theta0, ood_task)
            am_ood = saliency_ood.abs().mean().item()
            dm_ood = self.adapter.compute_adaptation_gain(
                theta0, ood_support_x, task['support_y'],
                task['query_x'], task['query_y']
            ).item()
            
            id_magnitudes.append(am_id)
            ood_magnitudes.append(am_ood)
            id_delta_Ms.append(dm_id)
            ood_delta_Ms.append(dm_ood)
        
        # OAS: OOD Attribution Shift
        oas = np.mean(ood_magnitudes) / (np.mean(id_magnitudes) + 1e-8)
        
        # ODA: AUROC để phân biệt ID vs OOD dùng attribution magnitude
        labels = [0] * len(id_magnitudes) + [1] * len(ood_magnitudes)
        scores = id_magnitudes + ood_magnitudes
        try:
            oda = roc_auc_score(labels, scores)
        except Exception:
            oda = None
        
        # Statistical test
        stat, p_value = scipy.stats.mannwhitneyu(
            ood_magnitudes, id_magnitudes,
            alternative='greater'  # H1: OOD có magnitude cao hơn
        )
        
        results = {
            'id_magnitude_mean': float(np.mean(id_magnitudes)),
            'id_magnitude_std': float(np.std(id_magnitudes)),
            'ood_magnitude_mean': float(np.mean(ood_magnitudes)),
            'ood_magnitude_std': float(np.std(ood_magnitudes)),
            'oas': float(oas),  # > 1 → OOD attribution cao hơn
            'oda_auroc': float(oda) if oda else None,
            'id_delta_M_mean': float(np.mean(id_delta_Ms)),
            'ood_delta_M_mean': float(np.mean(ood_delta_Ms)),
            'mannwhitney_pvalue': float(p_value),
            'significant': p_value < 0.05,
            'n_tasks': n_tasks
        }
        
        print(f"\n=== OOD Task Analysis ===")
        print(f"ID Attribution Magnitude: {results['id_magnitude_mean']:.4f} ± {results['id_magnitude_std']:.4f}")
        print(f"OOD Attribution Magnitude: {results['ood_magnitude_mean']:.4f} ± {results['ood_magnitude_std']:.4f}")
        print(f"OAS (OOD/ID ratio): {results['oas']:.2f}x")
        if oda:
            print(f"ODA (AUROC): {results['oda_auroc']:.3f}")
        print(f"Statistical significance: p={results['mannwhitney_pvalue']:.4f}")
        
        return results


class HeTRoMRunner:
    """Chạy toàn bộ HeTRoM Diagnostic Framework."""
    
    def __init__(
        self,
        adapter: MAMLAdapter,
        config: HeTRoMConfig
    ):
        self.hard_easy_analyzer = HardEasyTaskAnalyzer(adapter, config)
        self.noisy_analyzer = NoisyTaskAnalyzer(adapter, config)
        self.ood_analyzer = OODTaskAnalyzer(adapter, config)
        self.config = config
    
    def run_all(
        self,
        theta0: Dict[str, torch.Tensor],
        tasks: List[Dict],
        fama_fn: callable
    ) -> Dict:
        """Chạy toàn bộ HeTRoM framework."""
        
        print("\n" + "=" * 70)
        print("HeTRoM DIAGNOSTIC POWER EVALUATION")
        print("=" * 70)
        
        results = {}
        
        # Scenario 1: Hard/Easy Task
        print("\n[1/3] Hard vs. Easy Task Analysis...")
        results['hard_easy'] = self.hard_easy_analyzer.run(
            theta0, tasks, fama_fn
        )
        
        # Scenario 2: Noisy Task
        print("\n[2/3] Noisy Task Analysis...")
        results['noisy'] = self.noisy_analyzer.run(
            theta0, tasks, fama_fn
        )
        
        # Scenario 3: OOD Task
        print("\n[3/3] OOD Task Analysis...")
        results['ood'] = self.ood_analyzer.run(
            theta0, tasks, fama_fn
        )
        
        # Overall Diagnostic Power Score (DPS)
        dps_components = []
        
        he = results['hard_easy']
        if he.get('nar_significant'):
            dps_components.append(he['hard_nar_vs_easy_nar_ratio'])
        
        noisy = results['noisy']
        for rate in self.config.noise_rates:
            if rate in noisy and noisy[rate]['significant']:
                dps_components.append(noisy[rate]['lndr_vs_chance'])
        
        ood = results['ood']
        if ood['significant']:
            dps_components.append(ood['oas'])
        
        results['diagnostic_power_score'] = (
            float(np.mean(dps_components)) if dps_components else 0.0
        )
        results['n_significant_components'] = len(dps_components)
        
        print(f"\n{'=' * 70}")
        print(f"DIAGNOSTIC POWER SCORE: {results['diagnostic_power_score']:.2f}x")
        print(f"(Significant components: {results['n_significant_components']}/3)")
        print(f"{'=' * 70}")
        
        return results
```

---

### 4.6. Diễn Giải Kết Quả và Viết Cho Paper

**Bảng tổng hợp HeTRoM (đề xuất cho paper)**:

| Kịch bản | Metric | FAMA | Baseline Rand | Significance |
|----------|--------|------|---------------|-------------|
| Hard Task | NAR (Hard/Easy ratio) | X.Xx | 1.0x | p < 0.05 |
| Noisy Task (20%) | LNDR | X.XX (Xx chance) | 0.04 | p < 0.01 |
| Noisy Task (40%) | LNDR | X.XX | 0.04 | p < 0.01 |
| OOD Task | OAS | X.Xx | ~1.0x | p < 0.05 |
| OOD Task | ODA AUROC | X.XX | 0.50 | — |

**Câu narrative cho paper:**

> *"Under the HeTRoM diagnostic framework, FAMA demonstrates strong diagnostic capability across all three heterogeneous task scenarios. For hard tasks (where adaptation fails, δM > −0.05), FAMA's Negative Attribution Ratio is X.X× higher than for easy tasks (p < 0.05, Mann-Whitney), indicating that the model correctly identifies conflicting support set features when adaptation struggles. In noisy label scenarios, FAMA achieves a Label Noise Detection Rate of X.XX at 20% noise — X.X× above chance level — suggesting that mislabeled shots tend to receive negative attributions, consistent with their gradient direction opposing the adaptation objective. Under distribution shift (strong color jitter), attribution magnitude increases by X.X× (OAS = X.X), and FAMA achieves an OOD detection AUROC of X.XX, suggesting that attribution magnitude serves as a proxy for adaptation uncertainty."*
