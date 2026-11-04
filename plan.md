# Kế hoạch sửa code & thí nghiệm `src/` theo `paper.pdf` (FAMA)

> **Trạng thái: CHỈ LÀ KẾ HOẠCH.** Chưa sửa dòng code nào trong `src/`, chưa chạy thí nghiệm nào.
> Mọi lệnh trong tài liệu này dự kiến chạy trên **máy GPU từ xa** (qua `push.sh`), không chạy trên máy local
> (local chỉ có `torch 2.11.0+cpu`).
>
> Nguồn: đọc toàn bộ `paper.pdf` (18 trang, gồm Appendix A–D) và đọc code trong `src/`
> (`interpreters/fama.py`, `algos/`, `models/`, `loaders/`, `scripts/`, `scripts/check_explain/`, `configs/`, `run.sh`, `main.py`).

---

## 0. Tóm tắt

1. **Explainer hiện tại chưa phải phương pháp chính của paper.** `FAMAExplainer._saliency_core_batched`
   (`interpreters/fama.py:152-192`) dùng *feature bị detach* + *Grad-CAM (GAP trọng số kênh)*. Đó đúng là hai dòng
   **ablation** của Table 3 ("detached A_k", "Grad-CAM channel pooling"). Nhánh feature-reuse còn chạy adjoint trên
   **toàn bộ tham số** thay vì chỉ trên head (§3.2 cuối, §3.5). Cần viết lại lõi theo Prop 3.4 / Algorithm 1 (mục WS1).
2. **Con số "negative transfer is common" (49.7% và 12.5%) tái lập được từ log cũ, nhưng chưa so sánh được với nhau.**
   `src/.logs/explain3/tiered2cub_ex` có **5 ảnh query/task** và config dùng `sm_loss`. `mini2cub_ex` có **120 ảnh query/task**
   và dùng `ce_loss`. Với 5 ảnh query, ΔM rất nhiễu, tỉ lệ 49.7% gần mức tung đồng xu. Phải chạy lại trước khi đưa vào paper (mục 3, R1–R2).
3. **Phần lớn protocol đánh giá của paper chưa có code:** counterfactual re-adaptation (§4.2), planted-feature (§4.3),
   9 baseline (§5.1/C.2), edit-mitigation (§5.4), Waterbirds, CLIP+LoRA/ISIC (§5.5), multi-seed, thống kê (Wilcoxon + Holm + bootstrap CI).
4. **Có 5 chỗ paper và repo mâu thuẫn** (ResNet-12 vs ResNet-10, loss, `second_order`, BatchNorm vs "linear in the loss",
   Grad-CAM@φ_T chưa định nghĩa). Mục 4 đề xuất mặc định cho từng quyết định.
5. **Đường găng là training** (12+ lần train, cần `--seed`, `ce_loss`, `test_k_query` hợp lý). Nên chốt WS0 rồi đẩy lên máy GPU
   trước, trong lúc viết explainer.

### Những gì đã / chưa kiểm chứng

| Nhận định | Cách biết | Mức chắc chắn |
|---|---|---|
| Code dùng detach + Grad-CAM; freeze-branch dùng full λ | Đọc code, đối chiếu Remark A.4 và Table 3 | Cao (đọc trực tiếp) |
| Hậu quả định lượng (sai số so với sai phân hữu hạn, maps cũ khác maps mới bao nhiêu) | **Chưa đo.** Đã viết prototype nhưng bị dừng; xem mục 1.1 | Chưa biết |
| 49.7% / 12.5% có trong log cũ | Đã đọc `adaptation_gain.csv` (600 task mỗi file) | Cao |
| 5 vs 120 ảnh query | Đã đếm trong `Q_paths.csv` | Cao |
| `second_order: false` nhưng thực tế là bậc 2 | Suy từ code (`maml.py` chỉ lưu cờ, `_deploy` dùng `torch.func.grad_and_value`) | Trung bình, cần xác minh (R4) |

### 1.1 Prototype chưa chạy
`/tmp/claude-1000/-home-hariknguyen-Workdir-xai-meta/f4dba67a-92d5-43a4-a60e-c314891203fa/scratchpad/verify_fama.py`
(ngoài repo) so Φ chính xác, các biến thể sai và code hiện tại với sai phân hữu hạn trên Conv4 float64.
**Chưa chạy xong lần nào** (lần đầu lỗi do tôi bọc rollout trong `no_grad`, đã sửa nhưng chưa chạy lại). Có thể dùng làm
hạt giống cho `tests/test_fama_exact.py` (WS2). Chỉ chạy trên máy GPU khi bạn đồng ý.

---

## 2. Đối chiếu paper ↔ code hiện tại

Trạng thái: ✅ khớp · ⚠️ có nhưng lệch · ❌ chưa có.

| ID | Paper | Code hiện tại | TT |
|---|---|---|---|
| G1 | ΔM := R(φᶠ_T) − R(φ_T) (Eq.2); báo cả ΔM% := 100·ΔM/R(φ_T), CI percentile (C.1) | `fama.py:131-149` chỉ trả ΔM% (tương đối), không CI, không accuracy từng nhánh. Dấu của ΔM% = dấu của ΔM nên con số "% task âm" vẫn đúng | ⚠️ |
| G2 | Φ(x_k) = α Σ_t (G_{t,k} − Gᶠ_{t,k}), G = ∂/∂m_k[λᵀ∇_φ L(φ_{t−1},S;m)] tại m=1 (Prop 3.4) | `fama.py:152-192`: Grad-CAM (`:184-185`), feature detach (`:171-172`) | ⚠️ (là 2 dòng ablation) |
| G3 | Nhánh feature-reuse: λᶠ chỉ trên head, HVP lấy block head của Hessian (§3.2, §3.5, Lemma A.2) | `fama.py:275-283`: λ seed và `_hvp` trên toàn bộ tham số, body block của λᶠ ≠ 0 và rò vào head qua H^{hb} | ⚠️ |
| G4 | Mỗi bước 1 double-backward cho cả (v, G) (Alg.1 dòng 7-9) | 2 lượt double-backward/bước/nhánh (`_hvp` + `_saliency_core_batched`) | ⚠️ (chậm ~2×) |
| G5 | λ_T = ∇R(φ_T) trên toàn pool query; bootstrap chỉ để ước lượng bất định (C.1) | `fama.py:113-129`: trung bình gradient của 100 resample ×2 nhánh | ⚠️ (xem R5) |
| G6 | Φ tích phân Φ^int, n_β khoảng, completeness ΣΦ^int = ΔM (Cor 3.5) | Không có | ❌ |
| G7 | Kiểm chứng số: sai phân hữu hạn theo hướng u, completeness (C.3) | Không có | ❌ |
| G8 | Hiển thị: một thang màu đối xứng **cho cả task** (§3.5) | `scripts/explain.py:127-128` tính `max_abs` **từng ảnh** | ⚠️ |
| G9 | BI-ADT: ABC↓/ABC↑ trên ΔM, random order, paired test (§4.1) | `bi_adt.py`: PDAS/NDAS ≡ ABC↓/ABC↑ (`:220-222`), nhưng dùng ΔM%; random 1 hoán vị/task (`:54`); chỉ lấy mean | ⚠️ |
| G10 | Counterfactual re-adaptation ảnh/vùng, Spearman (§4.2) | Không có | ❌ |
| G11 | Planted-feature, AUROC + sign acc (§4.3) | Không có | ❌ |
| G12 | Sanity: cascading ✅, independent layer ❌, data randomization ❌, support edits ✅ + ΔM đi kèm ❌ (§4.4, C.6) | `sanity_params.py` (chỉ cascading), `sanity_support_set.py` (chỉ tương quan, không lưu ΔM) | ⚠️ |
| G13 | 9 baseline: Random, Center, Grad-CAM@φ_T, IG pixel, TracIn-CAM, IF-CAM, TracIn, Influence, SGD-influence (§5.1, C.2) | Không có | ❌ |
| G14 | Ablation Table 3 (5 biến thể) | Không có (nhưng đều là cờ của lõi mới) | ❌ |
| G15 | Backbone: Conv4 ✅; ResNet-12 ❌ (repo có ResNet-10); CLIP ViT-B/16 + LoRA ❌ | `models/` | ⚠️/❌ |
| G16 | Dữ liệu: mini/tiered/CUB ✅; Waterbirds ❌; ISIC (BSCD-FSL) ❌; bbox CUB ❌ (§5.4) | `loaders/`, `prepare_*.py` | ❌ |
| G17 | 600 task × **3 seed meta-training**; Wilcoxon + Holm; bootstrap CI 95% (§5.1) | Không có `--seed`; không có thống kê. Test loader đã cố định `seed=42` (`warm_up.py:68`) ✅ | ❌ |
| G18 | Quét T ∈ {1,5,10,20}, α ∈ {0.005,0.01,0.02} lúc test (§5.3) | `T_test`, `base_lr` chỉ đọc từ YAML | ❌ |
| G19 | Phân tích contextual bias + edit mitigation (§5.4, Table 2 trái) | Không có | ❌ |
| G20 | Siêu tham số train: α=0.01, T=5 (train)/10 (test), Adam 1e-3, meta-batch 4, 60k iter, 5-way 1-shot | `configs/*/conv4.yaml` | ✅ |

---

## 3. Vấn đề phải xử lý trước (blocker / rủi ro khoa học)

**R1. Loss không đồng nhất, lệch paper.** Paper: "L và R là cross-entropy" (§2.1, §3.1).
`tiered2cub`, `cub2tiered`, `tiered2strokes_omnig` (cả conv4 và resnet10) dùng `sm_loss` (`SmoothMarginLoss`).
`tiered2cub/explain_test_conv4.yaml` lại dùng `ce_loss` → checkpoint train bằng margin loss nhưng check_explain chạy bằng CE.
→ Mọi config cho paper dùng `ce_loss`; **`tiered2cub` (setting chính của paper) phải train lại.**

**R2. `test_k_query: 1` ở hầu hết config.** Với 1 ảnh/lớp, R chỉ có 5 ảnh, bootstrap vô nghĩa, dấu ΔM gần như ngẫu nhiên.
Log cũ: `tiered2cub_ex` 5 ảnh query/task → 49.7% âm; `mini2cub_ex` 120 ảnh → 12.5% âm.
Log `explains/` còn có 50.3% (mini2cub-resnet10) và 67.3% (cub2mini-resnet10); config tương ứng cũng `test_k_query: 1`,
nên **chưa đáng tin** cho tới khi chạy lại (chưa kiểm tra số ảnh query của chúng).
→ Đặt `test_k_query ≥ 24` (như `explain_test_*`) cho mọi thí nghiệm giải thích; báo cáo tỉ lệ ΔM<0 kèm CI.

**R3. `grad_clip: 10` trong mọi config.** `_fast_weights` clamp từng phần tử gradient (`maml.py:52-55`), còn adjoint giả định
φ_t = φ_{t−1} − α∇L (Eq.1). Khi clamp kích hoạt, Jacobian bị sai tại phần tử đó.
→ (a) đếm tần suất clamp kích hoạt khi explain; (b) test C.3 chạy với `grad_clip=None`;
(c) nếu clamp hay kích hoạt thì tắt clamp lúc explain (nếu accuracy không đổi) hoặc ghi rõ vào paper.

**R4. Cờ `second_order` không được dùng.** `maml.py` chỉ lưu `self.second_order`; `_deploy` dùng `torch.func.grad_and_value`
và `meta_loss.backward()` đi xuyên qua `grads` nên khả năng cao là **MAML bậc 2** dù YAML ghi `false`.
→ Xác minh trên GPU bằng một test nhỏ (so meta-gradient với bản detach `grads`); sửa YAML/cờ hoặc câu "original protocol" trong paper cho khớp.

**R5. BatchNorm batch-statistics làm R không tuyến tính theo từng mẫu.** `track_running_stats=False` (`conv_block.py`, `resnet10.py`)
→ loss của một batch query phụ thuộc thành phần batch. Câu của C.1 ("gradient tuyến tính theo loss nên resample không làm tăng độ chính xác")
không đúng tuyệt đối, và "trung bình bootstrap" ≠ "toàn pool". Để Φ = ∂ΔM/∂m **đúng với đúng hàm R đã dùng**, cần một định nghĩa duy nhất (xem D3).

**R6. Không tái lập / không "paired".** Bootstrap dùng RNG toàn cục (`loaders/utils.py:42-56`), thứ tự xóa ngẫu nhiên dùng `np.random`
(`bi_adt.py:54`), `permute_label` dùng `np.random`, và các task chạy song song bằng thread dùng chung RNG (`scripts/utils.py::parallel_map`).
Ngoài ra `FewShotDataset.__get_filenames` dùng `os.listdir` **không sắp xếp** → chỉ số ảnh trong lớp (và do đó task sinh từ `seed=42`)
có thể khác giữa các máy.
→ `sorted(os.listdir(...))`, generator riêng theo `(seed, task_id)`, lưu danh sách task ra file JSON để mọi explainer/seed dùng đúng cùng task.

**R7. TF32 bật toàn cục** (`main.py:142-145`) làm HVP/adjoint kém chính xác (Prop B.1 cảnh báo sai số lan truyền).
→ Tắt TF32 ở mode `explain`/`check_explain`; test chính xác chạy float64.

**R8. `tests/test_loaders.py` đã cũ:** import `MiniImagenetDataset`, nhưng `loaders/datasets.py` chỉ có `FewShotDataset`. Sửa hoặc bỏ.

---

## 4. Quyết định cần bạn chốt (kèm mặc định đề xuất)

| ID | Câu hỏi | Đề xuất | Lý do |
|---|---|---|---|
| D1 | Paper ghi ResNet-12, repo dùng ResNet-10 (`edits/replace_res12_with_resnet10.md`: Res12 OOM 16GB, ~28h/60k iter) | **Sửa paper thành ResNet-10 (Chen et al. 2019)** | Đã có quyết định cũ và checkpoint; mục tiêu "robust theo độ sâu" vẫn đạt. Giữ Res12 nghĩa là 6+ lần train ×28h |
| D2 | "Waterbirds task" 5-way thế nào (Waterbirds gốc chỉ có 2 lớp, head hiện là 5 đầu ra) | **Tạo task kiểu Waterbirds 5-way:** vật thể CUB (mask segmentation) dán lên nền thuộc 5 nhóm cảnh; support nền tương quan với nhãn mức c, query nền ngẫu nhiên. c ∈ {0, .25, .5, .75, 1} | Giữ nguyên checkpoint 5-way. Cần ảnh nền (Places hoặc tương đương) và `segmentations` của CUB |
| D3 | Định nghĩa R khi BN dùng batch stats | **`risk=pool` (mặc định):** R = loss trên cả pool query như một batch cho ΔM điểm, λ_T và test C.3. Bootstrap chỉ để CI của ΔM và độ bất định của Φ. Báo thêm ΔM bootstrap-mean. Sửa câu ở C.1 | Φ khớp chính xác đạo hàm của đúng R; tốn 1 backward thay vì 200 |
| D4 | Head của CLIP | **Linear trên trung bình các patch token** của block cuối (không dùng CLS) | Appendix B.3: chỉ khi đó ΔM(0)=0 và Cor 3.5 đúng |
| D5 | Loss cho mọi thí nghiệm paper | **`ce_loss`**, train lại `tiered2cub` | R1 |
| D6 | "Grad-CAM at φ_T" chưa được định nghĩa trong paper | Grad-CAM của logit lớp đúng trên từng ảnh support tại φ_T, **có dấu** (không ReLU) cho Table 1; thêm bản ReLU chuẩn ở phụ lục. Bổ sung 1 câu vào C.2 | Cần dấu để chấm ABC↑ và sign acc |
| D7 | Spearman ρ_img/ρ_reg tính theo task hay gộp? | ρ_reg: theo **từng task** rồi lấy trung bình (nhiều vùng/task). ρ_img: **gộp qua task** (1-shot chỉ có 5 ảnh/task) và/hoặc dùng 5-shot. Nêu rõ trong §4.2 | 5 điểm/task cho Spearman quá nhiễu |
| D8 | Planted-feature "tỉ lệ ảnh bị ảnh hưởng" khi 1-shot | Dùng **5-shot** cho benchmark này, hoặc gieo patch ở nhiều lớp (1-shot) | 1-shot mỗi lớp chỉ có 1 ảnh support |
| D9 | "1,200 task" ở §5.4 lấy từ đâu | `metatest_iterations: 1200` (sampler tuần tự theo seed 42, nên 600 task đầu trùng bộ 600) | Cố định task chung |
| D10 | Thống kê qua 3 seed | Trung bình theo task qua 3 seed rồi Wilcoxon trên 600 task (bảo thủ); báo CI bootstrap theo task | Tránh coi 1800 quan sát là độc lập |

---

## 5. Kế hoạch sửa code

Quy ước: **[M]** sửa file có sẵn, **[N]** file mới. Thứ tự phụ thuộc xem mục 7.

### WS0 — Hạ tầng, tái lập, vệ sinh config (làm đầu tiên)

| File | Thay đổi |
|---|---|
| [M] `main.py` | Thêm `--seed`, `--T_test`, `--alpha`, `--n_boot`, `--risk {pool,bootstrap}`, `--methods`, `--n_beta`, `--n_rand`, `--results_dir`, `--float64`, `--permute_train_labels`; mở rộng `--check_method` thêm `cf_image, cf_region, planted, edit, bg_share`. Tắt TF32 khi mode là `explain`/`check_explain` (R7) |
| [M] `scripts/__init__.py`, `scripts/warm_up.py` | Truyền seed vào: `torch.manual_seed` trước khi khởi tạo `MAML` (θ₀), `BatchTaskSampler(seed=seed)` cho train/val; giữ `seed=42` cho test (cố định task) |
| [M] `loaders/datasets.py` | `sorted(os.listdir(...))` (R6) |
| [M] `loaders/utils.py` | `get_stratified_bootstrap_batches(..., generator=)` nhận `torch.Generator`; thêm `split_query_halves(que_x, que_y, gen)` cho edit-mitigation |
| [N] `scripts/rng.py` | `task_generator(seed, task_id, tag)` dùng chung cho bootstrap, thứ tự xóa ngẫu nhiên, patch ngẫu nhiên, label permutation |
| [N] `tasks/` | Xuất/đọc danh sách task cố định (`<scenario>_seed42.json` chứa đường dẫn support/query) |
| [M] `configs/*` | `criterion: ce_loss` cho mọi config dùng trong paper; `test_k_query: 24`; thêm `configs/paper/` nếu không muốn đụng config cũ |
| [M] `run.sh` → [N] `run_paper.sh` | Tên checkpoint `<scenario>_<backbone>_s<seed>_ckpt`; vòng lặp seed |
| [M] `tests/test_loaders.py` | Sửa import theo `FewShotDataset` hoặc bỏ (R8) |

**Chấp nhận:** cùng `--seed` cho cùng checkpoint ban đầu; chạy hai lần `explain` cho cùng ΔM từng task (sai khác 0 khi không dùng thread, hoặc ≤ sai số số học khi dùng thread).

### WS1 — Viết lại lõi FAMA (`interpreters/fama.py`) — quan trọng nhất

**API mới** (giữ `interpret()` và `compute_gain_only()` làm wrapper để `check_explain` cũ chạy được):
`explain(task, cfg) -> FamaResult` gồm `delta_m` (tuyệt đối), `delta_m_pct`, `delta_m_ci`, `acc_full`, `acc_freeze`,
`phi_grid` [N,1,H′,W′] (độ phân giải gốc của feature map), `phi_img` [N,1,H,W] (upsample bilinear, giữ dấu), `phi_full`/`phi_freeze` riêng từng nhánh (cho ablation).

**Thuật toán một nhánh** (với α tổng quát từng tham số, Appendix B.2: đặt u = α⊙λ):

```python
def _adjoint_pass(ctx, branch, beta=1.0):             # branch ∈ {"full","freeze"}, σ = +1 / −1
    phis  = ctx.rollout(branch, beta)                 # T+1 trạng thái; support loss có mask m=β·1
    adapt = ctx.adapt_idx[branch]                     # full: tất cả tham số | freeze: chỉ head
    lam   = grad_R(phis[T])[adapt]                    # seed theo D3; freeze: body block = 0
    Phi   = zeros(N, 1, H_, W_)
    for t in range(T, 0, -1):
        w = leaf(phis[t-1], adapt)                    # freeze: body = θ0^b hằng số
        m = full(β).requires_grad_()
        A = ctx.features(x, w)                        # full: trong graph | freeze: tính 1 lần, hằng số
        g = autograd.grad(L(head(m * A, w), y), w_adapt, create_graph=True)   # KHÔNG detach A
        s = sum(((alpha * lam) * g).sum())            # s = (α⊙λ)ᵀ ∇_φ L
        v, G = autograd.grad(s, (w_adapt, m))         # MỘT double-backward ra cả (v, G)
        Phi += σ * G                                  # G đã chứa α
        lam = lam - v                                 # λ_{t-1} = λ_t − H(α⊙λ_t)
    return Phi
```

Điểm khác code hiện tại: (i) `A` nằm trong graph nên block backbone của ∇_φL và λ được giữ (Remark A.4);
(ii) đạo hàm theo **mask** `m` thay cho Grad-CAM; (iii) freeze-branch chỉ trên head; (iv) một double-backward/bước.

**Tối ưu đáng làm:**
- Freeze-branch: body cố định ⇒ feature map `A` không đổi qua các bước ⇒ tính 1 lần, mỗi bước chỉ forward/backward head (rẻ hơn nhiều).
- Tính trajectory của cả hai nhánh **một lần/task** trong `TaskContext` và dùng chung cho mọi baseline (paper: "All methods reuse the same trajectories").
- ΔM bootstrap: gom `n_boot` resample vào một lượt (`torch.func.vmap` hoặc batch lớn) thay vì 100 lượt tuần tự.

**Biến thể (cờ) cho ablation Table 3 và baseline:**
`contrast` (full − freeze | chỉ full), `signed` (có/không ReLU), `steps` (`all` | `last`), `detach_A`, `pooling` (`mask` | `gradcam`),
`adjoint` (`exact` | `myopic` → TracIn-CAM | `implicit` → IF-CAM), `n_beta` (0 = Φ thường; >0 = Φ^int, quy tắc hình thang, rollout lại ở mỗi β).

**Φ^int (Cor 3.5):** với β ∈ linspace(0,1,n_β+1): rollout lại dưới mask β·1, gọi `_adjoint_pass(beta=β)`, tích phân hình thang.
Chi phí (n_β+1)×. Kiểm tra ΔM(0)=0 và ΣΦ^int = ΔM.

**Khác:** `ignore_grad_clip` (R3); `dtype=float64` cho test; `alpha` nhận float hoặc list theo tham số (cần cho LoRA/Meta-SGD).
Hợp đồng model: `forward(x, weights, only_features=True)` trả [N,C,H′,W′] trước pooling/flatten; thêm
`head_param_indices()` thay cho giả định "module cuối là head" (`_get_head_mask`, `fama.py:53-64`); đã đúng với Conv4 và ResNet10.

**Chấp nhận:** toàn bộ WS2 pass; `phi_grid` trùng sai phân hữu hạn; `explain` nhanh hơn hoặc bằng bản cũ trên cùng task.

### WS2 — Kiểm chứng số (`tests/test_fama_exact.py`, dữ liệu tổng hợp, không cần dataset)

Theo Appendix C.3, float64, `grad_clip=None`:

| Test | Nội dung | Ngưỡng đề xuất |
|---|---|---|
| T1 exactness | ⟨Φ, u⟩ vs [ΔM(1+εu) − ΔM(1−εu)]/2ε, u ngẫu nhiên ‖u‖=1, ε ∈ {1e-2,1e-3,1e-4}, Conv4 và ResNet10 | sai số tương đối ≤ 1e-5 (chặt hơn khi ε nhỏ, tới sai số chia sai phân) |
| T2 biến thể sai | Cùng test với `detach_A`, `pooling=gradcam`, freeze-full-λ | **Phải thất bại có chủ đích**, ghi kết quả vào C.3 |
| T3 completeness | ΣΦ^int vs ΔM, n_β ∈ {4,8,16,32}; ΔM(0)=0 | residual giảm theo n_β |
| T4 gộp (v,G) | Kết quả một double-backward = hai lượt tách | trùng đến sai số máy |
| T5 head-only | λᶠ body block = 0 trong mọi bước | chính xác |
| T6 Prop 3.2 | T=1: ΔM ≈ Tα⟨g_b, ∇_{φb}L(θ₀,S)⟩, phần dư O(α³) | độ dốc log-log ≈ 3 |
| T7 Prop 3.7 | |I[λ] − I[Λᵐʸᵒ]| giới hạn ∝ α² khi T cố định | độ dốc ≈ 2 |
| T8 R5 | ‖mean-bootstrap-grad − pool-grad‖ và chênh ΔM bootstrap vs pool | chỉ đo, quyết định D3 |

**Thí nghiệm E0.3 (quyết định tái sinh kết quả cũ):** trên checkpoint cũ, Spearman giữa map cũ (Grad-CAM) và map mới (mask) theo từng task.
Dù kết quả thế nào, bảng của paper vẫn phải tính lại bằng explainer mới; E0.3 chỉ cho biết hình trong luận văn có phải thay hay không.

### WS3 — Protocol đánh giá (`scripts/check_explain/`)

Nguyên tắc: **mọi explainer chạy trong cùng một vòng lặp trên cùng task**; phần ground-truth không phụ thuộc explainer
(đường cong random, ΔM_{−k}) tính **một lần/task** rồi dùng chung.

- **[M] `bi_adt.py` (§4.1).** (a) tính trên ΔM tuyệt đối (giữ ΔM% làm cột phụ); (b) random order trung bình `n_rand` lần (mặc định 5, generator theo task);
  (c) xuất ABC↓, ABC↑ từng task + cờ `both_positive`; đổi tên cột PDAS/NDAS → ABC↓/ABC↑ (giữ alias); (d) nhận registry explainer;
  (e) dùng cùng resample bootstrap cho mọi mức ρ. Chi phí/task/explainer: 2 thứ tự × 10 mức ρ lần `compute_gain_only` (đường random dùng chung).
- **[N] `counterfactual.py` (§4.2).** Cấp ảnh: thay x_k bằng bản blur, `effect_k = ΔM − ΔM_{−k}`, Spearman với `Σ_ω Φ(x_k)_ω` (D7).
  Cấp vùng: trên tập con task (đề xuất 100), blur từng superpixel (đề xuất n_sp=50, hoặc lấy mẫu ≤100 vùng/task để chi phí ≈ 100 × 2T bước/vùng),
  attribution của vùng = **tổng** Φ trên vùng ở độ phân giải gốc (dấu đúng với xấp xỉ bậc nhất), Spearman theo task.
- **[N] `planted.py` + [N] `loaders/planted.py` (§4.3).** Sinh patch (kích thước, tương phản, vị trí **ngoài tâm**, tỉ lệ ảnh bị ảnh hưởng) cho hai loại:
  *helpful* (patch cả ở support và query của lớp c\*) và *harmful* (chỉ ở support, hoặc ở query của lớp khác). Chỉ giữ task mà ΔM dịch đúng hướng thiết kế
  (kiểm định dấu ΔM_planted − ΔM_clean, CI bootstrap). Chấm AUROC của `σ·Φ` (σ=+1 helpful, −1 harmful) so với mask patch, và sign acc = tỉ lệ patch có dấu ΣΦ đúng thiết kế.
- **[M] `sanity_params.py` (§4.4).** Thêm chế độ `independent` (randomize từng block rồi khôi phục) bên cạnh `cascading` hiện có;
  thêm `data` (so map giữa checkpoint thường và checkpoint train với nhãn hoán vị, cần `--permute_train_labels`); báo cả Pearson/Spearman trên giá trị thô và trị tuyệt đối.
- **[M] `sanity_support_set.py`.** Lưu thêm ΔM trước/sau để báo "thay đổi map kèm thay đổi ΔM" (C.6).
- **[N] `edit_mitigation.py` (§5.4, Table 2 trái).** Chia pool query thành Q_a / Q_b (phân tầng). Explainer chỉ thấy Q_a (tính ΔM, λ_T, Φ). Chọn task có ΔM<0 trên Q_a.
  Blur các superpixel âm nhất tới ngân sách ρ_edit (cố định, đề xuất 0.2; nhạy cảm {0.1,0.2,0.3}), adapt lại, đo accuracy trên Q_b.
  `Recovered% = 100·(acc_edit − acc_{φ_T}) / (acc_{φᶠ_T} − acc_{φ_T})`, gộp theo **tỉ số của tổng** (không lấy trung bình tỉ số vì mẫu số có thể ≈ 0).
  Baseline: Random cùng diện tích, Grad-CAM@φ_T, TracIn-CAM, SGD-influence (blur cả ảnh), FAMA.
- **[N] `analysis/bg_share.py` (§5.4).** Tỉ lệ khối lượng attribution **có hại** (`max(−Φ,0)`) nằm ngoài bbox CUB, tương quan với ΔM qua 1,200 task. Cần WS6.

### WS4 — Baseline (`interpreters/baselines/`, [N])

Registry thống nhất `{tên: hàm → Attribution(grid|None, img|None, image_scores)}` để protocol lặp qua mọi phương pháp. Tất cả theo khung Eq.9 `I_ψ[Λ]`, tính trên **cả hai nhánh rồi lấy feature-reuse − full** (Eq.7, C.2):

| Baseline | Cách làm |
|---|---|
| Random, Center prior | Bản đồ Gaussian ngẫu nhiên; Gaussian dương ở tâm (σ ≈ 0.25·H). Ghi định nghĩa vào C.1 |
| Grad-CAM@φ_T | Theo D6 |
| TracIn-CAM | Thay λ_t bằng ∇R(φ_{t−1}) (myopic), không lan truyền; mỗi bước 1 backward query + 1 đạo hàm theo m |
| IF-CAM (Eq.15) | (H_T + τI)⁻¹ ∇_φL(φ_T,S;m) bằng CG qua HVP; τ chọn trên tập validation (không phải test), vd {0.01, 0.1, 1}; nhánh freeze chỉ block head |
| ∂ΔM/∂x (IG) | Cùng khung nhưng đạo hàm theo **pixel** thay mask, tích phân từ baseline blur, n_ig=16 (rollout lại mỗi điểm) |
| TracIn | Σ_t α ∇R(φ_{t−1})ᵀ ∇_φℓ(φ_{t−1};x_k,y_k)/\|S\| |
| SGD-influence | Như trên nhưng với λ_t chính xác |
| Influence function | −∇R(φ_T)ᵀ(H+τI)⁻¹∇_φℓ_k tại φ_T |

Ghi chú: ℓ_k lấy từ loss từng mẫu (`reduction='none'`) của **cùng batch** để đúng với BatchNorm (đạo hàm theo trọng số mẫu w_k chỉ nhân vào loss từng mẫu).
Log thời gian (đồng bộ CUDA) từng phương pháp cho cột "Time (s)" của Table 1.

### WS5 — Mô hình

- **ResNet-12:** theo D1 mặc định *không làm*, chỉ sửa paper. Nếu bạn chọn giữ: thêm `models/resnet12.py` (kênh 64-160-320-640), dùng `grad_accum_tasks` đã có, ước lượng ~28h/60k iter/seed theo ghi chú cũ.
- **CLIP ViT-B/16 + LoRA ([N] `models/clip_lora.py`, §5.5, B.3):** tham số φ = [LoRA A,B của các lớp được chọn] + head tuyến tính (D4).
  Forward hàm (explicit params hoặc `torch.func.functional_call`) hỗ trợ double-backward. `only_features` trả lưới patch token [N,D,14,14]; `forward_features` = Linear trên trung bình token đã nhân mask.
  Nhánh feature-reuse = chỉ head trên đặc trưng đóng băng. θ₀: LoRA khởi tạo chuẩn (B=0), head khởi tạo 0 (mặc định, cần chốt).
  Tốc độ học khác nhau giữa LoRA và head ⇒ cần `alpha` theo tham số ở WS1. Giá trị [to fill] đề xuất: rank r=4 trên q,v của 4 block cuối, 5-way 5-shot, T=10; chọn trên split validation ISIC, không phải test.
  Phụ thuộc mới: `open_clip` hoặc `transformers` (+ `peft` nếu dùng).
- **Giao diện chung:** thêm vào `models/` phương thức `head_param_indices()` và `grid_shape(x)`.

### WS6 — Dữ liệu

| Việc | Chi tiết |
|---|---|
| [N] bbox CUB | `prepare_cub_bbox.py`: ánh xạ tên file ↔ `images.txt` ↔ `bounding_boxes.txt` của CUB-200-2011 gốc. Ảnh đã bị `prepare_dataset.py` resize **trực tiếp** về 84×84 không giữ tỉ lệ, nên bbox co theo (x·84/W, y·84/H) với W,H lấy từ ảnh gốc. Xuất `test_bbox.csv` (đường dẫn, x, y, w, h). Cần tải `images.txt`, `bounding_boxes.txt` và ảnh gốc (hoặc kích thước ảnh) từ bản phát hành CUB-200-2011 chính thức |
| [N] Waterbirds-style (D2) | `prepare_waterbirds.py` + `loaders/spurious.py`: sampler nhận `c` và dán vật thể lên nền theo nhóm cảnh. Cần mask segmentation CUB và bộ ảnh nền |
| [N] ISIC | `prepare_isic.py` đưa ISIC2018 Task 3 (bản BSCD-FSL) về thư mục theo lớp; transform 224×224 với mean/std của CLIP (`make_transform(size, mean, std)`) |
| [M] 1,200 task | `metatest_iterations: 1200` (D9) |

### WS7 — Thống kê, bảng, hình

- [N] `analysis/stats.py`: Wilcoxon cặp (`scipy.stats.wilcoxon`) + hiệu chỉnh Holm, CI bootstrap 95% (percentile, 10k resample, theo task); trung bình qua seed theo D10.
- [N] `analysis/make_tables.py`: sinh Table 1, 2, 3 (LaTeX) từ thư mục kết quả; in * khi khác biệt có ý nghĩa sau Holm so với á quân.
- [N] `analysis/make_figs.py`: Fig 2 (ρ_reg vs Tα, trục log, mỗi backbone một panel), Fig 3 (a: map task dương/âm nhất; b: bg share vs ΔM; c: Waterbirds).
- [M] `scripts/explain.py`: sửa G8 (thang màu đối xứng theo task); lưu `phi_grid` ra `.npz` để phân tích lại không cần chạy lại GPU.
- Cấu trúc kết quả: `results/<exp>/<scenario>_<backbone>/seed<k>/<protocol>.csv` với cột `task_id, seed, method, metric, value`.

---

## 6. Kế hoạch thí nghiệm

### Giá trị đề xuất cho các ô `[to fill]` (C.1) — để bạn duyệt

| Mục | Đề xuất |
|---|---|
| Superpixel | SLIC `n_sp=150`, compactness 10 (giá trị hiện có trong `bi_adt.py`); `n_sp=50` cho region-level |
| Blur | Gaussian k=11, σ=5 (hiện có); làm mềm biên mask σ=5 |
| Lưới ρ | 0, 0.1, …, 1.0 (11 điểm), quy tắc hình thang |
| Random order | `n_rand=5` |
| Planted patch | kích thước {8,12,16}px, tương phản {0.5,1.0}, tỉ lệ ảnh ảnh hưởng {0.5,1.0}, vị trí ngoài đĩa bán kính 0.25·H quanh tâm |
| Randomization | cascading (output→input) + independent, từng block |
| n_boot | 100 (CI của ΔM) |
| n_β | 16 (Φ^int); thử {4,8,16,32} ở C.3 |

### Pha E0 — Chuẩn bị và kiểm chứng (rẻ, làm trước khi tốn GPU)

| ID | Việc | Đầu ra |
|---|---|---|
| E0.1 | WS2 T1–T8 | Bảng sai số C.3; quyết định D3 |
| E0.2 | Đo lại tỉ lệ ΔM<0 trên mini2cub/tiered2cub × conv4/resnet10 với `test_k_query=24`, kèm CI; làm cả `sm_loss` và `ce_loss` để thấy độ nhạy | Có/không giữ câu "49.7% vs 12.5%" |
| E0.3 | Spearman map cũ vs mới trên checkpoint cũ | Quyết định thay hình trong luận văn |
| E0.4 | Xác minh `second_order` (R4) và tần suất clamp `grad_clip` (R3) | Sửa YAML/paper |

### Pha E1 — Meta-training (đường găng, bắt đầu ngay sau WS0)

| Setting | Backbone | Seed | Số lần train |
|---|---|---|---|
| mini → CUB | conv4, resnet10 | 3 | 6 |
| tiered → CUB | conv4, resnet10 | 3 | 6 |
| tiered, nhãn hoán vị (data randomization) | conv4 | 1–3 | 1–3 |

CE loss, α=0.01, T=5, Adam 1e-3, meta-batch 4, 60k iter. Thời gian chưa biết: đo 200 iteration rồi nhân lên thay vì đoán.
`best` hay `last`: chốt trước một lần (hiện run.sh dùng `--use_last`), không chọn theo kết quả test.

### Pha E2 — Faithfulness chính (Table 1, tiered→CUB, Conv4, 5-way 1-shot, 600 task × 3 seed)

11 phương pháp (Random, Center, Grad-CAM@φ_T, ∂ΔM/∂x, TracIn-CAM, IF-CAM, TracIn, Influence, SGD-influence, FAMA, FAMA-int n_β=16)
× protocol {BI-ADT (ABC↓, ABC↑), counterfactual ảnh (ρ_img), counterfactual vùng (ρ_reg, 100 task), planted (AUROC, sign acc), thời gian}.
Ground truth chung cho mọi phương pháp ⇒ tiết kiệm đáng kể. Thống kê theo WS7.

### Pha E3 — Khi nào adjoint chính xác quan trọng (§5.3, Fig 2)

T ∈ {1,5,10,20} × α ∈ {0.005,0.01,0.02}, 3 phương pháp (Grad-CAM@φ_T, TracIn-CAM, FAMA), ρ_reg vs Tα, mỗi backbone một panel.
Lưu ý ground truth cũng phụ thuộc (T, α) nên phải tính lại cho mỗi tổ hợp; dùng 100 task. Giả thuyết của paper (Prop 3.7) là TracIn-CAM trùng FAMA khi Tακ nhỏ và tách ra khi lớn — **đây là giả thuyết cần kiểm, không phải kết quả đã biết.**

### Pha E4 — Contextual bias (§5.4, Table 2 trái, Fig 3)

1. CUB: bg share của attribution có hại vs ΔM qua 1,200 task (cần bbox).
2. Waterbirds-style (D2): quét c ∈ {0,.25,.5,.75,1}; theo dõi ΔM và bg share của FAMA theo c.
3. Edit-mitigation trên task ΔM<0 với 5 phương pháp (WS3); báo Recovered% theo ρ_edit.

### Pha E5 — Ngoài MAML: CLIP + LoRA trên ISIC (§5.5, Table 2 phải)

Tối thiểu: BI-ADT (ABC↓/ABC↑) cho Random, Grad-CAM@φ_T, TracIn-CAM, FAMA trên ISIC; một ca định tính (thước kẻ/vết mực) chọn **trước** theo tiêu chí ghi sẵn, không chọn theo kết quả đẹp.
Phần này tách rời được; **là ứng viên cắt đầu tiên** nếu thiếu thời gian.

### Pha E6 — Phụ lục

Ablation Table 3 (5 biến thể × ABC↓, ABC↑, ρ_reg, sign acc); ResNet-10 (C.4); sanity C.6 (cascading, independent, data randomization, 3 support edit kèm ΔM); C.3 từ E0.1.

### Nguyên tắc phân tích (đặt trước khi chạy)

- Chỉ số chính cho từng claim và họ so sánh Holm khai báo trước. Kết quả âm được báo cáo; nếu kết quả mâu thuẫn paper (vd Grad-CAM@φ_T bám ΔM tốt hơn dự đoán) thì sửa paper, không sửa phương pháp.
- Siêu tham số (τ của IF-CAM, rank LoRA, ρ_edit) chọn trên split validation, không trên 600 task test.
- Không đổi định nghĩa task giữa các phương pháp: mọi phương pháp dùng cùng file task cố định và cùng trajectory.

---

## 7. Thứ tự thực hiện và mốc

```
WS0 (hạ tầng, config CE, seed, sorted listdir)  ──►  E1 (train 12+ run, chạy nền trên GPU, đường găng)
      │
      └─► WS1 (lõi FAMA) ─► WS2 (test số) ══ CỔNG M1: T1, T3, T4, T5 phải pass ══►
              ├─► WS3 (BI-ADT, CF, planted, sanity)  ┐
              ├─► WS4 (baseline: Grad-CAM, TracIn-CAM, Random/Center trước; IF-CAM, IG, Influence sau)  ├─► E2 ─► E3, E4, E6
              └─► WS6 (bbox CUB, Waterbirds) ─────────────────────────────────┘
WS5-CLIP + ISIC ─► E5 (độc lập, làm cuối)      WS7 (thống kê/bảng/hình) song song với E2
```

| Mốc | Nội dung | Điều kiện qua |
|---|---|---|
| M0 | Chốt D1–D10; WS0 xong; đẩy train lên GPU | Test seed tái lập pass |
| M1 | WS1 + WS2 | T1, T3, T4, T5 pass; so sánh tốc độ với bản cũ |
| M2 | MVP Table 1: FAMA vs Grad-CAM@φ_T, TracIn-CAM, TracIn, SGD-influence, Random, Center trên tiered→CUB Conv4 (BI-ADT + CF vùng) | Có kết quả cặp trên 600 task |
| M3 | Đủ 11 baseline, planted, edit, E3, E4 | — |
| M4 | ResNet-10, sanity, ablation, E5 | — |
| M5 | Thống kê, bảng, hình, sửa text paper (mục 8) | — |

**Nếu thiếu thời gian, cắt theo thứ tự:** E5 (CLIP/ISIC) → IG-pixel → IF-CAM → ResNet-10 → data-randomization → Waterbirds. Headline của paper (Table 1, §5.4 phần CUB, Table 2 trái) vẫn đứng được.

---

## 8. Chỗ phải sửa trong paper (phát hiện khi đối chiếu)

1. ResNet-12 → ResNet-10 (nếu chọn D1).
2. C.1: câu "gradient tuyến tính theo loss" cần sửa cho BatchNorm batch-stats; nêu rõ R dùng pool hay bootstrap (D3).
3. C.2: định nghĩa Grad-CAM@φ_T, Center prior, Random (D6).
4. §4.2: nêu Spearman gộp hay theo task (D7); §4.3: số shot và "tỉ lệ ảnh bị ảnh hưởng" (D8).
5. §5.1 / C.1: ghi MAML là bậc 2 hay bậc 1 theo kết quả E0.4; số ảnh query và loss dùng để ước lượng ΔM; có/không `grad_clip` (R3).
6. §5.4: nêu nguồn "1,200 task" (D9) và định nghĩa Recovered%.
7. C.1: nêu ΔM tuyệt đối dùng cho Φ và ABC, ΔM% chỉ để báo cáo.

---

## Phụ lục A — Danh sách file

| File | Loại | WS |
|---|---|---|
| `main.py`, `scripts/__init__.py`, `scripts/warm_up.py`, `loaders/datasets.py`, `loaders/utils.py`, `configs/*`, `run.sh` | M | 0 |
| `scripts/rng.py`, `tasks/*.json`, `run_paper.sh` | N | 0 |
| `interpreters/fama.py` (viết lại), `interpreters/__init__.py` | M | 1 |
| `tests/test_fama_exact.py` | N | 2 |
| `scripts/check_explain/{bi_adt,sanity_params,sanity_support_set,__init__}.py` | M | 3 |
| `scripts/check_explain/{counterfactual,planted,edit_mitigation}.py`, `loaders/planted.py`, `analysis/bg_share.py` | N | 3 |
| `interpreters/baselines/*.py` | N | 4 |
| `models/clip_lora.py`, (`models/resnet12.py` nếu D1 giữ Res12), `models/{conv4,resnet10}.py` (thêm `head_param_indices`, `grid_shape`) | N/M | 5 |
| `prepare_cub_bbox.py`, `prepare_waterbirds.py`, `prepare_isic.py`, `loaders/spurious.py`, `loaders/transforms.py` | N/M | 6 |
| `analysis/{stats,make_tables,make_figs}.py`, `scripts/explain.py` | N/M | 7 |

## Phụ lục B — Nguồn số liệu trong tài liệu này

- 49.7% / 12.5% (và 45.8%, 52.7%, 7.8%, 17.7% cho các biến thể blur / flip-label): `src/.logs/explain3/{tiered2cub,mini2cub}_{ex,exb,exwf}/adaptation_gain.csv`, tỉ lệ `adaptation_gain < 0` trên 600 task.
- 50.3% (mini2cub-resnet10), 67.3% (cub2mini-resnet10), 14.0% (mini2cub-conv4): `src/.logs/explains/...`.
- 5 vs 120 ảnh query/task: đếm số đường dẫn trong cột `query_path` của `Q_paths.csv` cùng thư mục.
- Quyết định cũ về Res12 → ResNet-10: `edits/replace_res12_with_resnet10.md` (thư mục gitignored).
