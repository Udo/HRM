# Adding a New Problem Domain (Concise How‑To)

Goal: integrate a new symbolic/grid reasoning task so HRM can train & infer on it

---
## 1. Encode Training Data
Target disk layout mirrors existing domains:
```
<data_root>/
  train/  all__inputs.npy  all__labels.npy  all__puzzle_indices.npy  all__group_indices.npy  all__puzzle_identifiers.npy  dataset.json
  test/   (same fields)
  identifiers.json
```
File semantics:
* all__inputs.npy  (int32) shape (N, seq_len)
* all__labels.npy  (int32) shape (N, seq_len) – token targets (pad positions may be 0 or ignored via ignore_label_id in metadata)
* all__puzzle_indices.npy  (int32) shape (P+1,) prefix-sum delimiting examples belonging to each original puzzle (p0..p{P-1}); final entry = N.
* all__group_indices.npy   (int32) shape (G+1,) prefix-sum delimiting augmentation groups (for balanced sampling). Often G==P if 1 group per puzzle.
* all__puzzle_identifiers.npy (int32) shape (N,) puzzle id per example (0 reserved for blank). Enables optional learned puzzle embedding.
* dataset.json  (metadata) must include at least:
  - seq_len
  - vocab_size
  - pad_id (int)
  - ignore_label_id (int or null) – labels with this id excluded from loss (set = pad_id or null)
  - blank_identifier_id (int) usually 0
  - num_puzzle_identifiers (int) (#distinct +1 for blank; if you assign ids 0..K, set = K+1)
  - total_groups (int)
  - mean_puzzle_examples (float) (N / P)
  - sets: ["all"] (current loader expectation)
* identifiers.json  List[str] length num_puzzle_identifiers giving textual names (index 0 = "<blank>").

Tokenization pattern (recommend):
* Reserve 0 for PAD.
* Shift natural symbol set upward by +1 or +2 to keep 0 free (and optionally 1 for EOS/border if you need a boundary token like ARC).
* Keep vocab dense in [0, vocab_size-1].

Sequence construction strategies:
* Pure grid: flatten HxW row-major so seq_len = H*W (Sudoku 9x9 -> 81; Maze/ARC 30x30 -> 900).
* Variable shapes: pad into a fixed canvas; optionally place an EOS / boundary token (treated as normal class) to encode extent.
* Multi-example reasoning: group_indices should aggregate related variants (original + augmentations) of the same latent puzzle so the sampler can balance across groups.

Minimal builder template (pseudo):
```python
# produce arrays: inputs, labels (N, L) int32
# produce arrays: puzzle_indices (P+1,), group_indices (G+1,), puzzle_identifiers (N,)
# write dataset.json using TinyMetadata / PuzzleDatasetMetadata fields
# write identifiers.json
```
Look at `dataset/build_tiny_dataset.py` for a tiny end-to-end concrete pattern.

Edge cases to handle:
* Ensure prefix arrays end with N.
* If you have puzzles with variable #examples, puzzle_indices[i+1]-puzzle_indices[i] gives that count.
* Set ignore_label_id=null when every position should contribute to loss (e.g. ARC). Use pad_id when padded cells should be ignored (e.g. inside a smaller grid embedded in a canvas but you prefer not to learn boundary tokens).

---
## 2. Configure Training
No new code needed if you reuse generic loader contract. Steps:
1. Place dataset at e.g. `data/mydomain-1k/` with structure above.
2. Run training:
```
python pretrain.py data_path=data/mydomain-1k epochs=E eval_interval=I global_batch_size=GBS \
  lr=... puzzle_emb_lr=... weight_decay=... puzzle_emb_weight_decay=... \
  arch.hidden_size=... arch.num_heads=... arch.H_layers=... arch.L_layers=... arch.H_cycles=... arch.L_cycles=...
```
Hydra overrides after `--` also work. Ensure `eval_interval` divides `epochs`.
Optional env toggles: `HRM_DEVICE`, `HRM_CLIP_GRAD`, `HRM_QK_CLAMP`, `DISABLE_COMPILE`, `HRM_DISABLE_WANDB`.
If you introduced a boundary/EOS token that should be learned, do NOT set it as ignore_label_id.

Validate metadata quickly (Python REPL):
```python
import json, numpy as np
meta=json.load(open('data/mydomain-1k/train/dataset.json'))
for k in ['seq_len','vocab_size','pad_id','num_puzzle_identifiers']: print(k, meta[k])
print('inputs shape', np.load('data/mydomain-1k/train/all__inputs.npy').shape)
```

---
## 3. Encode a Single Problem for Inference
At inference, you need one or more examples sharing the same seq_len & vocab.
Options:
* Reuse test split: simplest (already formatted).
* Ad-hoc encode: build arrays with shape (1, seq_len). For missing / padded cells use pad_id. Apply same symbol shift as during training.
* Provide a minimal dataset folder clone (train/test identical) if you want to use existing CLI evaluation scripts unchanged.
Make sure puzzle_identifiers array is length N (use 0 if no specific id) and puzzle_indices=[0,1], group_indices=[0,1] for a single-example dataset.

Quick ad-hoc folder (single sample):
```
my_infer_ds/
  test/  all__inputs.npy  all__labels.npy (optional dummy) ... metadata ...
```
You may set labels to a copy of inputs if ground truth unknown; metrics will be meaningless but logits are produced.

---
## 4. Run Inference / Evaluation
Two approaches:

A) Using existing evaluation harness (preferred when you saved a checkpoint with `all_config.yaml`):
```
python evaluate.py checkpoint=checkpoints/<ProjectName>/<RunName>/model_step_<S>.pt \
  data_path=data/mydomain-1k  # only needed if config fallback occurs
```
Outputs: metrics printed; optionally configure what to dump via `save_outputs=[...]` CLI list (see evaluate.py default fields).

B) Forward pass manually (custom script):
```python
import torch, yaml, numpy as np
from pretrain import PretrainConfig, init_train_state, create_dataloader
cfg=PretrainConfig(**yaml.safe_load(open('checkpoints/.../all_config.yaml')))
train_state=init_train_state(cfg, train_metadata=None, world_size=1)  # metadata auto from dataloader later
state=torch.load('checkpoints/.../model_step_XX.pt', map_location='cpu')
try: train_state.model.load_state_dict(state, assign=True)
except: train_state.model.load_state_dict({k.removeprefix('_orig_mod.'):v for k,v in state.items()}, assign=True)
arr=np.load('data/mydomain-1k/test/all__inputs.npy')[:1]
with torch.no_grad():
    logits=train_state.model(torch.from_numpy(arr))  # adapt to model forward signature if different
```
(Use actual loader path; above is schematic—prefer `create_dataloader` for correct batching & device moves.)

For quick sanity without full dataset, fabricate a tiny set using section 3 then run `evaluate.py` (it will infer minimal config if `all_config.yaml` missing).

---
## 5. Checklist Before Training
- [ ] Shapes consistent: inputs/labels (N,L), identifiers (N,), prefix arrays end at N.
- [ ] metadata.vocab_size > max token id.
- [ ] pad_id matches zeros in arrays.
- [ ] ignore_label_id either null or equals pad_id (or sentinel to skip ignoring).
- [ ] num_puzzle_identifiers == 1 + max(puzzle_identifiers).
- [ ] identifiers.json length == num_puzzle_identifiers.
- [ ] eval_interval divides epochs.

---
## 6. Common Pitfalls
| Issue | Fix |
|-------|-----|
| Mismatch seq_len vs array second dim | Regenerate arrays or adjust padding logic. |
| Loss never decreases | Wrong token shift; confirm label ids within [0,vocab_size). |
| All accuracy ~0 | Possibly set ignore_label_id incorrectly (ignored all tokens). |
| Evaluation crash missing all_config.yaml | Provide dataset via `data_path=` so fallback config has correct vocab/seq_len. |
| Puzzle embedding shape mismatch | num_puzzle_identifiers changed between dataset versions; retrain or adjust embedding size. |

---
## 7. Minimal Generator Snippet
```python
import numpy as np, json, os
N=8; H=W=10; seq_len=H*W; vocab_base=5
inp=np.random.randint(1,vocab_base+1,(N,seq_len),dtype=np.int32)
lab=np.random.randint(1,vocab_base+1,(N,seq_len),dtype=np.int32)
pi=np.arange(N+1,dtype=np.int32); gi=pi.copy(); ids=np.zeros(N,dtype=np.int32)
meta=dict(seq_len=seq_len,vocab_size=vocab_base+1,pad_id=0,ignore_label_id=0,blank_identifier_id=0,
          num_puzzle_identifiers=1,total_groups=N,mean_puzzle_examples=1.0,sets=["all"])
root='data/mydomain-demo'; os.makedirs(root+'/train',exist_ok=True); os.makedirs(root+'/test',exist_ok=True)
for split in ['train','test']:
  np.save(f'{root}/{split}/all__inputs.npy',inp)
  np.save(f'{root}/{split}/all__labels.npy',lab)
  np.save(f'{root}/{split}/all__puzzle_indices.npy',pi)
  np.save(f'{root}/{split}/all__group_indices.npy',gi)
  np.save(f'{root}/{split}/all__puzzle_identifiers.npy',ids)
  json.dump(meta,open(f'{root}/{split}/dataset.json','w'))
json.dump(["<blank>"],open(f'{root}/identifiers.json','w'))
print('Wrote',root)
```
Run training afterward pointing `data_path` to `data/mydomain-demo`.


---
## 8. Game Integration Tips

### 8.1 Choosing a Representation
Pick a fixed canvas (H x W) that upper-bounds your largest board/state. Flatten row‑major.

Token design patterns:
* Tile Types: Reserve contiguous id block (e.g. floor=1, wall=2, enemy=3, treasure=4, exit=5).
* Entity Layers: If multiple entities can share a cell, either (a) compose into a single categorical token via small combinatorial code table, or (b) render priority order (e.g. entity overrides terrain) then optionally add a second pass with an auxiliary mask (can be concatenated as another “puzzle variant” within same group to give multi-channel info).
* Attributes (HP, status): Quantize into buckets and map to extra tokens (e.g. HP 0..9 -> +10 offset). Keep total vocab tight; avoid sparse huge id ranges.
* Fog / Unknown: dedicate a token (e.g. 0=PAD for off‑board, 1=UNKNOWN, shift real content by +2) if partial observability matters.
* Temporal Context: Provide k recent frames as k separate examples inside the same group (share puzzle_identifier) so the model can infer state transitions; order can be implied by position in group (earlier indices) or by reserving small “time marker” tokens written into unused tail cells.

### 8.2 Grouping Strategy
Use group_indices to package together: [current_state, goal_spec, hint_examples, past_frames]. Model can attend across them during training (balanced sampler). For inference you can feed the same multi-example group to get a plan conditioned on all pieces.

Example grouping (platformer level assist):
1. Frame t-2
2. Frame t-1
3. Current frame t
4. Goal sketch (desired end layout / objective markers)
Labels: predicted next optimal frame (t+1) or a solved/annotated path overlay. All four inputs + one label = 5 examples; puzzle_indices prefixes reflect 5.

### 8.3 Using ACT Steps (Iterative Refinement)
Each forward pass internally performs multiple halting steps. For interactive UIs you can visualize intermediate logits to show the “thinking” progression (use visualization code as reference). You can early‑stop after a fixed number of steps for latency-sensitive frames (set arch.halt_max_steps lower) or run full to maximize accuracy during pause screens.

### 8.4 Inference Loop Patterns
Real-time (per frame):
1. Serialize current game state grid -> int32 token array.
2. Batch multiple concurrent player instances (stack along batch dim) to amortize GPU call.
3. Run model -> logits (B, L, vocab).
4. Derive next action map: for each cell take argmax or sample top-k; optionally extract a special “action head” region if you reserve final cells to encode global action proposals (e.g. last 16 positions store high-level action tokens instead of board cells).
5. Convert predicted tokens back to engine events (mapping table reverse of encoding).

Turn-based / puzzle solve:
* Run once; if output not consistent (e.g. violates rules) apply rule-based repair or re-sample low-confidence cells (logit margin < threshold).
* Optionally perform iterative self‑refinement: feed model’s own prediction as new input variant inside same group to allow another pass (curriculum style) — treat each iteration as a fresh forward; stop when stable.

### 8.5 Confidence & Fallback
* Per-cell confidence: softmax(logits) max probability. If below e.g. 0.55, trigger heuristic / search fallback just for that cell.
* Whole-plan confidence: mean max prob over non-pad cells or exact_accuracy from evaluate harness run on a validation holdout.
* Multi-sample: temperature sample K variants (temp 0.8..1.0) and pick one maximizing downstream heuristic score (e.g. path length feasibility, resource balance).

### 8.6 Puzzle Identifier Usage
Assign stable puzzle_identifiers for recurring levels so the learned embedding captures persistent structural quirks (layout style). For procedurally generated levels you can:
* Use 0 for all (no specialization), or
* Hash a coarse level signature (dimensions, biome type) -> small id space (cap at, say, 256). Track mapping in identifiers.json; avoid explosive growth.

### 8.7 Streaming / Partial Updates
If only a small region changes (player moves 1 tile), you can either:
* Re-encode full grid (simplest), or
* Maintain cached embedding states (requires model surgery — not provided). Start with full re-encode; optimize later if needed.

### 8.8 Tooling Hooks
* Use `evaluate.py` with a custom tiny test folder representing current live snapshot series to get structured metrics / dumps (e.g. save logits to drive UI heatmaps).
* Adapt `visualize_*_cli.py` logic to build an in-engine overlay (heatmap of correctness probability or planned path updated each step).

### 8.9 Latency Optimization
* Reduce seq_len: pack only relevant viewport instead of entire world.
* Reduce cycles: lower `arch.H_cycles` / `arch.L_cycles` or cap halting steps.
* Mixed precision (if on CUDA) once stable.
* Batch multiple NPC decisions together.

### 8.10 Typical Encoding Table (Example)
| Concept | Token Id |
|--------|----------|
| PAD / off-map | 0 |
| Unknown fog | 1 |
| Floor | 2 |
| Wall | 3 |
| Player | 4 |
| Enemy | 5 |
| Exit / Goal | 6 |
| Key Item | 7 |
Extend contiguously; update vocab_size accordingly.

Reverse mapping must exist in engine code for rendering / action selection.

### 8.11 Action Extraction Pattern
If the model’s task is state prediction but you want discrete actions (e.g., move N/E/S/W):
1. Predict next-state grid.
2. Compare player position cell between current & predicted to infer intended move.
3. If unchanged & low confidence globally -> fallback to scripted AI.
Alternative: dedicate final 4 cells of sequence to action logits (encode them as dummy positions in training labels), allowing direct action decoding.

### 8.12 Safety / Rule Enforcement
Always validate model proposals against game rules (path passable, inventory constraints). Quick filter before committing updates keeps training objective pure (predict optimal) while runtime stays safe.

### 8.13 Logging & Telemetry
Store (state, prediction, confidence, chosen action) tuples periodically; you can later distill a smaller policy or fine-tune HRM on failure cases (active learning loop).

---
End of game dev tips.

