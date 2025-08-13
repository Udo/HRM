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
Keep this doc tight; for deeper rationale on grouping/augmentation examine existing domain builders (`dataset/build_*_dataset.py`).
