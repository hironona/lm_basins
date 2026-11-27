# lm_basins

Geometry of activation plateaus and steering basins in language models, with TransformerLens.
Split out of `activation_plateaus_toy_models` (the LM pipelines `vis_plateaus_lm/` and `steering_basins/`).

## Layout

```
common/                  reusable across experiments (import as `common.<module>`)
  config.py              YAML config + `--set key=value` overrides, device choice
  lm.py                  TransformerLens model loading, tokenization, chat prompt, EOS ids, hook names
  steering.py            steered passes with a reusable KV cache, continuations, greedy decoding
  joint_kl.py            m-token joint KL estimator (prefix tree + Monte-Carlo tail)
  metrics.py             next-token KL, norm ratio, in-plane projection
  last_token.py          last-token forward from a layer, Jacobian norm to the logits
  surfaces.py            flat / slerp-like surfaces through three points, grids on them
  plotting.py            minimal matplotlib style, heat maps, boundaries, arrow fields
experiments/
  plateau_plane/         argmax tokens, Jacobian norm and KL on a surface through three sentences (GPT-2)
  steering_basins/       steering vectors, basin check, joint-KL and norm-ratio heatmaps (Qwen3-8B)
tests/                   pytest on a tiny random Qwen3 (CPU, ~1 min)
```

A new experiment is a new folder in `experiments/` with a `config.yaml`, a compute script and a plot script. Scripts
put the project root on `sys.path` and import from `common`. Compute scripts write `.pt` files to `outputs/<experiment>/`
and plot scripts read only those files, so you can plot on a CPU.

**Layer convention.** Layer L is the residual stream *entering* block L (`blocks.L.hook_resid_pre`; 0 = the embeddings,
n_layers = the output of the last block). Models are loaded with `from_pretrained_no_processing`, so this is exactly
HF's `hidden_states[L]`.

## Setup

```bash
uv sync
uv run pytest            # tiny random Qwen3: TransformerLens == HF, joint KL vs exact enumeration, steering scope, ...
```

Models come from the Hugging Face cache. Set `HF_HUB_CACHE=<dir>` to use another cache, and set `HF_HUB_OFFLINE=1` on machines
without internet. All commands run from the project root. `--set key=value` overrides any config entry.

## plateau_plane

Three sentences `prefix + " " + word` give three last-token activations at layer L. On a grid over the surface through
them (`flat` plane, or a `spherical` slerp-like surface whose edges from the first anchor are slerp paths with linearly
interpolated norm), every point gets its argmax next token, ‖d logits / d x‖_F, and KL(P_a ‖ P_x) to each anchor a.

```bash
python experiments/plateau_plane/compute.py --set layer=6 surface=flat
python experiments/plateau_plane/plot.py outputs/plateau_plane/gpt2/<file>.pt
```

## steering_basins

For the en→zh and refusal steering vectors v (Finlayson et al. 2026), on contexts drawn at random from held-out data:

1. **Steering vectors** at the input of block L: mean zh minus mean en Wikipedia token activation; harmful minus
   harmless activation at the last chat-prompt token.
2. **Basin check.** The circle probe γ(θ) = ‖x‖(x̂ cos θ + u sin θ), with u the tangent of v at x, goes through blocks
   L..L+3, and its speed ‖dη/dθ‖ shows basins separated by barriers. Two steered points on the ray x + αv:
   α* (the paper's adaptive steering, which reaches the target basin θ*) and α_cls (the first α whose greedy
   continuation has the target label). A greedy sweep along the ray gives the behavior labels.
3. **Grid.** On `n_planes` planes of offsets a v + b u_k (u_k random, ⟂ v, ‖u_k‖ = ‖v‖), added at the context positions
   (`scope`), everything is measured right after the context and after each of the first `n_decoded` unsteered greedy
   tokens y_j (teacher-forced, never steered):
   - **joint KL** D_m = KL(P(X_1..X_m | ctx, y_≤j) ‖ P_steered(…)) via the chain rule with exact per-step KL. A
     probability-pruned prefix tree of P gives an exact lower bound (head), and Monte-Carlo samples give an unbiased
     tail with its standard error. Every m' ≤ m comes out of the same pass. `tree_nodes` and `mc_samples` set the accuracy.
   - **norm ratio** log10 R = log10((‖f(h) − f(x)‖ / s_out) / (‖h − x‖ / s_src)), from block L's input to the last block's
     output, with the plane's RMS spreads s removing the overall gain (< 0 contraction, > 0 expansion);
   - **residual shift** arrows: [(f(h) − f(x)) − (h − x)] projected onto (u_k, v) in units of ‖v‖, i.e. how blocks
     L..last move the steered point within the plane relative to the unsteered one;
   - the **greedy label** (en / zh / other by script; refusal / non-refusal by refusal phrases).
4. **Trajectory-difference criterion** (`diff_points`, default both α_cls and α*). From the context steered by α v, the
   greedy tokens z_1..z_J; from the unsteered context, y_1..y_J. At each j, d_j = h_j − x_j is the input of block L at
   the last position of the steered ctx + z_≤j minus that of the unsteered ctx + y_≤j, and cos(d_j, v) is reported
   (d_0 = α v, so cos = 1 at j = 0). On `n_planes` planes x_j + a d_j + b u_k (u_k random ⟂ d_j, ‖u_k‖ = ‖d_j‖),
   placed at that last position only (the prefix ctx + y_≤j is otherwise unsteered; (a, b) = (1, 0) is h_j), the same
   KL / label / norm ratio / residual shift are measured → `*_diff_<point>.png` (rows j, cos(d_j, v) in the row labels)
   and `*_diff_cos.png` (cos(d_j, v) and ‖d_j‖/‖v‖ against j). For j ≥ 1 the steering offset never reaches position
   n−1+j at block L's input, so d_j there comes only from the different tokens z_≤j vs y_≤j.

```bash
python experiments/steering_basins/prepare_data.py                      # once, needs internet
python experiments/steering_basins/run.py --quick                        # short end-to-end test
python experiments/steering_basins/run.py                                # full run (one GPU)
python experiments/steering_basins/plot.py outputs/steering_basins
```

With several GPUs: `--stage vectors` once, then one process per GPU with `--stage basin` and then `--stage grid`
(`--shard i --num-shards n`), then `--stage merge`. The outputs are `vectors.pt`, `contexts.json`,
`<experiment>/<context>/{basin.pt, plane_XX.pt, result.pt, summary.json}`, `summary.json` and `plots/`.
