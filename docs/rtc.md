# Real-Time Chunking (RTC)

Inference-time guidance that keeps the head of each new action chunk
coherent with the unexecuted tail of the previous chunk — eliminates
chunk-boundary jitter without any training-side changes. Split across
a fork of openpi (server) and the robots_realtime policy agent (client).

Paper reference: [Real-Time Execution of Action Chunking Flow Policies](https://arxiv.org/abs/2506.07339)
(Black, Galliker, Levine — 2025).

## TL;DR

Flow-matching policies (π₀, π₀.₅, SmolVLA) emit **action chunks** of length
`action_horizon`. A naive client consumes the chunk, re-inferences for a
fresh one, and hard-swaps when it arrives. Because the new chunk is
generated from a fresh obs by a stochastic model, the first few actions
of the new chunk can disagree with the last few of the old — the arm
**jitters at every chunk boundary**.

RTC's trick: inside each denoising step, take the current denoising state
`x_t`, compute the Euler projection to `t=0` (`x1_t = x_t − time·v_t`),
compare it to the prefix the client is still committed to executing, and
use the VJP (JAX) / autograd (PyTorch) of `x1_t` w.r.t. `x_t` to nudge
`v_t` toward producing chunks that agree with that prefix. The correction
is weighted by a time-dependent scalar that peaks mid-denoising and tapers
at both ends.

## Architecture

```
                           ┌──────────── client (robots_realtime) ─────────────┐
                           │  AsyncDiffusionAgent._action_loop (bg thread):    │
                           │    • captures last_actions[counter:] as prefix    │
                           │    • pads to fixed (action_horizon, A) shape      │
                           │    • sends action_prefix + inference_delay        │
                           │      + execution_horizon + max_guidance_weight    │
                           │      in the obs dict                              │
                           └──────────────────────┬────────────────────────────┘
                                                  │  msgpack + websocket
                                                  ▼
┌──────────────────────── server (openpi fork) ────────────────────────────────┐
│  Policy.infer():                                                             │
│    • pops RTC keys off obs                                                   │
│    • NORMALIZES action_prefix via inverse of output_transform.Unnormalize    │
│    • forwards as sample_kwargs to Pi0.sample_actions                         │
│                                                                              │
│  Pi0.sample_actions():                                                       │
│    • if action_prefix is None  → runs the pre-RTC denoiser (unchanged)       │
│    • if action_prefix is given → dispatches to sample_actions_rtc            │
│                                                                              │
│  Pi0.sample_actions_rtc():                                                   │
│    • lax.while_loop over denoising steps, each step:                         │
│        jax.vjp(λ x: (x − time·v(x), v(x)),  has_aux=True)                    │
│        err = (prefix_padded − x1_t) × prefix_weights                         │
│        correction = vjp_fn(err)                                              │
│        guidance_weight = min(c(τ)·inv_r2(τ), max_guidance_weight)            │
│        v_t ← v_t − guidance_weight·correction                                │
│        x_t ← x_t + dt·v_t                                                    │
│    • returns the final denoised chunk                                        │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Enabling RTC

**Server**: must be the openpi fork with the RTC hunks applied (pi0.py,
policy.py). No env vars required for normal operation. Optional debug:

```bash
# Diagnose recompilation:
JAX_LOG_COMPILES=1 uv run scripts/serve_policy.py policy:named-checkpoint --policy.name ...

# Per-denoising-step RTC stats (adds host-device sync latency; keep off in prod):
RTC_DEBUG=1 uv run scripts/serve_policy.py ...

# Both:
JAX_LOG_COMPILES=1 RTC_DEBUG=1 uv run scripts/serve_policy.py ...
```

**Client**: in the YAML, set `inference_mode: async_rtc` on the
AsyncDiffusionAgent. That's the only mode that emits prefix; the other
three modes (`sync`, `async`, `async_rate_limited`) work as before.

```yaml
- type: AgentNode
  agent_class: robots_realtime.agents.policy_learning.async_pi0_agent:AsyncDiffusionAgent
  agent_kwargs:
    inference_mode: async_rtc
    inference_interval_s: 0.5        # rate cap; see tuning notes below
    rtc_max_guidance_weight: 1.0     # server clamp on guidance weight (see below)
    rtc_consumer_rate_hz: 30         # used to convert latency → inference_delay ticks
    rtc_debug: true                  # client-side chunk-jump log (cheap)
```

## Knobs

### Client side (AsyncDiffusionAgent kwargs)

| kwarg | what it does | default | when to change |
|---|---|---|---|
| `inference_mode` | picks the async / sync / RTC path | `async_rate_limited` | set `async_rtc` to enable RTC |
| `action_horizon` | chunk length (must match server) | `30` | match the server's `action_horizon` |
| `inference_interval_s` | universal rate cap for any async mode | `0.5` | set lower for slower GPUs; `None` for flat-out |
| `min_smoothed_actions` / `max_smoothed_actions` | client-side linear blend at chunk boundary | `1` / `8` | redundant when RTC is on — leave defaults |
| `rtc_execution_horizon` | overrides the "real prefix length" the server is told | `None` (auto = length of prefix tail) | rarely needed; auto is usually right |
| `rtc_max_guidance_weight` | upper clamp on server's guidance weight | `None` (server default `2.0`) | lower if arm is jerky, raise if RTC barely tracks the prefix |
| `rtc_consumer_rate_hz` | converts ms inference latency → `inference_delay` ticks | `30.0` | match your `poll_freq` |
| `rtc_debug` | client chunk-jump log per merge | `False` | turn on while tuning |

### Server side (Pi0.sample_actions_rtc kwargs, via obs dict)

| obs key | what it does | default (server) |
|---|---|---|
| `action_prefix` | (T_prev, A) or (1, T_prev, A) — the prefix. The server normalizes it into model action space. | required |
| `inference_delay` | how many steps the robot is "already committed" to; guidance weight = 1 for idx < this | `0` |
| `execution_horizon` | beyond this idx, guidance weight = 0 (model free-forms) | full chunk |
| `max_guidance_weight` | clamp on the time-varying guidance weight schedule | `2.0` |

## Reading the logs

### Client (`rtc_debug: true` in YAML)

One line per chunk merge:

```
[AsyncDiffusionAgent.rtc] chunk-jump  full_norm=0.0432  arm_norm=0.0401
    consumed_during_inference=4  infer_dt=180ms  server_chunk_len=30  new_len=26
    delay_ema_ticks=5.4
```

| field | healthy | bad |
|---|---|---|
| `arm_norm` | `< 0.1 rad` | `> 0.3 rad` = arm will jump visibly |
| `consumed_during_inference` | `5–15` ticks at 30 Hz / 0.5 s inference | approaching `action_horizon` = inference is too slow |
| `infer_dt` | stable once warm (~100–500 ms) | growing or erratic = JIT recompilation or server stall |
| `new_len` | ≈ `action_horizon − consumed_during_inference` | `<= 2` = degenerate merge skipped (see "known caveats") |
| `delay_ema_ticks` | should stabilize within the first ~5 inferences | unbounded growth = inference getting slower |

### Server (`RTC_DEBUG=1` env var)

One line per denoising step (10 per inference by default), inside the JIT'd loop:

```
[rtc] t=0.8 gw=1.0 |v_t|=1.23 |corr|=0.18 |err|=0.15 |x1_t|=0.89 |prefix|=0.92
```

| field | healthy | bad |
|---|---|---|
| `|v_t|` | stable across steps within one inference | wildly varying = model instability (not RTC's fault) |
| `|corr|` | smaller than `|v_t|` at every step | larger than `\|v_t\|` → guided v_t sign-flips → arm shoots forward |
| `|err|` | drops across denoising as x1_t converges to prefix | rising = guidance not tracking |
| `|x1_t| − |prefix|` | shrinks to ~0 by `t=0.1` | stays large = prefix incompatible with obs (model disagrees strongly) |
| `gw` | binds to `max_guidance_weight` at edges only (`t≈1`, `t≈0`); smooth in the middle | clamped throughout = `max_guidance_weight` too low |

### JIT compile log (`JAX_LOG_COMPILES=1`)

A healthy run:

```
Compiling sample_actions  # startup, non-RTC path
... (N inferences, no new compiles) ...
Compiling sample_actions_rtc  # first RTC inference after pause → unpause
... (no more compiles, ever) ...
```

If you see `Compiling sample_actions_rtc` on every RTC inference, some
input shape or Python-static value is changing call-to-call. The client
currently pads `action_prefix` to a fixed `(action_horizon, A)` so the
shape stays constant; value-varying scalars (`inference_delay`,
`execution_horizon`, `max_guidance_weight`) are JAX scalars not static
ints, so their value changes don't recompile.

## Known caveats / future work

### 1. First RTC inference is slow

The `sample_actions_rtc` JIT compile takes ~5–10 s on first call (fresh
trace of a different code path from the non-RTC denoiser). During that
time the consumer drains the existing chunk; the degenerate-short-chunk
guard in `_action_loop` detects this (`new_len < 2`) and keeps the old
buffer rather than swapping in a length-1 stub.

### 2. Prefix-weight schedules

LeRobot exposes three schedules for `prefix_weights`:

| schedule | shape | when |
|---|---|---|
| `zeros` | `1.0` for `idx < inference_delay`, `0.0` after — sharp cutoff | you're highly confident about committed ticks and want no guidance past them |
| `ones`  | `1.0` up to `execution_horizon`, `0.0` after — full-strength guidance, no ramp | when you want aggressive coherence throughout |
| `linear` | `1.0` up to `inference_delay`, linear ramp to `0.0` at `execution_horizon` | safest default; smooth transition between committed / free |

Ours hard-codes `linear`. To expose: add a `schedule: Literal["linear","zeros","ones"] = "linear"` kwarg and a 3-line branch in `sample_actions_rtc` to pick. No reason to change unless you're diagnosing specific behavior.

### 3. Latency tracker / debug visualizer

LeRobot ships two more utilities we didn't port:

- **`DebugStep` + `Tracker`** (`modeling_rtc/debug_tracker.py`): collects
  per-step denoising state (x_t, v_t, x1_t, correction, err, weights,
  guidance_weight, time) into a sliding window for post-run analysis.
  Serializes to a dict for plotting.
- **`latency_tracker.py`**: separate concerns — rolling statistics on
  inference round-trip time, helps diagnose when the server is the
  bottleneck vs. the network.

Our `RTC_DEBUG=1` prints cover the same signals but only live — no
post-hoc analysis. If you want a proper tracker, the cleanest port is
to replace the `lax.while_loop` in `sample_actions_rtc` with `lax.scan`
so you get the sequence of intermediate states as a stacked array,
then pass it back in the output dict and let the client serialize.

### 4. RTC on models not trained with it

Per the RTC paper, this is **inference-time only** — no training change.
Works for any flow-matching policy (π₀ family, SmolVLA). If you ever
train a model with RTC in the loop (forcing consistency across chunks
during training), that would be a separate effort in the openpi
training code, not this inference-time guidance layer.

### 5. Prefix is normalized, padding is zero in model-space

The client's prefix is in client-space (post-Unnormalize) — the policy
server normalizes it before use, which lives in
`Policy._rtc_action_normalizer`. Zero-padded time positions (when the
client's tail is shorter than `action_horizon`) become the normalized
value of zero in client-space, which is **not** zero in model-space
(it's `−mean/std`). For YAM + π0, this is harmless: `execution_horizon`
told the server to weight only the real tail length, so the zero-padded
tail positions have `weight = 0` and never contribute to `err`.

### 6. Padded-action-dim leakage

Similarly, the server pads the action dimension from the client's
`A_client` (14 for bimanual YAM) to the model's `A_model` (32) with
zeros. The guidance `err` on those padding dims is `(0 − x1_t_pad)`,
which could in principle pull the model's Jacobian in a weird direction.
In practice the model was trained to output ~0 on padding dims for
single-task clients, so this is a no-op. If you ever suspect it matters,
the fix is to slice `prefix_weights` to only apply to the first
`A_client` action dims.

## One-line summary per mode

| `inference_mode` | sends RTC prefix | inference blocks consumer | chunk-boundary smoothing |
|---|---|---|---|
| `sync` | no | yes (during every chunk swap) | hard swap; no client blend |
| `async` | no | no | client linear blend of `num_smoothed = consumed_during_inference` actions |
| `async_rate_limited` | no | no | client blend, rate-capped |
| `async_rtc` | **yes** | no | **server-side model guidance** + (redundant) client blend |

## File index

Server side (`~/openpi`):

| file | what lives there |
|---|---|
| `src/openpi/models/pi0.py` | `sample_actions` (unchanged pre-RTC) + `sample_actions_rtc` (new); reads `RTC_DEBUG` env var |
| `src/openpi/policies/policy.py` | pops RTC keys off obs, normalizes prefix, forwards to `sample_actions` |
| `src/openpi/transforms.py` | `Normalize` / `Unnormalize` — the action normalizer is constructed from the `Unnormalize` step in `output_transforms` |

Client side (`~/robots_realtime`):

| file | what lives there |
|---|---|
| `robots_realtime/agents/policy_learning/async_pi0_agent.py` | `AsyncDiffusionAgent` with the four inference modes; RTC prefix capture, padding, EMA delay, chunk-jump log |
| `configs/yam/yam_bimanual_openpi_policy_xdof_hq.yaml` | example session wiring |
