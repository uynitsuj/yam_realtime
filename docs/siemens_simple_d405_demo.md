# Siemens simple-D405 pi0.5 demo

This worktree is the deployment counterpart of OpenPI config
`pi05_siemens_simple_d405_bs128` at commit `af7014b`.

## Pinned paths

- robots_realtime: `/nfs_us_2/karim/worktrees/robots-realtime-siemens-simple-d405`
- OpenPI: `/nfs_us_2/karim/worktrees/openpi-industrial-packing-v3`
- session config: `configs/yam/yam_bimanual_openpi_policy_xdof_hq_us05_siemens_simple_d405.yaml`
- checkpoint: `/nfs_us_2/siemens/policy_ckpts/pi05_siemens_simple_d405_bs128/siemens_simple_d405_pi05_20260901/14999`

The deployment contract must stay aligned with training:

| Training property | Deployment setting |
| --- | --- |
| pi0.5, 30-step action horizon | `action_horizon: 30` |
| Three D405 views at 30 Hz | three `RealSenseCamera` nodes at VGA/30 |
| 224x224 resize-with-pad, full FOV | `publish_resize_mode: pad`, `publish_fov_crop: 1.0`, `image_preprocess: pad` |
| Prompt `industrial packing` | explicit client `prompt` and server `--default-prompt` |
| Converter reverses each arm's six joints | `flip_joint_order: true` |
| 14-D absolute position actions | `use_joint_state_as_action: false` |

Do not use the ZED crop config for this checkpoint. The simple-D405 dataset did
not use a ZED top camera or a deployment-time FOV crop.

## One-time environment setup

The robot worktree's required git submodules are already initialized. Create or
refresh isolated environments if `.venv` does not exist:

```bash
cd /nfs_us_2/karim/worktrees/openpi-industrial-packing-v3
uv sync --frozen

cd /nfs_us_2/karim/worktrees/robots-realtime-siemens-simple-d405
uv sync --frozen --extra realsense
```

## Preflight

From the robots_realtime worktree:

```bash
uv run --frozen scripts/preflight_siemens_simple_d405_demo.py
```

Run it again after the inference server is up and the robot is connected:

```bash
uv run --frozen scripts/preflight_siemens_simple_d405_demo.py \
  --require-server --require-hardware
```

The configured US05 D405 mapping is:

- top: `427622273494`
- left: `218622275506`
- right: `218722271050`

Verify the physical views before unpausing if cameras have been remounted.

## Start the demo

Terminal 1, OpenPI server:

```bash
cd /nfs_us_2/karim/worktrees/openpi-industrial-packing-v3
uv run --frozen scripts/serve_policy.py \
  --default-prompt "industrial packing" \
  policy:checkpoint \
  --policy.config=pi05_siemens_simple_d405_bs128 \
  --policy.dir=/nfs_us_2/siemens/policy_ckpts/pi05_siemens_simple_d405_bs128/siemens_simple_d405_pi05_20260901/14999
```

Wait for the websocket server to listen on port 8012. The first inference will
compile the model and can be much slower than steady state.

Terminal 2, robot session:

```bash
cd /nfs_us_2/karim/worktrees/robots-realtime-siemens-simple-d405
./scripts/setup_can_yam_bimanual_us05.sh
uv run --frozen rr-session \
  configs/yam/yam_bimanual_openpi_policy_xdof_hq_us05_siemens_simple_d405.yaml
```

The session starts paused and begins recording on unpause. Confirm that all
three policy views are correct in Viser at `http://localhost:8080`, that both
arms are at the startup pose, and that the preflight passes before pressing
space to hand control to the policy.

## Safety-critical notes

- Keep `image_preprocess`, all camera resize modes, and `flip_joint_order`
  unchanged as a set. A mismatch changes the model's observation/action
  convention and can produce unsafe commands.
- The model-specific YAML uses the current US05 `xdof_hq` robot configs from
  `karim/zed-openpi-deploy` (gravity factor 1.45 and current joint gains).
- The config is synchronous: one 30-step chunk is consumed before the next
  server inference. Change the inference mode only after separately validating
  chunk-boundary behavior.
