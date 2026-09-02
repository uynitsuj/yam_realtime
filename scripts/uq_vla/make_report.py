"""Assemble the self-contained VFD report artifact (figures + video inlined)."""
from __future__ import annotations

import argparse
import base64
import json
import pathlib

FONTS = ("https://fonts.googleapis.com/css2?"
         "family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&"
         "family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&"
         "family=IBM+Plex+Mono:wght@400;500&display=swap")


def b64(path: pathlib.Path, mime: str) -> str:
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figs", type=pathlib.Path, required=True)
    ap.add_argument("--summary", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--video", default="overlay_episode_164557_c3c5323a.mp4")
    a = ap.parse_args()

    s = json.loads(a.summary.read_text())
    v, em = s["vfd"], s["episode_mean"]
    sp = s["spearman_vs_vfd"]
    img = {p.stem: b64(p, "image/png") for p in sorted(a.figs.glob("*.png"))}
    vid = a.figs / a.video
    video_uri = b64(vid, "video/mp4") if vid.exists() else None
    joints = ", ".join(f"{j['joint']} ({j['share_pct']:.0f}%)" for j in s["top_joints"])

    def plate(key: str, label: str, question: str, caption: str) -> str:
        return f"""
    <figure class="plate-wrap">
      <figcaption class="plate-head">
        <span class="tag">{label}</span>
        <span class="q">{question}</span>
      </figcaption>
      <div class="plate"><img src="{img[key]}" alt="{question}"></div>
      <p class="cap">{caption}</p>
    </figure>"""

    stat = lambda val, lab: f'<div class="stat"><div class="v">{val}</div><div class="l">{lab}</div></div>'

    html = f"""<title>Velocity-Field Disagreement on YAM Bottles</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="{FONTS}">
<style>
  :root {{
    --ground:#f7f8fa; --panel:#ffffff; --plate:#fcfcfb;
    --ink:#14181f; --ink2:#4a5261; --ink3:#8b93a1;
    --rule:#dde1e8; --accent:#2a78d6; --accent-soft:#eaf2fd; --warn:#c8501f;
    --shadow:0 1px 2px rgba(20,24,31,.05), 0 8px 24px -12px rgba(20,24,31,.12);
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      --ground:#10141a; --panel:#171c24;
      --ink:#eef1f6; --ink2:#a8b1c0; --ink3:#727c8c;
      --rule:#262d38; --accent:#4f9bf0; --accent-soft:#182635; --warn:#e87a4a;
      --shadow:0 1px 2px rgba(0,0,0,.4), 0 8px 24px -12px rgba(0,0,0,.6);
    }}
  }}
  :root[data-theme="dark"] {{
    --ground:#10141a; --panel:#171c24;
    --ink:#eef1f6; --ink2:#a8b1c0; --ink3:#727c8c;
    --rule:#262d38; --accent:#4f9bf0; --accent-soft:#182635; --warn:#e87a4a;
    --shadow:0 1px 2px rgba(0,0,0,.4), 0 8px 24px -12px rgba(0,0,0,.6);
  }}
  * {{ box-sizing:border-box; }}
  body {{
    background:var(--ground); color:var(--ink);
    font-family:"Source Serif 4",Georgia,serif; font-size:17px; line-height:1.62;
    margin:0; padding:0 24px 96px;
    -webkit-font-smoothing:antialiased;
  }}
  .wrap {{ max-width:1080px; margin:0 auto; }}
  .col {{ max-width:63ch; }}
  h1,h2,h3,.tag,.stat .v,.eyebrow {{ font-family:"Bricolage Grotesque","Helvetica Neue",sans-serif; }}
  h1 {{ font-size:clamp(2.1rem,4.6vw,3.15rem); line-height:1.05; font-weight:700;
       letter-spacing:-.02em; margin:0 0 18px; text-wrap:balance; }}
  h2 {{ font-size:1.45rem; font-weight:700; letter-spacing:-.01em; margin:64px 0 14px;
       text-wrap:balance; padding-top:22px; border-top:1px solid var(--rule); }}
  h3 {{ font-size:1.05rem; font-weight:700; margin:34px 0 8px; }}
  p {{ margin:0 0 16px; }}
  a {{ color:var(--accent); }}
  .eyebrow {{ font-size:.72rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase;
             color:var(--accent); margin:52px 0 18px; }}
  .lede {{ font-size:1.18rem; color:var(--ink2); }}
  code,.mono,.num {{ font-family:"IBM Plex Mono",ui-monospace,monospace; }}
  code {{ font-size:.86em; background:var(--accent-soft); color:var(--ink);
         padding:.1em .34em; border-radius:3px; }}
  .stats {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(132px,1fr));
           gap:1px; background:var(--rule); border:1px solid var(--rule);
           border-radius:8px; overflow:hidden; margin:30px 0 8px; }}
  .stat {{ background:var(--panel); padding:15px 16px 13px; }}
  .stat .v {{ font-size:1.5rem; font-weight:700; letter-spacing:-.02em;
             font-variant-numeric:tabular-nums; line-height:1.1; }}
  .stat .l {{ font-family:"IBM Plex Mono",monospace; font-size:.66rem; line-height:1.35;
             text-transform:uppercase; letter-spacing:.07em; color:var(--ink3); margin-top:6px; }}
  .eq {{ background:var(--panel); border:1px solid var(--rule); border-left:3px solid var(--accent);
        border-radius:6px; padding:18px 20px; margin:22px 0; overflow-x:auto;
        font-family:"IBM Plex Mono",monospace; font-size:.83rem; line-height:1.85;
        white-space:pre; color:var(--ink); }}
  .note {{ background:var(--panel); border:1px solid var(--rule); border-radius:8px;
          padding:20px 22px; margin:26px 0; box-shadow:var(--shadow); }}
  .note.caveat {{ border-left:3px solid var(--warn); }}
  .note h3 {{ margin-top:0; }}
  .note p:last-child {{ margin-bottom:0; }}
  .plate-wrap {{ margin:44px 0 8px; }}
  .plate-head {{ display:flex; gap:12px; align-items:baseline; margin-bottom:10px; flex-wrap:wrap; }}
  .tag {{ font-size:.66rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase;
         color:var(--accent); background:var(--accent-soft); padding:4px 8px; border-radius:4px;
         white-space:nowrap; }}
  .q {{ font-size:1.02rem; font-weight:600; color:var(--ink); }}
  .plate {{ background:var(--plate); border:1px solid var(--rule); border-radius:8px;
           padding:12px; overflow-x:auto; box-shadow:var(--shadow); }}
  .plate img, .plate video {{ display:block; width:100%; height:auto; border-radius:2px; }}
  .cap {{ font-family:"IBM Plex Mono",monospace; font-size:.73rem; line-height:1.6;
         color:var(--ink3); margin:11px 0 0; max-width:80ch; }}
  table {{ border-collapse:collapse; width:100%; margin:22px 0; font-size:.92rem; }}
  th,td {{ text-align:left; padding:9px 12px; border-bottom:1px solid var(--rule); }}
  th {{ font-family:"IBM Plex Mono",monospace; font-size:.68rem; text-transform:uppercase;
       letter-spacing:.08em; color:var(--ink3); font-weight:500; }}
  td.n {{ font-family:"IBM Plex Mono",monospace; font-variant-numeric:tabular-nums; }}
  ul {{ padding-left:1.15em; }} li {{ margin-bottom:9px; }}
  .foot {{ margin-top:72px; padding-top:22px; border-top:1px solid var(--rule);
          font-family:"IBM Plex Mono",monospace; font-size:.72rem; color:var(--ink3);
          line-height:1.85; }}
  :focus-visible {{ outline:2px solid var(--accent); outline-offset:2px; }}
  @media (prefers-reduced-motion:reduce) {{ * {{ animation:none!important; transition:none!important; }} }}
</style>

<div class="wrap">
  <div class="col">
    <p class="eyebrow">Offline uncertainty analysis</p>
    <h1>How unsure is pi0 about putting bottles in the bin?</h1>
    <p class="lede">Velocity-Field Disagreement, computed frame by frame over
    {s['episodes']} recorded rollouts of the <span class="mono">sss45</span> checkpoint,
    replaying exactly the observations the policy saw on the robot.</p>
  </div>

  <div class="stats">
    {stat(s['episodes'], 'rollouts')}
    {stat(f"{s['frames']:,}", 'frames scored')}
    {stat(f"{v['median']:.0f}", 'median VFD')}
    {stat(f"{v['p90']:.0f}", '90th pct')}
    {stat(f"{v['max']:.0f}", 'peak VFD')}
    {stat(f"{v['ratio_p99_median']:.1f}&times;", 'p99 / median')}
    {stat(f"{em['spread_ratio']:.1f}&times;", 'most / least unsure rollout')}
  </div>
  <p class="cap">VFD is an unnormalised squared-velocity quantity in the model's normalised
  action space &mdash; comparable across frames and rollouts of this ensemble, not across
  different ensembles or policies.</p>

  <div class="col">
    <h2>What is actually being measured</h2>
    <p>A flow-matching policy does not emit a distribution you can read off. It emits a
    <em>velocity field</em> that transports Gaussian noise into an action chunk along an ODE.
    VFD asks a question about that field rather than about its samples: given two models
    trained on the same data, how much do their velocity fields <em>disagree at the same
    point</em> along the generative path?</p>

    <div class="eq">u(y) = 1/(M(M-1)N&#8339;) &middot; E[ &Sigma;&#7522;&#8800;&#11388; &Sigma;&#8467; &kappa;&#8347; &middot; &#8214; v&#7522;(x&#7522;&#8347;, y) &minus; v&#11388;(x&#7522;&#8347;, y) &#8214;&sup2; ]

&kappa;&#8347; = s/(1&minus;s)     s = &#8467;&middot;&delta;s,  &#8467; = 0 &hellip; N&#8339;&minus;1     x&#7522;&#8347;&#8330;&#948;&#8347; = x&#7522;&#8347; + v&#7522;(x&#7522;&#8347;, y)&middot;&delta;s</div>

    <p>Two details carry all the meaning. Both members are evaluated at
    <em>the same state</em> <span class="mono">x&#7522;&#8347;</span> &mdash; so the metric is blind to which
    noise sample produced that state. And <span class="mono">&kappa;&#8347;</span> grows without bound toward
    the data end of the path, so disagreement late in denoising &mdash; where the chunk is nearly
    committed &mdash; dominates the score.</p>

    <div class="note">
      <h3>Why noise seeds can't stand in for an ensemble</h3>
      <p>A single checkpoint sampled with different seeds gives VFD <strong>exactly zero</strong>.
      The velocity field is a deterministic function; with
      <span class="mono">&theta;&#7522; = &theta;&#11388;</span> every term in the sum cancels regardless of the seed.
      Seeds change <em>where</em> you probe the field, never the disagreement between fields.</p>
      <p>Measured on this data: with the same checkpoint twice, VFD = <span class="num">0</span>
      while that checkpoint's own sampled chunks still sit <span class="num">1.28 rad</span> apart.
      Swapping in a second checkpoint moves VFD to <span class="num">153</span> and leaves the
      sample spread untouched at <span class="num">1.28 rad</span>. Sampling spread is real; it is
      simply a different quantity (aleatoric, not epistemic).</p>
    </div>

    <h3>The ensemble used here</h3>
    <p>The paper builds members by fine-tuning one base VLA <span class="mono">M</span> times on
    reshuffled data. No such pair exists for this checkpoint, so the closest available stand-in
    is used: <span class="mono">sss45</span> and <span class="mono">sss30</span> at step 59999 &mdash;
    two independent WARP-BC fine-tunes of the <em>same</em> base on the <em>same</em> episodes,
    differing only in the reward-model stride used to reweight frames. Both were verified to
    carry byte-identical action normalisation statistics, so their velocity fields live in the
    same coordinates; the pipeline hard-fails if they don't.</p>
  </div>

  {plate('fig1_timelines', 'Fig 1', 'Where in each rollout is the policy unsure?',
         'One panel per rollout, VFD at 3 Hz. Uncertainty is spiky rather than slowly varying: '
         'brief excursions well past the pooled 90th percentile, separated by long confident stretches. '
         'Several rollouts collapse to near-zero at the end, where the arms have stopped moving.')}

  {plate('fig2_ranking', 'Fig 2', 'Which rollouts were hardest?',
         f"Mean and peak VFD per rollout. Mean uncertainty spans {em['spread_ratio']:.1f}x from "
         f"{em['least_uncertain'].replace('episode_','')} to {em['most_uncertain'].replace('episode_','')}. "
         'A long bar is a rollout that was mostly confident but spiked hard somewhere.')}

  {plate('fig3_heatmap', 'Fig 3', 'Is there a shared hard phase across rollouts?',
         'Every rollout as one row, time normalised. Largely not: the dark cells scatter rather than '
         'aligning into a vertical band, so on this task the uncertain moments are situation-specific '
         'rather than tied to a fixed phase of the task.')}

  {plate('fig4_flowtime', 'Fig 4', 'Where along the ODE does the disagreement live?',
         f"{s['late_ode_share_mean']*100:.0f}% of the score accumulates in the second half of the "
         'integration. Partly a real effect (the members agree while the chunk is still mostly noise) '
         'and partly by construction: the kappa weight on the right panel is the multiplier being applied.')}

  {plate('fig5_joints', 'Fig 5', 'Which joints and which part of the chunk?',
         f"Disagreement concentrates on the left arm's distal joints &mdash; {joints}. "
         'Within a chunk it rises monotonically with lookahead: the members agree on the next few '
         'commands and diverge about where the arm should be a second from now.')}

  {plate('fig6_baselines', 'Fig 6', 'Could a single checkpoint have told us this?',
         f"Largely no. The two signals one checkpoint can compute track epistemic VFD only weakly "
         f"(self-dispersion rho = {sp['self_dispersion']:.2f}, STAC rho = {sp['stac']:.2f}), while "
         f"ensemble-based Action-L2 does track it (rho = {sp['action_l2']:.2f}). This is the "
         'quantitative form of the seed argument above.')}

  {plate('fig7_chunk_fan', 'Fig 7', 'What does a high-VFD moment look like in action space?',
         'Every sampled chunk from both members at the least and most uncertain frames. At low VFD the '
         'two fans lie on top of each other. At high VFD each member is internally tight but they pull '
         'toward visibly different futures &mdash; disagreement about intent, not sampling jitter.')}

  {'''
  <figure class="plate-wrap">
    <figcaption class="plate-head">
      <span class="tag">Video</span>
      <span class="q">The signal against what the cameras saw</span>
    </figcaption>
    <div class="plate"><video controls preload="metadata" src="''' + video_uri + '''"></video></div>
    <p class="cap">Rollout 164557_c3c5323a at true rate (3 Hz, the eval stride). Top / left-wrist /
    right-wrist as the policy received them, with the live VFD trace beneath; the marker turns orange
    above the pooled 90th percentile.</p>
  </figure>''' if video_uri else ''}

  <div class="col">
    <h2>What holds up</h2>
    <ul>
      <li><strong>The implementation is validated by construction.</strong> Identical weights give
      exactly zero; the one term the algorithm forces to vanish
      (<span class="mono">&kappa;&#8320; = 0</span>) does vanish.</li>
      <li><strong>The signal is not flat.</strong> Frame-level VFD spans
      <span class="num">{v['min']:.0f}</span>&ndash;<span class="num">{v['max']:.0f}</span>, with the
      99th percentile <span class="num">{v['ratio_p99_median']:.1f}&times;</span> the median &mdash; enough
      dynamic range for a threshold to be meaningful.</li>
      <li><strong>It carries information the single-model signals don't</strong> (Fig 6), which is the
      case for paying the cost of a second checkpoint.</li>
      <li><strong>It is spatially interpretable</strong> &mdash; attributable to specific joints and to
      specific lookahead within the chunk (Fig 5).</li>
    </ul>

    <h2>What this analysis cannot claim</h2>
    <div class="note caveat">
      <p><strong>No success labels, so no calibration and no validation.</strong> The paper's headline
      results &mdash; Spearman 0.71 against task success, and a conformal failure detector calibrated
      from 10 successful rollouts &mdash; both need per-rollout outcomes. None exist for these
      {s['episodes']} rollouts, so every threshold drawn here is a percentile of the observed
      distribution, not a calibrated detector. High VFD is <em>not</em> shown to predict failure on
      this task; it is only shown to vary, and to vary in an interpretable way.</p>
      <p><strong>The ensemble is a stand-in.</strong> sss45 and sss30 differ in data curation, not in
      shuffle seed. They are two genuinely distinct fine-tunes of one base, which is the right shape,
      but their disagreement includes a curation component the paper's construction wouldn't have.
      A same-recipe, different-seed pair would be the clean version.</p>
      <p><strong>Reconstruction is faithful but not bit-exact.</strong> Observations are rebuilt from
      the recorded MP4s through the same crop-and-resize chain the camera nodes applied, so they differ
      from the live bus frames by video compression. Camera-to-tick alignment is within 62 ms worst case.</p>
    </div>

    <h2>Reproducing it</h2>
    <table>
      <tr><th>Stage</th><th>Command</th></tr>
      <tr><td>Rebuild observations</td>
          <td class="n">extract_obs.py &lt;session&gt; --out-dir obs --stride 10</td></tr>
      <tr><td>Score VFD</td>
          <td class="n">compute_vfd.py --obs-dir obs --out-dir uq --members sss45,sss30</td></tr>
      <tr><td>Figures + video</td>
          <td class="n">plot_uq.py --uq-dir uq --out-dir figs --video-episode max</td></tr>
    </table>
    <p class="cap">Stage 2 runs in the openpi venv (<span class="mono">uv run --project
    /home/us07/openpi</span>); the others in robots_realtime. Per-episode outputs are cached, so an
    interrupted sweep resumes by re-running the same command.</p>

    <div class="foot">
      Method: Roemer et al., <em>Uncertainty Quantification for Flow-Based Vision-Language-Action
      Models</em>, arXiv:2606.18043 &mdash; Algorithm 2 / Eq. 7.<br>
      Policy: pi0, action horizon 30, N&#8339; = {s['num_steps']} Euler steps, B = 5 noise samples per member,
      M = {len(s['members'])} members ({' + '.join(s['members'])}).<br>
      {s['frames']:,} frames scored at 136 ms/frame on one RTX 5090.
    </div>
  </div>
</div>"""

    a.out.write_text(html)
    mb = len(html.encode()) / 1e6
    print(f"wrote {a.out} ({mb:.1f} MB)")
    if mb > 15:
        print("WARNING: approaching the 16 MB artifact limit")


if __name__ == "__main__":
    main()
