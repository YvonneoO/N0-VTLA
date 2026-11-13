**Dual-arm USB plug (T10), TouchScale full mid-train init** -- `usb_posttrain_tsall20k_realtac_post100traj40k_vision8gpu/40000/`:
all 100 operator-labelled successful teleop trajectories of `SingleBicycle/rss-dual-arm-tactile` `task_usb_plug_dual_20261108`
(57, no pushes) + `task_usb_plug_dual_20261110` (43, teleop pushes on), **real** glove tactile (USB gloves stored straight, not
crossed as in the pour batches), one dataset episode per trajectory (`canonical_wetlab_usb_dual_post100traj_realtac`, asset
`usb_plug_dual_post100traj`, tactile norm fitted on these 100). Same recipe as the task4 post92traj40k runs: bs64 on 8 GPUs,
40000 steps, cosine over 40000. Init = `n0-vtla-base` + Stage-1 mid-train on the full TouchScale human ITW corpus
(`n0vtla_touchscale_midtrain/ts_all_20k/20000`, merged). Push-ramp frames are kept (the post-train loss does not read
`action_mask`). Only the final step (40000, with `optimizer.pt`) is uploaded.
