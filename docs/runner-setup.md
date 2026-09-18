# Registering the workstation as a GPU runner

This is the one setup step that cannot be done from a session — it needs shell
access to the machine with the GPU. It is worth doing early, because it is what
turns every GPU question from a day-long round trip into a few minutes.

## Why

`atlas.functional` is proven on CPU, and that covers almost everything. What it
cannot cover is `gsplat.rasterization` — the tiled CUDA kernel that actually
renders. `tests/gpu/test_rasteriser_parity.py` checks that the kernel respects
the property the whole method rests on: that shading and alpha compositing
commute. Until that runs, the CPU proof is a proof about a program the renderer
does not use.

Once the runner is registered, every push runs it automatically and the result
comes back through GitHub — readable by anyone working on the project, including
an agent with no shell on the machine.

## Before you start

- A Linux machine with an NVIDIA GPU, the driver, and a CUDA toolkit new enough
  for `gsplat` to build.
- `python3` with `venv`, and `git`.
- Admin rights on the `atlas-relight` repository.

The runner needs **no inbound network**. It polls GitHub outbound over HTTPS, so
no port forwarding and no exposed SSH.

## Register

GitHub generates a token that expires in an hour, so take the exact commands
from the page rather than from here:

**Settings → Actions → Runners → New self-hosted runner**, Linux, x64.

Then, in a directory you are happy to keep — `~/actions-runner` is conventional:

```bash
mkdir -p ~/actions-runner && cd ~/actions-runner
# curl + tar lines: copy them from the GitHub page, they name the current version
./config.sh --url https://github.com/Cyruskraad/atlas-relight --token <FROM THE PAGE>
```

At the prompts:

| Prompt | Answer |
| --- | --- |
| Runner group | `Default` |
| Name of runner | anything; the hostname is fine |
| **Additional labels** | **`gpu`** — this one matters |
| Work folder | `_work` |

The label is load-bearing. `.github/workflows/gpu.yml` selects
`runs-on: [self-hosted, gpu]`, so a runner without it will never be picked and
the job will queue forever with no error.

## Run it as a service

So it survives a reboot and a closed terminal:

```bash
sudo ./svc.sh install
sudo ./svc.sh start
sudo ./svc.sh status
```

## Check it worked

The runner should appear **Idle** under Settings → Actions → Runners. Then
trigger the workflow by hand — Actions → `gpu` → *Run workflow* — and read the
**Report the hardware** step. It prints `nvidia-smi` and the Python version, and
that output arriving in GitHub *is* the proof that the loop is closed. The
**Report the stack** step follows with the torch and `gsplat` versions and the
device name.

If the job never starts, the label is wrong. If it starts and fails at **CUDA is
actually present**, torch cannot see the GPU — usually a driver/toolkit mismatch,
and `nvidia-smi` in the step above will already have said so.

## What the workflow does to the machine

Deliberately little:

- It creates a `.venv` **inside the job's workspace** and installs there. The
  system Python is never modified.
- It does not run on `pull_request`. A self-hosted runner executes whatever is
  in the branch it checks out, so running it on outside pull requests would hand
  the workstation to whoever opened one. Pushes and manual dispatch only.
- `concurrency: gpu` with `cancel-in-progress: false` — one job at a time, and
  no cancelling mid-run, because an interrupted CUDA job can leave memory held.
- A 60-minute timeout, so nothing occupies the GPU indefinitely.
- Documentation-only commits are skipped via `paths-ignore`.

Each run leaves a workspace under `~/actions-runner/_work/atlas-relight/`. GitHub
does not prune these; delete old ones when disk gets tight.

## Turning it off

```bash
cd ~/actions-runner
sudo ./svc.sh stop && sudo ./svc.sh uninstall
./config.sh remove --token <a fresh token from the same page>
```

Nothing else on the machine is touched.

## If you would rather not

The alternative is running `make check` and `pytest tests/gpu -q` by hand and
pasting the output. That works — it is how the gsplat branches operate, since
Actions is disabled there. It costs a round trip per question, and an agent
handover is exactly where that discipline slips.
