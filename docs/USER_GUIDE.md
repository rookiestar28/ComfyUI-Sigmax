# ComfyUI-Sigmax 1.0 User Guide

ComfyUI-Sigmax provides versioned, model-aware sigma schedules for Krea 2 Turbo and Krea 2 RAW.
It keeps schedule construction separate from the model and numerical sampler, exposes every
material transform, and fails closed when a model variant cannot be resolved safely.

This guide covers package version `1.0.0`. The supported package floor is Python 3.10 and
ComfyUI 0.29.0. The package has no mandatory third-party runtime dependencies.

## Installation

Review custom nodes before installation: they execute Python code inside ComfyUI. The official
[ComfyUI custom-node guide](https://docs.comfy.org/installation/install_custom_node) recommends
Manager for normal lifecycle management and Git or a reviewed ZIP for manual installation.

### ComfyUI Manager

Use [ComfyUI Manager](https://docs.comfy.org/manager/overview) only when `ComfyUI-Sigmax` is visible
in its Registry search. Absence from search means the package has not been published there yet; use
the Git method below. Install the exact stable version you intend to use, restart ComfyUI, and
review the startup log.

### Git installation

Stop ComfyUI, open a terminal in its `custom_nodes` directory, and clone into one direct child
directory:

```bash
git clone https://github.com/rookiestar28/ComfyUI-Sigmax comfyui-sigmax
```

The expected layout is `ComfyUI/custom_nodes/comfyui-sigmax/__init__.py`. Avoid a doubled wrapper
such as `custom_nodes/ComfyUI-Sigmax-main/ComfyUI-Sigmax-main`.

Sigmax 1.0.0 declares zero mandatory runtime dependencies. Do not install optional development,
plotting, or reference-framework packages into ComfyUI merely to use the nodes. Restart ComfyUI
after installation.

### Reviewed ZIP installation

Use only a release ZIP whose checksum and provenance you have reviewed. Extract it into one direct
child directory under `ComfyUI/custom_nodes`, confirm that `__init__.py` is at that child root, and
restart ComfyUI. A source-repository download can contain development material and is not
equivalent to the validated Registry artifact.

## Verify the installation

Check the ComfyUI startup log for an import failure, duplicate ID, or registry-collision message.
Then search the node menu for the eight built-in IDs:

- `Sigmax.AdvancedFlowMatchScheduler`
- `Sigmax.Krea2SigmaScheduler`
- `Sigmax.ModelAwareSigmaScheduler`
- `Sigmax.ProfileInspector`
- `Sigmax.RawWorkflowOutput`
- `Sigmax.ScheduleComparison`
- `Sigmax.ScheduleInspector`
- `Sigmax.TurboWorkflowOutput`

The installation directory may be renamed; the IDs stay namespaced and stable. Presence proves
registration, not model compatibility or image quality.

## Choose a node

| Need | Node | Boundary |
| --- | --- | --- |
| Build an explicit Krea 2 Turbo or RAW schedule | `Sigmax.Krea2SigmaScheduler` | Produces `SIGMAS`; it is not a sampler |
| Resolve a connected Krea 2 model conservatively | `Sigmax.ModelAwareSigmaScheduler` | Auto rejects family-only RAW/Turbo ambiguity |
| Construct experimental unit-flow schedules | `Sigmax.AdvancedFlowMatchScheduler` | No generic model-compatibility claim |
| Inspect an exact profile | `Sigmax.ProfileInspector` | Read-only |
| Inspect or compare verified schedules | `Sigmax.ScheduleInspector`, `Sigmax.ScheduleComparison` | No conversion or resampling |
| Emit canonical model-free workflow evidence | `Sigmax.TurboWorkflowOutput`, `Sigmax.RawWorkflowOutput` | Does not claim model or sampler execution |

Prefer `Sigmax.Krea2SigmaScheduler` when you know the exact variant. Use
`Sigmax.ModelAwareSigmaScheduler` only when its trusted evidence can resolve the exact variant;
select Turbo or RAW explicitly when Auto reports ambiguity.

## Krea 2 Turbo example

For the official control recipe:

1. Add `Sigmax.Krea2SigmaScheduler` and select `Turbo` explicitly.
2. Set 8 steps and the intended width/height; keep the full terminal-inclusive range.
3. Connect its external `SIGMAS` output directly to a sampling path that accepts supplied sigmas.
4. Use deterministic Euler sampling and CFG 1.0. The native comparison baseline is
   **Euler + Simple**, 8 steps, CFG 1.0.
5. Do not run the supplied sigmas through another scheduler or apply another model time shift.

The official profile applies the fixed exponential shift once with `mu = 1.15` and appends
terminal zero. A different step count is allowed only as visibly `modified` evidence; it is not
the official Turbo recipe.

## Krea 2 RAW example

RAW schedule construction depends on image geometry:

1. Add `Sigmax.Krea2SigmaScheduler` and select `RAW` explicitly.
2. Enter the actual requested width and height. Effective dimensions round upward to a multiple of
   16, and the schedule records both requested and effective geometry.
3. Choose one named recipe: 52 steps with ComfyUI CFG 4.5 for the official full recipe, or 28 steps
   with ComfyUI CFG 5.5 for the Diffusers framework-reference recipe.
4. Connect the external `SIGMAS` output directly to the sampling path; do not add a second shift.
5. Inspect `schedule_info` when debugging geometry, sequence length, dynamic `mu`, evidence, or
   slicing.

The shared Krea 2 model class, filenames, and common tensor keys identify only the family. They do
not safely distinguish RAW from Turbo. Auto must reject when exact variant evidence is absent.

## Workflow metadata

Canonical workflow metadata lives under `extra.comfyui_sigmax` and binds package, node, host, and
profile versions plus deterministic fingerprints. Preserve that namespace when saving or editing a
workflow. Package version `1.0.0` does not replace node schema version `1` or profile version `1`;
they are independent compatibility axes.

The four packaged fixtures validate model-free Turbo and RAW schedule/artifact paths on the pinned
ComfyUI 0.29.0 host baseline. They do not bundle model weights, run a model, or prove image output.
See the [workflow metadata](WORKFLOW_METADATA_SPEC.md) and
[workflow validation](WORKFLOW_VALIDATION_SPEC.md) specifications.

## Update, upgrade, and rollback

Before changing versions, save exported workflows and record the current Sigmax revision or
Manager version. If Manager is available, create a Manager snapshot.

For a Git installation, fetch reviewed versions without modifying workflow files:

```bash
cd ComfyUI/custom_nodes/comfyui-sigmax
git fetch --tags
git switch --detach <reviewed-tag-or-commit>
```

Restart ComfyUI and verify all eight IDs and important workflows. Do not use an unreviewed moving
branch as a production rollback point. To roll back, select the previously recorded tag/commit or
restore the Manager snapshot, restart, and revalidate. Keep workflow backups until node IDs,
schemas, schedule fingerprints, and results are confirmed.

Removing Sigmax is directory-scoped: stop ComfyUI, move the `comfyui-sigmax` child directory to a
backup location outside `custom_nodes`, restart, and confirm that only Sigmax nodes are missing.

## Troubleshooting

| Symptom | Check | Resolution |
| --- | --- | --- |
| Import failed | Folder nesting and Python floor | Put one package root directly under `custom_nodes`; use ComfyUI's Python 3.10+ environment |
| Nodes absent | Startup log and disabled state | Restart ComfyUI; enable the node in Manager; confirm all eight IDs |
| Duplicate/collision error | Multiple Sigmax copies | Keep one reviewed installation; remove or disable stale duplicates, then restart |
| Auto rejects the model | Evidence resolves only Krea family | Select Turbo or RAW explicitly; do not rely on filename or shared class |
| Schedule or result looks shifted twice | Model/scheduler ownership | Supply Sigmax sigmas once; remove any second scheduler or model time shift |
| RAW geometry differs | Requested/effective dimensions | Inspect the ceil-to-16 dimensions, sequence length, and dynamic `mu` in `schedule_info` |
| Workflow metadata fails | Package/node/host/profile drift | Preserve `extra.comfyui_sigmax`; migrate intentionally instead of editing fingerprints |
| Optional plot unavailable | Plotting extra absent | Treat plots as optional presentation; reports and numerical evidence remain available |
| New ComfyUI release behaves differently | Host outside the pinned support baseline | Reproduce on the supported baseline and consult [Compatibility](COMPATIBILITY.md) |

Never solve an error by silently changing variants, using dummy sigmas, disabling validation, or
applying a second shift.

## Migrate to 1.0.0

Development builds used package identity `0.1.0.dev0`; version 1.0.0 is the stable public-contract
baseline. Before migration, back up workflows and note their package, node, profile, and host
versions. Install the reviewed 1.0.0 source, restart, and resave only after validation succeeds.

The eight node IDs remain stable. Workflow package versions change to `1.0.0`, while current node
schema and Krea profile versions remain `1`. A future breaking node ID, schema, artifact, or
schedule-semantics change requires a new project major, migration note, and compatibility review;
do not hand-edit fingerprints to bypass that contract.

## Security and known limitations

- Custom nodes execute code inside ComfyUI. Install reviewed sources and verify checksums,
  provenance, and licenses.
- Model weights are not included. Obtain them separately under their own license and access terms.
- The Registry artifact validator is model-free and non-publishing. Name availability does not
  prove publisher ownership, authenticity, or safety.
- Sigmax 1.0.0 supports verified Krea 2 Turbo and RAW model profiles. The schedule-only
  `flowmatch.generic.fixed` (`framework_reference`) and `flowmatch.generic.dynamic`
  (`experimental`) declarations require explicit selection and are never official model
  profiles. They do not establish model-specific compatibility, guidance, step recommendations,
  or quality, and are never silently applied to another model family.
- The advanced FlowMatch node is experimental. It exposes schedule math, not arbitrary-model
  compatibility.
- Native deterministic Euler equivalence is proven on the pinned host; stochastic, resumable,
  partial-denoise, advanced workflow, real-weight GPU, and image-quality claims remain unsupported
  or unvalidated.
- macOS and native hosted Ubuntu evidence are not available. Latest-host rows are informational and
  do not expand the pinned support baseline.
- Image quality is not numerical correctness evidence.

For exact environment lanes and evidence limits, see [Compatibility](COMPATIBILITY.md). For stable
wire contracts and breaking-change rules, see [Stable public contracts](STABLE_PUBLIC_CONTRACTS.md).
The official [Registry overview](https://docs.comfy.org/registry/overview) explains Registry-backed
discovery, semantic versions, and workflow version recording.
