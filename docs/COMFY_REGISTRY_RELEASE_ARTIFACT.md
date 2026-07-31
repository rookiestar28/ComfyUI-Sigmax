# Comfy Registry Release Artifact

## Status

- **Manifest schema:** `sigmax.registry-release-manifest/1`
- **Manifest envelope:** `sigmax.registry-release-manifest-envelope/1`
- **Report schema:** `sigmax.registry-artifact-report/1`
- **Command:** `scripts/validate_registry_artifact.py`
- **Publishing:** never performed by this command

The Comfy Registry distributes a custom node from a ZIP assembled from Git-tracked files after
applying `.comfyignore`. This boundary is separate from the Python wheel and source distribution.
The validator reproduces that selection locally, builds a deterministic ZIP, audits every member,
and proves that the archive imports when installed under an arbitrary custom-node directory name.

The implementation follows the official
[publishing workflow](https://docs.comfy.org/registry/publishing) and
[node specification](https://docs.comfy.org/registry/specifications). Its file-selection contract
is pinned to the reviewed official `comfy-cli` release and source revision recorded in the
embedded manifest.

## Canonical command

Stage the intended release files first, then run:

```powershell
python scripts/validate_registry_artifact.py `
  --archive .tmp/comfy-registry/comfyui-sigmax-1.0.0.zip `
  --check-manifest `
  --observe-registry `
  --output .tmp/comfy-registry/report.json
```

The archive is built from Git index blobs, so unstaged working-tree edits cannot silently alter
the candidate. Fixed member order, timestamps, permissions, compression settings, and canonical
JSON make equal inputs byte-for-byte reproducible across supported Windows and WSL environments.

## Identity and manifest binding

Version `1.0.0` is the first Registry-ready semantic version. The embedded manifest freezes:

- the `comfyui-sigmax` package, `rookiestar28` publisher, and `ComfyUI-Sigmax` display identity;
- Python and ComfyUI compatibility requirements plus license identity;
- the frozen public-contract fingerprint and all twelve built-in node ID/schema pairs;
- the four canonical workflow package, node, host, profile, and workflow fingerprints;
- exact hashes for release-defining source files;
- the reviewed official `comfy-cli` release and source revision.

Any drift requires regenerating and reviewing the manifest. A source hash mismatch, stale
manifest, missing required member, or unexpected top-level entry fails closed.

## Archive safety boundary

`.comfyignore` excludes development automation, tests, scripts, public project documentation,
requirements inputs, and repository tooling that are not needed at runtime. The validator also
rejects absolute, drive-relative, traversal, backslash, duplicate, linked, oversized, secret-like,
model-weight, cache, environment, and undeclared member paths before extraction.

The normalized installation probe extracts only an already-audited archive into a newly named
directory. A fresh isolated interpreter imports its root bootstrap and verifies version `1.0.0`,
the exact twelve node IDs, empty runtime dependency metadata, and the absence of eager Torch,
Diffusers, NumPy, and aiohttp imports.

## Registry observation and publication boundary

`--observe-registry` performs three unauthenticated `GET` requests: node lookup, publisher lookup,
and publisher-name availability. It records HTTP status and canonical response fingerprints only.
It does not prove publisher ownership, reserve an identity, authenticate, upload, or publish.

Every report records `publication_performed: false`. The validator accepts no token, API-key,
login, upload, or publish option. Actual publication remains a separate human-authorized release
operation after all blocking release gates pass.

## Limitations

The report proves the reviewed source-to-ZIP contract, archive hygiene, reproducibility, and a
model-free import boundary. It is not a malware scan, package signature, ownership proof, live
ComfyUI workflow run, model-quality claim, or authorization to publish.
