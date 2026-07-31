# Security and Release Audit

## Status

- **Audit schema:** `sigmax.release-audit/1`
- **Envelope schema:** `sigmax.release-audit-envelope/1`
- **Command:** `scripts/run_release_audit.py`
- **Publishing:** forbidden; this audit is local and read-only with respect to external systems

The M8-02 audit keeps five questions separate: tracked-file hygiene, dependencies,
software-source/framework/model-weight provenance and licenses, Comfy Registry metadata, and generated
archive contents. A pass in one section cannot satisfy another.

Example Windows invocation using fresh repository-local ignored paths:

```powershell
python scripts/run_release_audit.py `
  --dist-dir .tmp/release-audit-dist `
  --output .tmp/release-audit.json
```

The script builds one wheel and one sdist through the selected project-local interpreter, runs
the repository's detect-secrets hook, reads the Git tracked-file inventory, and writes canonical
JSON. It prints `RELEASE_AUDIT=PASS` only when every section passes. The fingerprint covers the
semantic member-content inventory rather than ZIP/TAR timestamps; byte-for-byte release
reproducibility remains a later release gate.

## Tracked-file boundary

Tracked public material may include package code/data, scripts, tests, CI configuration, and
public documentation. The audit rejects private governance and research roots, temporary/local
environments, build outputs, caches, logs, sensitive filenames, and model-weight suffixes. The
ignore policy remains the hard deny-list, and the detect-secrets all-files hook is a separate
mandatory section.

## Dependency classes

The report inventories these independently:

| Class | Current policy |
| --- | --- |
| Mandatory runtime | Must remain empty unless separately approved |
| Optional runtime | `plot` and `reference`, version-bounded and non-default |
| Development | Version-bounded local validation tools |
| Build | Version-bounded PEP 517 backend requirements |

Direct URL, local-file, or unbounded requirements fail the audit. A reviewed optional or
development dependency never becomes a mandatory runtime dependency implicitly.

## Provenance and license layers

Every built-in profile is audited from its immutable schema projection. Each profile must retain
non-empty, independently identified layers for:

- official inference software source and its license;
- corroborating frameworks and each framework license;
- model weights, immutable revision and SHA-256, and the weight license.

Resource identities cannot be reused across layers. URLs must be public HTTPS locators and source
revisions must remain pinned. This is an inventory and consistency control, not legal advice or a
claim that third-party terms permit every use or redistribution scenario.

## Registry metadata

`[tool.comfy]` is reviewed separately from `[project]`. The audit checks the frozen publisher and
display identities plus the package version, Python requirement, and ComfyUI requirement. The
report always records `publish_performed: false`; M8-02 neither authenticates to nor publishes to
the Registry. Registry artifact dry validation is owned by M8-06.

## Archive boundaries

Both archive formats reject absolute/drive-relative/traversal paths, backslash paths, duplicate
members, links, oversized payloads, model weights, sensitive filenames, internal roots, tests,
scripts, and undeclared top-level content. Archives are inspected without extraction.

The wheel may contain only:

- `comfyui_sigmax/` package code and declared JSON/type data;
- its generated `.dist-info/` metadata and license directory.

The sdist may contain only the package tree, generated `.egg-info/`, and the reviewed root files
`LICENSE.TXT`, `MANIFEST.in`, `NOTICE`, `PKG-INFO`, `README.md`, `pyproject.toml`, and generated
`setup.cfg`. `MANIFEST.in` explicitly prunes internal records, references, tests, scripts, docs,
environments, and caches. Both formats must contain the package entry point, frozen public
contract manifest, license, and notice; the sdist must also contain README and pyproject metadata.

## Stable findings

Findings use these namespaces:

- `tracked.*` for tracked private/sensitive/model material;
- `dependency.*` for class, bound, and source violations;
- `provenance.*` for missing, aliased, or invalid resource/license layers;
- `registry.*` for identity, version, or requirement drift;
- `archive.*` for unsafe paths, links, size, required-file, and boundary violations;
- `secret_scan.failed` when the independent detect-secrets stage does not pass.

The report is evidence of the inspected source tree and built archives. It is not a signature,
malware scan, vulnerability database query, authenticity proof, or authorization to publish.
