# Schedule Report and Optional Plot Specification

## Status

- **Schedule report projection:** `sigmax.schedule-report/1`
- **Schedule report envelope:** `sigmax.schedule-report-envelope/1`
- **Comparison projection:** `sigmax.schedule-comparison-report/1`
- **Comparison envelope:** `sigmax.schedule-comparison-report-envelope/1`
- **Plot validation:** `sigmax.schedule-plot-validation/1`
- **Maturity:** implemented and normative for pre-alpha v1; not yet a stable public API

This specification defines read-only views over validated immutable construction artifacts and
execution receipts. A report cannot create, repair, or upgrade execution evidence. A plot is a
presentation derived from a report and is never numerical acceptance evidence.

## 1. Inputs and trust boundary

`build_schedule_report()` accepts:

1. a validated immutable `ScheduleArtifact`; and
2. optionally, a validated `ExecutionReceipt` whose construction fingerprint, numerical
   fingerprint, and effective inputs exactly match the artifact.

`build_schedule_report_from_bundle()` accepts a validated `PortableExecutionBundle`, whose
constructor has already enforced the same cross-links.

Arbitrary dictionaries and file paths are not builder inputs. Untrusted serialized reports must
go through the strict deserializers.

## 2. Schedule report projection

The schedule report has these exact top-level members:

| Member | Meaning |
| --- | --- |
| `schema` | Exact value `sigmax.schedule-report/1` |
| `artifact` | Construction and numerical fingerprints |
| `domain` | Lowercase artifact sigma domain |
| `precision` | `float32` or `float64` |
| `source` | Bounded artifact source identity |
| `evidence` | Artifact evidence level/reference |
| `effective_inputs` | Exact artifact effective inputs |
| `construction` | Base grid, ordered transforms, terminal policy, and slicing |
| `receipt_present` | Whether execution evidence is attached |
| `execution` | Receipt fingerprint, status/reason/counts, compatibility, host/component identities, and RNG ownership, or `null` |
| `samples` | Terminal-inclusive sigma rows with signed adjacent deltas |

Every sigma is the exact typed IEEE-754 token from the numerical artifact. A non-terminal row
defines:

```text
delta_to_next = next_sigma - sigma
```

The last terminal-inclusive row has `delta_to_next = null`. Deltas are quantized to the
artifact precision and encoded as typed IEEE-754 tokens.

A construction-only report sets `receipt_present=false` and `execution=null`. It does not imply
that a model or sampler executed.

## 3. Comparison report

`build_schedule_comparison_report()` consumes two validated `ScheduleReport` values.

Reports are comparable only when their sigma domains and terminal-inclusive sample lengths
match. Alignment is exact `sigma_index`; the implementation never truncates, pads, resamples,
interpolates, or converts domains.

For each aligned index:

```text
absolute_difference = abs(sigma_a - sigma_b)
relative_difference = 0                         when both sigmas are zero
relative_difference = absolute_difference
                      / max(abs(sigma_a), abs(sigma_b)) otherwise
```

Comparison metrics use typed float64 IEEE-754 tokens. The summary records:

- exact-match count;
- maximum absolute difference and its first index;
- maximum symmetric relative difference and its first index;
- `math.fsum` mean absolute difference;
- `math.fsum` mean symmetric relative difference.

Domain mismatch returns `comparison.domain_mismatch`. Equal-domain length mismatch returns
`comparison.length_mismatch`. Both are non-comparable reports with no aligned samples or
statistics.

## 4. Canonical transport

Each projection is wrapped with its own lowercase SHA-256 fingerprint:

```json
{
  "report": {},
  "report_fingerprint": "sha256:...",
  "schema": "sigmax.schedule-report-envelope/1"
}
```

The comparison envelope uses
`sigmax.schedule-comparison-report-envelope/1`.

Serialization is canonical UTF-8 without a BOM, insignificant whitespace, or trailing newline.
Deserialization rejects:

- unsupported schemas or unknown members;
- duplicate object names;
- BOMs, malformed Unicode, noncanonical JSON, and untyped JSON floats;
- invalid or non-finite typed IEEE-754 values;
- unbounded depth, collections, strings, or byte size;
- noncontiguous indices or transform stages;
- sigma/delta, metric/summary, status/count, and mismatch-reason inconsistencies;
- stale report fingerprints.

## 5. Optional plotting

Default installation remains dependency-free. Plotting is installed explicitly:

```bash
python -m pip install "comfyui-sigmax[plot]"
```

The bounded extra is `matplotlib>=3.10,<3.12`. This covers the Python-3.10-compatible 3.10
line and the current Python-3.11+ 3.11 line while preventing an unreviewed next-major upgrade.

`render_schedule_plot()` and `render_schedule_comparison_plot()`:

- import Matplotlib lazily;
- use the object-oriented, headless canvas path;
- accept only `png` or `svg`;
- return bytes in memory;
- do not open GUI windows, write paths, access the network, or mutate reports;
- remove clock metadata where supported;
- reject non-comparable comparison reports.

Plot layout and bytes may vary across reviewed Matplotlib/font/rendering versions. Plot bytes
have no Sigmax fingerprint and must not replace artifacts, receipts, report transports, golden
vectors, or numerical parity.

The optional renderer is validated without pytest by:

```bash
python scripts/validate_schedule_plots.py
```

It renders single and comparison PNG/SVG payloads, checks their signatures, and verifies that
the canonical source reports remain byte-identical.

## 6. Public API example

```python
from comfyui_sigmax.core import (
    build_schedule_comparison_report,
    build_schedule_report_from_bundle,
    deserialize_portable_execution_bundle,
    serialize_schedule_comparison_report,
    serialize_schedule_report,
)

bundle = deserialize_portable_execution_bundle(bundle_payload)
report = build_schedule_report_from_bundle(bundle)
report_payload = serialize_schedule_report(report)

comparison = build_schedule_comparison_report(report, other_report)
comparison_payload = serialize_schedule_comparison_report(comparison)
```

With the optional extra:

```python
from comfyui_sigmax.plotting import render_schedule_plot

png_bytes = render_schedule_plot(report, image_format="png")
svg_bytes = render_schedule_plot(report, image_format="svg")
```

The caller owns any decision to persist those bytes.

## 7. Dependency sources

- [Matplotlib installation](https://matplotlib.org/stable/install/index.html)
- [Matplotlib dependencies](https://matplotlib.org/stable/install/dependencies.html)
- [Matplotlib pyplot/API overview](https://matplotlib.org/stable/api/pyplot_summary.html)
- [Matplotlib PyPI release metadata](https://pypi.org/project/matplotlib/)
