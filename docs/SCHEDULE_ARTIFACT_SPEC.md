# Schedule Artifact Specification

## Status

- **Specification:** Sigmax schedule artifact
- **Version:** 1
- **Artifact schema:** `sigmax.schedule-artifact/1`
- **Numerical projection schema:** `sigmax.numerical-schedule/1`
- **Transport envelope schema:** `sigmax.schedule-artifact-envelope/1`
- **Canonicalization profile:** Sigmax Canonical Projection v1
- **Maturity:** frozen public v1 contract under `sigmax.public-contract-manifest/1`

This document defines how Sigmax will describe a completed schedule construction and how it
will assign deterministic identities to the resulting numerical schedule and its construction
provenance. The pure-core runtime implements construction, canonical transport, strict
parsing, fingerprint verification, and golden fixtures.

## 1. Goals

The artifact format must make these questions independently answerable:

1. What exact normalized sigma sequence will the sampler receive?
2. Which requested inputs produced it?
3. Which effective inputs were actually used?
4. Which ordered transforms, terminal policy, and slicing policy were applied?
5. What evidence and source provenance support the selected behavior?
6. Did two constructions produce the same numerical schedule?
7. Did any semantically relevant construction input change?

The design therefore assigns two identities:

- A **numerical fingerprint** identifies the normalized ordered sigma values, sigma domain,
  and precision.
- A **construction fingerprint** identifies the effective construction and embeds the
  numerical fingerprint.

The fingerprints intentionally answer different questions. Two constructions may share a
numerical fingerprint while having different construction fingerprints.

## 2. Terminology

### Artifact

A transport object that contains the public construction record, its numerical schedule, and
both fingerprints. The construction fingerprint projection uses the artifact schema
`sigmax.schedule-artifact/1`.

### Projection

The bounded, validated object serialized into canonical bytes before hashing. A projection is
not arbitrary JSON and cannot contain unspecified host state.

### Requested inputs

The profile, step count, dimensions, precision, and other user-visible values requested before
resolution or policy processing.

### Effective inputs

The values actually used by the schedule engine after profile resolution, defaults, and
approved overrides. Requested inputs and effective inputs MUST remain distinct even when they
are equal.

### Override

An ordered record of one requested-to-effective change. An override identifies the field path,
requested value, effective value, and reason. A difference between requested inputs and
effective inputs without a corresponding override is invalid.

### Transform

One ordered transform stage with a stable identifier, input domain, output domain, stage
number, and complete typed parameters. Array order is execution order.

## 3. Artifact Model

The construction projection contains the following required top-level members:

| Member | Meaning |
| --- | --- |
| `schema` | Exact value `sigmax.schedule-artifact/1` |
| `engine` | Engine name and public version |
| `source` | Stable, non-private source identifier and NFC label |
| `evidence` | Evidence level and reference identifier |
| `ownership` | Schedule and shift ownership |
| `requested` | Complete requested inputs |
| `effective` | Complete effective inputs |
| `overrides` | Ordered requested-to-effective changes; empty array when none |
| `base_grid` | Stable builder identifier and complete typed parameters |
| `transforms` | Ordered transform records; empty array when none |
| `terminal` | Terminal policy and typed terminal value |
| `slicing` | Full, manual-range, or denoise-tail policy and effective bounds |
| `warnings` | Ordered public warnings; empty array when none |
| `numerical_fingerprint` | `sha256:` followed by 64 lowercase hexadecimal digits |

The following are excluded from the construction projection:

- the construction fingerprint itself;
- timestamps and process identifiers;
- private filesystem paths;
- credentials, passwords, cookies, API keys, or access tokens;
- model tensors, images, prompts, or other payloads;
- arbitrary ComfyUI or dependency dictionaries;
- diagnostic fields whose value does not change schedule construction.

New semantically relevant fields require a new schema version. Unknown members fail closed;
they are not silently ignored.

### 3.1 Transport Envelope

Serialized artifacts use an exact five-member envelope:

```json
{
  "construction": {},
  "construction_fingerprint": "sha256:...",
  "numerical": {},
  "numerical_fingerprint": "sha256:...",
  "schema": "sigmax.schedule-artifact-envelope/1"
}
```

The illustrative empty projections above are not valid runtime values. A valid envelope MUST:

1. contain exactly the five members shown;
2. embed complete v1 construction and numerical projections;
3. repeat both lowercase SHA-256 identities outside the projections;
4. carry the numerical identity inside the construction projection;
5. use the canonical UTF-8 bytes defined in Section 7;
6. contain no BOM, insignificant whitespace, or trailing newline.

The runtime parser accepts canonical `bytes` or Unicode text. Before JSON decoding, it rejects
transport larger than **1,048,576 bytes**, a Unicode or UTF-8 BOM, and invalid UTF-8. During
decoding, it rejects duplicate object names, JSON floating literals, and non-standard
`NaN`/`Infinity` constants. It then verifies the exact envelope fields, canonical byte form,
both projection schemas, schedule validity, the embedded numerical identity, and both
recomputed fingerprints.

Transport fingerprints prove integrity of these exact canonical values; they do not
authenticate the producer.

## 4. Numerical Projection

The numerical projection has exactly these members:

```json
{
  "domain": "unit_flow",
  "precision": "float64",
  "schema": "sigmax.numerical-schedule/1",
  "sigmas": [
    "3ff0000000000000",
    "0000000000000000"
  ]
}
```

Rules:

1. `schema` MUST equal `sigmax.numerical-schedule/1`.
2. `domain` MUST be the effective sigma-domain identifier.
3. `precision` MUST be `float32` or `float64`.
4. `sigmas` MUST preserve execution order.
5. Every sigma MUST be quantized to the declared precision before tokenization.
6. No profile, provenance, dimension, warning, or transform data appears in this projection.

The numerical fingerprint is:

```text
sha256:<lowercase SHA-256 of the exact numerical projection preimage>
```

## 5. Typed Floating-Point Tokens

Fingerprint projections contain no JSON floating-point numbers. Every semantic float is
represented as an object or list entry containing a precision and a fixed-width lowercase
IEEE-754 bit token.

| Precision | Encoding | Width |
| --- | --- | ---: |
| `float32` | big-endian IEEE-754 binary32 | 8 hexadecimal digits |
| `float64` | big-endian IEEE-754 binary64 | 16 hexadecimal digits |

The encoder MUST:

1. reject non-finite input, including NaN and positive or negative Infinity;
2. normalize to binary32 or binary64 before extracting bits;
3. normalize negative zero to positive zero;
4. serialize bytes in network/big-endian order;
5. emit lowercase hexadecimal without a prefix.

Thus both `-0.0` and `+0.0` become `00000000` in binary32 and
`0000000000000000` in binary64.

Integers that are schema-bounded counts, dimensions, indices, or versions remain JSON
integers. Booleans and null retain their JSON representations.

## 6. String and Key Rules

Before projection:

1. All string values MUST be valid Unicode scalar sequences.
2. All string values MUST be normalized to Unicode Normalization Form C (NFC).
3. All object keys MUST be schema-controlled ASCII identifiers.
4. User-controlled text MUST occur only in values, never as object keys.
5. Duplicate object names MUST be rejected during parsing or construction.

Restricting keys to ASCII makes ordinal ASCII, UTF-8, Unicode code-point, and UTF-16 ordering
equivalent for allowed keys.

## 7. Sigmax Canonical Projection v1

Canonicalization proceeds as follows:

1. Validate the exact schema version and allowed members.
2. Reject duplicate keys, unknown members, invalid Unicode, and unsupported types.
3. Normalize all string values to NFC.
4. Replace every semantic float with its typed IEEE-754 token.
5. Normalize negative zero and reject every non-finite value.
6. Sort every object's ASCII keys in ascending ordinal order, recursively.
7. Preserve array order.
8. Serialize compact JSON with `,` and `:` separators and no insignificant whitespace.
9. Encode as UTF-8 without a byte-order mark.
10. Use the exact bytes with no trailing newline as the hash preimage.

Repository fixture files use one final LF as a text-file transport convention. The fixture
loader removes exactly that one LF; the LF is not part of the canonical preimage.

This format is **JCS-informed** because it adopts deterministic key ordering, compact UTF-8
JSON, and strict values from [RFC 8785](https://www.rfc-editor.org/rfc/rfc8785). It is
**not JCS-compliant** because Sigmax applies NFC preprocessing and replaces JSON floating-point
numbers with typed bit tokens instead of ECMAScript number serialization.

## 8. Construction Fingerprint

The construction fingerprint is:

```text
sha256:<lowercase SHA-256 of the exact construction projection preimage>
```

Every field that can change the intended construction MUST appear in the projection,
including:

- profile and variant selection;
- evidence and source provenance;
- schedule and shift ownership;
- requested inputs;
- effective inputs;
- every override;
- base-grid identifier and typed parameters;
- every ordered transform and typed parameter;
- terminal policy;
- slicing policy and effective bounds;
- warnings that describe an effective assumption or deviation;
- the numerical fingerprint.

The construction fingerprint itself MUST NOT appear in its own projection.

## 9. Ordering and Provenance Invariants

The runtime implementation MUST validate:

1. transform `stage` values are contiguous and equal their array indices;
2. adjacent transform output/input domains match;
3. transform array order is execution order;
4. override array order is resolution order;
5. each requested/effective difference has exactly one explanatory override;
6. the terminal stage is represented exactly once;
7. slicing records effective transition bounds, not only the original request;
8. the embedded numerical fingerprint matches the numerical projection.

Changing provenance while preserving sigmas changes only the construction fingerprint.
Changing the domain, precision, order, length, or bits of sigmas changes the numerical
fingerprint and therefore also changes the construction fingerprint.

## 10. Evidence and Warnings

`evidence.level` uses the model-profile evidence vocabulary:

- `official`
- `framework_reference`
- `community_recommended`
- `experimental`

Any user override of official behavior makes the construction modified. The original evidence
source remains recorded, while the override and warning explain the deviation. Experimental
behavior must never be serialized as official.

Warnings are ordered because order can express resolution sequence and because unordered
collections would introduce unstable canonicalization. Equivalent producers MUST emit the
same warning identifiers and order for the same effective construction.

## 11. Security Properties and Non-Properties

SHA-256 provides deterministic identity and integrity for canonical bytes. It provides
identity and integrity, not authenticity, authorization, provenance attestation, or a digital
signature. Untrusted
artifacts remain untrusted after hashing.

Consumers MUST:

- validate before displaying or acting on fields;
- bound nesting, string lengths, arrays, and integers;
- never deserialize an artifact into executable objects;
- never treat source labels or warnings as trusted markup;
- never include secrets or private paths in projections;
- recompute both fingerprints before trusting identity comparisons.

A future signing design would be a separate protocol and schema.

## 12. Golden Fixtures

The repository publishes:

- `numerical_projection_v1.json`;
- `construction_projection_a_v1.json`;
- `construction_projection_b_v1.json`;
- `golden_hashes_v1.json`.

The two construction fixtures intentionally share one numerical fingerprint. They differ in
requested steps, effective-input provenance, override data, and warnings, so their construction
fingerprints differ.

The fixtures pin:

- exact canonical UTF-8 preimages;
- float32 and float64 token widths;
- positive-zero normalization;
- NFC text;
- requested/effective separation;
- transform order;
- cross-process and hash-seed stability.

## 13. Compatibility and Versioning

Schema identifiers are exact and versioned. Producers MUST NOT add fields to v1. Consumers
MUST reject an unknown major schema rather than guessing.

A v1 revision may clarify prose or add tests only when canonical semantics and existing golden
bytes do not change. Any change to member meaning, projection membership, normalization,
serialization, or fingerprint preimages requires a new schema identifier.

## 14. Standards Basis

- [RFC 8785 — JSON Canonicalization Scheme](https://www.rfc-editor.org/rfc/rfc8785)
- [RFC 8259 — The JavaScript Object Notation Data Interchange Format](https://www.rfc-editor.org/rfc/rfc8259)
- [Unicode Standard Annex #15 — Unicode Normalization Forms](https://unicode.org/reports/tr15/)
- [Python `struct` — packed binary data](https://docs.python.org/3/library/struct.html)
- [Python `hashlib` — secure hashes](https://docs.python.org/3/library/hashlib.html)
