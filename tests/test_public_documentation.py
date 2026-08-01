"""Contract tests for the repository's public documentation surface."""

from __future__ import annotations

import re
from pathlib import Path

from comfyui_sigmax.profiles import (
    KREA2_RAW_OFFICIAL_SHA256,
    KREA2_TURBO_OFFICIAL_SHA256,
)

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DOCUMENTS = (
    ROOT / "README.md",
    ROOT / "CONTRIBUTING.md",
    ROOT / "CHANGELOG.md",
    ROOT / "docs" / "ARCHITECTURE.md",
    ROOT / "docs" / "USER_GUIDE.md",
    ROOT / "docs" / "STABLE_PUBLIC_CONTRACTS.md",
    ROOT / "docs" / "SECURITY_RELEASE_AUDIT.md",
    ROOT / "docs" / "COMFY_REGISTRY_RELEASE_ARTIFACT.md",
    ROOT / "docs" / "PROFILE_SPEC.md",
    ROOT / "docs" / "SCHEDULE_ARTIFACT_SPEC.md",
    ROOT / "docs" / "SCHEDULE_REPORT_SPEC.md",
    ROOT / "docs" / "NUMERICAL_BENCHMARK_MATRIX_SPEC.md",
    ROOT / "docs" / "IMAGE_BENCHMARK_PROTOCOL_SPEC.md",
    ROOT / "docs" / "DEPENDENCY_COMPATIBILITY_MATRIX_SPEC.md",
    ROOT / "docs" / "COINSTALLATION_MUTATION_MATRIX_SPEC.md",
    ROOT / "docs" / "PERFORMANCE_BUDGET_SPEC.md",
    ROOT / "docs" / "EXECUTION_RECEIPT_SPEC.md",
    ROOT / "docs" / "WORKFLOW_METADATA_SPEC.md",
    ROOT / "docs" / "COMPATIBILITY.md",
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_required_public_documents_exist() -> None:
    missing = [path.relative_to(ROOT).as_posix() for path in PUBLIC_DOCUMENTS if not path.is_file()]
    assert not missing, f"Missing public documentation: {missing}"


def test_v1_changelog_has_a_dated_release_and_new_unreleased_section() -> None:
    changelog = _read(ROOT / "CHANGELOG.md")

    unreleased = "## [Unreleased]"
    release = "## [1.0.0] - 2026-08-01"
    assert changelog.count(unreleased) == 1
    assert changelog.count(release) == 1
    assert changelog.index(unreleased) < changelog.index(release)
    assert (
        "[Unreleased]: https://github.com/rookiestar28/ComfyUI-Sigmax/compare/v1.0.0...HEAD"
        in changelog
    )
    assert "[1.0.0]: https://github.com/rookiestar28/ComfyUI-Sigmax/tree/v1.0.0" in changelog


def test_v1_user_guide_covers_installation_examples_migration_and_support() -> None:
    readme = _read(ROOT / "README.md")
    guide = _read(ROOT / "docs" / "USER_GUIDE.md")
    compatibility = _read(ROOT / "docs" / "COMPATIBILITY.md")

    for heading in (
        "## Installation",
        "## Verify the installation",
        "## Choose a node",
        "## Krea 2 Turbo example",
        "## Krea 2 RAW example",
        "## Workflow metadata",
        "## Update, upgrade, and rollback",
        "## Troubleshooting",
        "## Migrate to 1.0.0",
        "## Security and known limitations",
    ):
        assert heading in guide
    assert "git clone https://github.com/rookiestar28/ComfyUI-Sigmax comfyui-sigmax" in guide
    assert "pip install -r" not in guide
    assert "1.0.0" in guide
    assert "Sigmax.Krea2SigmaScheduler" in guide
    assert "Sigmax.ModelAwareSigmaScheduler" in guide
    assert "Euler + Simple" in guide
    assert "CFG 1.0" in guide
    assert "explicit" in guide.lower() and "RAW" in guide
    assert "https://docs.comfy.org/installation/install_custom_node" in guide
    assert "USER_GUIDE.md" in readme
    assert "pre-alpha" not in readme.lower()
    assert "pre-alpha" not in compatibility.lower()


def test_v1_contributor_guide_covers_release_contract_and_evidence() -> None:
    contributing = _read(ROOT / "CONTRIBUTING.md")

    for heading in (
        "## Architecture boundaries",
        "## Change workflow",
        "## Public contract and migration changes",
        "## Release-facing validation",
        "## Review evidence",
    ):
        assert heading in contributing
    for required in (
        "Reproduce -> Pin -> Sweep",
        "scripts/validate_registry_artifact.py",
        "scripts/run_release_audit.py",
        "docs/PROFILE_CONTRIBUTION_GUIDE.md",
        "Windows",
        "WSL",
        "public-contract",
    ):
        assert required in contributing
    assert "pre-alpha" not in contributing.lower()


def test_public_documentation_exposes_frozen_contract_and_migration_policy() -> None:
    readme = _read(ROOT / "README.md")
    architecture = _read(ROOT / "docs" / "ARCHITECTURE.md")
    specification = _read(ROOT / "docs" / "STABLE_PUBLIC_CONTRACTS.md")
    ci_matrix = _read(ROOT / "tests" / "CI_TEST_MATRIX.md")

    for content in (readme, specification):
        assert "sigmax.public-contract-manifest/1" in content
        assert "migration" in content.lower()
    assert "manifest_v1.json" in architecture
    assert "all twelve built-in node" in readme
    assert "Unknown schema identifiers fail" in specification
    assert "| Stable public contracts |" in ci_matrix


def test_public_documentation_exposes_security_release_audit() -> None:
    readme = _read(ROOT / "README.md")
    contributing = _read(ROOT / "CONTRIBUTING.md")
    specification = _read(ROOT / "docs" / "SECURITY_RELEASE_AUDIT.md")
    ci_matrix = _read(ROOT / "tests" / "CI_TEST_MATRIX.md")

    for content in (readme, specification):
        assert "sigmax.release-audit/1" in content
        assert "never" in content.lower() and "publish" in content.lower()
    assert "MANIFEST.in" in readme
    assert "scripts/run_release_audit.py" in contributing
    assert "software-source" in specification
    assert "model-weight" in specification
    assert "| Security and release audit |" in ci_matrix


def test_public_documentation_exposes_registry_artifact_boundary() -> None:
    readme = _read(ROOT / "README.md")
    contributing = _read(ROOT / "CONTRIBUTING.md")
    specification = _read(ROOT / "docs" / "COMFY_REGISTRY_RELEASE_ARTIFACT.md")
    test_sop = _read(ROOT / "tests" / "TEST_SOP.md")
    ci_matrix = _read(ROOT / "tests" / "CI_TEST_MATRIX.md")

    for content in (readme, specification, test_sop):
        assert "sigmax.registry-release-manifest/1" in content
        assert "publication_performed: false" in content
    assert ".comfyignore" in specification
    assert "normalized" in specification.lower() and "directory" in specification.lower()
    assert "scripts/validate_registry_artifact.py" in contributing
    assert "never" in specification.lower() and "publish" in specification.lower()
    assert "| Comfy Registry artifact |" in ci_matrix


def test_public_documentation_exposes_coinstallation_mutation_boundary() -> None:
    readme = _read(ROOT / "README.md")
    architecture = _read(ROOT / "docs" / "ARCHITECTURE.md")
    compatibility = _read(ROOT / "docs" / "COMPATIBILITY.md")
    specification = _read(ROOT / "docs" / "COINSTALLATION_MUTATION_MATRIX_SPEC.md")
    ci_matrix = _read(ROOT / "tests" / "CI_TEST_MATRIX.md")
    changelog = _read(ROOT / "CHANGELOG.md")

    for content in (readme, architecture, compatibility, specification, changelog):
        assert "sigmax.host-mutation-snapshot/1" in content
        assert "double-shift" in content
    assert "sigmax.co-installation-evaluation/1" in specification
    assert "sigmax.co-installation-mutation-matrix/1" in readme
    assert "coinstallation_matrix.py" in architecture
    assert "no external" in compatibility.lower()
    assert "| Co-installation mutation matrix |" in ci_matrix


def test_public_documentation_exposes_performance_budget_boundary() -> None:
    readme = _read(ROOT / "README.md")
    architecture = _read(ROOT / "docs" / "ARCHITECTURE.md")
    compatibility = _read(ROOT / "docs" / "COMPATIBILITY.md")
    specification = _read(ROOT / "docs" / "PERFORMANCE_BUDGET_SPEC.md")
    ci_matrix = _read(ROOT / "tests" / "CI_TEST_MATRIX.md")

    for content in (readme, architecture, compatibility, specification):
        assert "sigmax.performance-budget-matrix/1" in content
    assert "zero explicit device transfers" in readme
    assert "30-second" in compatibility
    assert "not_evaluated" in specification
    assert "| Performance budgets |" in ci_matrix


def test_public_documentation_exposes_environment_guardrails() -> None:
    readme = _read(ROOT / "README.md")
    architecture = _read(ROOT / "docs" / "ARCHITECTURE.md")
    compatibility = _read(ROOT / "docs" / "COMPATIBILITY.md")
    contributing = _read(ROOT / "CONTRIBUTING.md")
    test_sop = _read(ROOT / "tests" / "TEST_SOP.md")
    ci_matrix = _read(ROOT / "tests" / "CI_TEST_MATRIX.md")

    for content in (architecture, compatibility, contributing, test_sop):
        assert "sigmax.environment-diagnostics/1" in content
    assert "--optional-lane" in readme
    assert "PRE_COMMIT_HOME" in test_sop
    assert "SIGMAX_TEMP_ROOT" in test_sop
    assert "| Environment guardrails |" in ci_matrix


def test_public_documentation_describes_current_maturity_honestly() -> None:
    readme = _read(ROOT / "README.md").lower()
    compatibility = _read(ROOT / "docs" / "COMPATIBILITY.md").lower()
    architecture = _read(ROOT / "docs" / "ARCHITECTURE.md").lower()
    profile_spec = _read(ROOT / "docs" / "PROFILE_SPEC.md").lower()

    assert "stable public-contract baseline" in readme
    assert "fourteen namespaced nodes" in readme
    assert "planned" in architecture and "implemented" in architecture
    assert "frozen" in profile_spec and "sigmax.model-profile/1" in profile_spec
    assert "not yet validated" in compatibility


def test_public_documentation_exposes_current_artifact_transport_contract() -> None:
    readme = _read(ROOT / "README.md")
    artifact_spec = _read(ROOT / "docs" / "SCHEDULE_ARTIFACT_SPEC.md")

    assert "serialize_schedule_artifact" in readme
    assert "deserialize_schedule_artifact" in readme
    assert "sigmax.schedule-artifact-envelope/1" in artifact_spec
    assert "1,048,576 bytes" in artifact_spec


def test_public_documentation_exposes_receipt_and_bundle_boundary() -> None:
    readme = _read(ROOT / "README.md")
    receipt_spec = _read(ROOT / "docs" / "EXECUTION_RECEIPT_SPEC.md")
    compatibility = _read(ROOT / "docs" / "COMPATIBILITY.md")

    assert "build_execution_receipt" in readme
    assert "sigmax.execution-receipt/1" in receipt_spec
    assert "sigmax.portable-execution-bundle/1" in receipt_spec
    assert all(
        status in receipt_spec for status in ("not_executed", "succeeded", "failed", "interrupted")
    )
    assert "contracts do not execute a" in receipt_spec
    assert "M5-01 H3 harness is the first validated real-host producer" in receipt_spec
    assert "does not let normal workflows self-assert success" in receipt_spec
    assert "not yet validated" in compatibility


def test_public_documentation_exposes_workflow_metadata_boundary() -> None:
    readme = _read(ROOT / "README.md")
    workflow_spec = _read(ROOT / "docs" / "WORKFLOW_METADATA_SPEC.md")
    architecture = _read(ROOT / "docs" / "ARCHITECTURE.md")

    assert "attach_workflow_metadata" in readme
    assert "sigmax.workflow-metadata/1" in workflow_spec
    assert "extra.comfyui_sigmax" in workflow_spec
    assert all(version in workflow_spec for version in ("0.4", "version `1`"))
    assert "does not validate" in workflow_spec
    assert "workflow_metadata.py" in architecture


def test_public_documentation_exposes_workflow_validation_boundary() -> None:
    readme = _read(ROOT / "README.md")
    validation_spec = _read(ROOT / "docs" / "WORKFLOW_VALIDATION_SPEC.md")
    compatibility = _read(ROOT / "docs" / "COMPATIBILITY.md")
    architecture = _read(ROOT / "docs" / "ARCHITECTURE.md")

    assert "validate_pinned_workflow_fixtures" in readme
    assert "sigmax.workflow-validation-report/1" in validation_spec
    assert all(
        issue in validation_spec
        for issue in (
            "missing_node",
            "widget_slot_drift",
            "invalid_fixed_combo_value",
            "normalized_directory_failure",
            "malformed_metadata",
        )
    )
    assert all(lane in validation_spec for lane in ("known_good", "latest_host"))
    assert "literal-loopback" in validation_spec
    assert "H2_RAW_M3_06" in compatibility
    assert "workflows/validation.py" in architecture


def test_public_documentation_exposes_raw_host_workflow_boundary() -> None:
    readme = _read(ROOT / "README.md")
    validation_spec = _read(ROOT / "docs" / "WORKFLOW_VALIDATION_SPEC.md")
    compatibility = _read(ROOT / "docs" / "COMPATIBILITY.md")
    architecture = _read(ROOT / "docs" / "ARCHITECTURE.md")

    for fixture_id in (
        "krea2-raw-official-square-1024",
        "krea2-raw-official-landscape-1353x761",
        "krea2-raw-diffusers-portrait-761x1353",
    ):
        assert fixture_id in validation_spec
    for content in (readme, validation_spec, compatibility, architecture):
        assert "Sigmax.RawWorkflowOutput" in content
        assert "e651b7bef55a5376343dcb1c0edb79f0142c985e" in content  # pragma: allowlist secret
        assert "not_executed" in content
    assert "run_comfyui_e2e.py" in architecture
    assert "metadata reload" in validation_spec.lower()
    assert "model-free" in compatibility.lower()
    for content in (validation_spec, compatibility):
        assert "runtime rejection" in content.lower()
        assert "prequeue http 400" in content.lower()


def test_public_documentation_exposes_capability_preflight_contract() -> None:
    readme = _read(ROOT / "README.md")
    architecture = _read(ROOT / "docs" / "ARCHITECTURE.md")
    profile_spec = _read(ROOT / "docs" / "PROFILE_SPEC.md")

    assert "ModelCapabilities" in readme
    assert "ProfileCapabilities" in readme
    assert "SamplerCapabilities" in readme
    assert "CompatibilityDecision" in architecture
    assert all(level in profile_spec.lower() for level in ("allow", "warn", "reject"))


def test_public_documentation_exposes_capability_resolution_contract() -> None:
    readme = _read(ROOT / "README.md")
    architecture = _read(ROOT / "docs" / "ARCHITECTURE.md")
    profile_spec = _read(ROOT / "docs" / "PROFILE_SPEC.md")
    changelog = _read(ROOT / "CHANGELOG.md")

    for content in (readme, architecture, profile_spec, changelog):
        assert "sigmax.capability-resolution/1" in content
    assert "resolve_profile_capabilities" in readme
    assert "profiles/resolution.py" in architecture
    assert all(lifecycle in profile_spec for lifecycle in ("landed", "experimental", "unsupported"))
    assert "does not inspect a live host" in profile_spec


def test_public_documentation_exposes_comfyui_adapter_boundary() -> None:
    readme = _read(ROOT / "README.md")
    architecture = _read(ROOT / "docs" / "ARCHITECTURE.md")
    profile_spec = _read(ROOT / "docs" / "PROFILE_SPEC.md")
    compatibility = _read(ROOT / "docs" / "COMPATIBILITY.md")
    changelog = _read(ROOT / "CHANGELOG.md")

    for content in (readme, architecture, profile_spec, changelog):
        assert "sigmax.comfyui-adapter/1" in content
    assert "adapters/comfyui.py" in architecture
    assert "/object_info" in readme and "Node Definition JSON v2" in readme
    assert "v0_0_2" in profile_spec and "STABLE = False" in profile_spec
    assert "static-contract window" in compatibility
    assert "real-host" in compatibility


def test_public_documentation_exposes_node_registration_boundary() -> None:
    readme = _read(ROOT / "README.md")
    architecture = _read(ROOT / "docs" / "ARCHITECTURE.md")
    profile_spec = _read(ROOT / "docs" / "PROFILE_SPEC.md")
    compatibility = _read(ROOT / "docs" / "COMPATIBILITY.md")
    changelog = _read(ROOT / "CHANGELOG.md")

    for content in (readme, architecture, profile_spec, compatibility, changelog):
        assert "sigmax.node-registration/1" in content
    assert "adapters/registration.py" in architecture
    assert "Sigmax.<Name>" in readme
    assert "GET_SCHEMA()" in architecture
    assert "Node Definition JSON v2" in profile_spec
    assert "Sigmax.Krea2SigmaScheduler" in compatibility
    assert "real-host node/workflow E2E" in compatibility


def test_public_documentation_exposes_krea2_sigma_scheduler_node() -> None:
    readme = _read(ROOT / "README.md")
    architecture = _read(ROOT / "docs" / "ARCHITECTURE.md")
    profile_spec = _read(ROOT / "docs" / "PROFILE_SPEC.md")
    compatibility = _read(ROOT / "docs" / "COMPATIBILITY.md")
    changelog = _read(ROOT / "CHANGELOG.md")

    for content in (readme, architecture, profile_spec, compatibility, changelog):
        assert "Sigmax.Krea2SigmaScheduler" in content
        assert "sigmax.krea2-sigma-node/1" in content
    assert "strict-official" in readme
    assert "terminal-inclusive" in readme
    assert "not a sampler" in readme
    assert "execution time" in architecture
    assert "real-host" in compatibility


def test_public_documentation_exposes_model_aware_sigma_scheduler_node() -> None:
    readme = _read(ROOT / "README.md")
    architecture = _read(ROOT / "docs" / "ARCHITECTURE.md")
    profile_spec = _read(ROOT / "docs" / "PROFILE_SPEC.md")
    compatibility = _read(ROOT / "docs" / "COMPATIBILITY.md")
    changelog = _read(ROOT / "CHANGELOG.md")

    for content in (readme, architecture, profile_spec, compatibility, changelog):
        assert "Sigmax.ModelAwareSigmaScheduler" in content
        assert "sigmax.model-aware-sigma-node/1" in content
    assert "family-only" in readme and "ambiguous" in readme
    assert "static_contract" in architecture
    assert "generic fallback" in profile_spec
    assert "real-host" in compatibility


def test_public_documentation_exposes_frozen_profile_schema_v1() -> None:
    readme = _read(ROOT / "README.md")
    architecture = _read(ROOT / "docs" / "ARCHITECTURE.md")
    profile_spec = _read(ROOT / "docs" / "PROFILE_SPEC.md")
    changelog = _read(ROOT / "CHANGELOG.md")

    for content in (readme, architecture, profile_spec, changelog):
        assert "ProfileSchemaV1" in content
    assert "sigmax.model-profile/1" in profile_spec
    assert "SoftwareSourceProvenance" in profile_spec
    assert "FrameworkProvenance" in profile_spec
    assert "ModelWeightProvenance" in profile_spec
    assert "profile_schema_fingerprint" in readme
    assert "profiles/schema_v1.py" in architecture


def test_public_documentation_exposes_namespaced_profile_registry() -> None:
    readme = _read(ROOT / "README.md")
    architecture = _read(ROOT / "docs" / "ARCHITECTURE.md")
    profile_spec = _read(ROOT / "docs" / "PROFILE_SPEC.md")
    changelog = _read(ROOT / "CHANGELOG.md")

    for content in (readme, architecture, profile_spec, changelog):
        assert "ProfileRegistry" in content
    assert "builtin_profile_registry" in readme
    assert "REPLACE_EXTERNAL" in profile_spec


def test_public_documentation_bounds_generic_flowmatch_profiles() -> None:
    readme = _read(ROOT / "README.md")
    architecture = _read(ROOT / "docs" / "ARCHITECTURE.md")
    profile_spec = _read(ROOT / "docs" / "PROFILE_SPEC.md")
    stable_contracts = _read(ROOT / "docs" / "STABLE_PUBLIC_CONTRACTS.md")
    user_guide = _read(ROOT / "docs" / "USER_GUIDE.md")
    changelog = _read(ROOT / "CHANGELOG.md")

    for content in (readme, architecture, profile_spec, stable_contracts, changelog):
        assert "sigmax.generic-flowmatch-profile/1" in content
    for content in (readme, profile_spec, user_guide):
        assert "flowmatch.generic.fixed" in content
        assert "flowmatch.generic.dynamic" in content
        assert "explicit" in content.lower()
        assert "official" in content.lower()
    assert "ProfileRegistry" in profile_spec
    assert "model-specific" in user_guide.lower()
    assert "profiles/registry.py" in architecture
    assert "cannot replace a built-in" in profile_spec


def test_public_documentation_exposes_local_checkpoint_evidence_boundary() -> None:
    readme = _read(ROOT / "README.md")
    architecture = _read(ROOT / "docs" / "ARCHITECTURE.md")
    profile_spec = _read(ROOT / "docs" / "PROFILE_SPEC.md")
    stable_contracts = _read(ROOT / "docs" / "STABLE_PUBLIC_CONTRACTS.md")
    user_guide = _read(ROOT / "docs" / "USER_GUIDE.md")
    compatibility = _read(ROOT / "docs" / "COMPATIBILITY.md")
    changelog = _read(ROOT / "CHANGELOG.md")

    for content in (readme, architecture, stable_contracts, user_guide, changelog):
        assert "Sigmax.CheckpointEvidenceInspector" in content
    for content in (readme, architecture, profile_spec, stable_contracts, compatibility):
        assert "sigmax.checkpoint-evidence-inspection/1" in content
    for content in (readme, architecture, profile_spec, user_guide, compatibility):
        assert "payload" in content.lower()
        assert "network" in content.lower()
        assert "confirm" in content.lower()
    assert "checkpoint_inspection" in stable_contracts


def test_public_documentation_exposes_core_independence_boundary() -> None:
    readme = _read(ROOT / "README.md")
    compatibility = _read(ROOT / "docs" / "COMPATIBILITY.md")
    architecture = _read(ROOT / "docs" / "ARCHITECTURE.md")

    assert "check_core_independence.py" in readme
    assert "ComfyUI" in architecture and "Diffusers" in architecture
    assert "python -I" in architecture
    assert "deterministic property" in compatibility
    assert "framework" in compatibility and "native ComfyUI" in compatibility


def test_public_documentation_exposes_structural_krea2_turbo_profile() -> None:
    readme = _read(ROOT / "README.md")
    architecture = _read(ROOT / "docs" / "ARCHITECTURE.md")
    profile_spec = _read(ROOT / "docs" / "PROFILE_SPEC.md")
    compatibility = _read(ROOT / "docs" / "COMPATIBILITY.md")

    assert "KREA2_TURBO_PROFILE" in readme
    assert "build_krea2_turbo_schedule" in readme
    assert "krea2.turbo.official" in profile_spec
    assert "profiles/krea2_turbo.py" in architecture
    assert "structural profile" in compatibility.lower()
    assert "4, 8," in compatibility and "12, and 16" in compatibility
    assert "authoritative framework parity" in compatibility.lower()
    assert "diffusers 0.39.0" in compatibility.lower()
    assert "native comfyui parity" in compatibility.lower()
    assert "real-host" in compatibility.lower()


def test_public_documentation_exposes_turbo_golden_boundary() -> None:
    readme = _read(ROOT / "README.md")
    architecture = _read(ROOT / "docs" / "ARCHITECTURE.md")
    compatibility = _read(ROOT / "docs" / "COMPATIBILITY.md")
    changelog = _read(ROOT / "CHANGELOG.md")

    assert "krea2_turbo_v1.json" in readme
    assert "krea2_turbo_parity_v1.json" in readme
    assert "krea2_turbo_comfy_native_parity_v1.json" in readme
    assert "ModelSamplingFlux" in readme
    assert "integer-index quantization" in readme
    assert "Decimal" in architecture
    assert "float64" in compatibility and "float32" in compatibility
    assert "5.960464477539063e-08" in compatibility
    assert "framework" in changelog.lower() and "native-comfyui parity" in changelog.lower()


def test_public_documentation_exposes_raw_structural_profile_boundary() -> None:
    readme = _read(ROOT / "README.md")
    architecture = _read(ROOT / "docs" / "ARCHITECTURE.md")
    profile_spec = _read(ROOT / "docs" / "PROFILE_SPEC.md")
    compatibility = _read(ROOT / "docs" / "COMPATIBILITY.md")
    changelog = _read(ROOT / "CHANGELOG.md")

    for content in (readme, architecture, profile_spec, compatibility, changelog):
        assert "krea2.raw.official" in content
    assert "krea2.raw.official-full-52" in readme
    assert "krea2.raw.diffusers-reference-28" in readme
    assert "256" in compatibility and "6400" in compatibility
    assert "0.5" in compatibility and "1.15" in compatibility
    assert "unclamped" in compatibility.lower()
    assert "derive_krea2_raw_shift" in readme
    assert "build_krea2_raw_schedule" in readme
    assert "krea2_raw_v1.json" in readme
    assert "requested" in profile_spec.lower() and "effective" in profile_spec.lower()
    assert "packed image sequence length" in compatibility.lower()
    assert "resolve_krea2_image_geometry" in architecture
    assert "M3-02" not in readme


def test_public_documentation_exposes_raw_parity_boundary() -> None:
    readme = _read(ROOT / "README.md")
    architecture = _read(ROOT / "docs" / "ARCHITECTURE.md")
    compatibility = _read(ROOT / "docs" / "COMPATIBILITY.md")
    changelog = _read(ROOT / "CHANGELOG.md")

    assert "krea2_raw_parity_v1.json" in readme
    assert "FlowMatchEulerDiscreteScheduler" in readme
    assert "1.1920928955078125e-07" in readme
    assert "run_krea2_raw_parity.py" in architecture
    assert "all 14" in compatibility
    assert "Krea 2 RAW parity" in changelog


def test_public_documentation_exposes_fail_closed_variant_resolution() -> None:
    readme = _read(ROOT / "README.md")
    architecture = _read(ROOT / "docs" / "ARCHITECTURE.md")
    profile_spec = _read(ROOT / "docs" / "PROFILE_SPEC.md")
    compatibility = _read(ROOT / "docs" / "COMPATIBILITY.md")

    assert "resolve_krea2_variant" in readme
    assert "suggestions" in readme.lower()
    assert "strict official mode" in readme.lower()
    assert "krea2_variant.py" in architecture
    assert "conflicting resolving evidence" in architecture.lower()
    assert KREA2_RAW_OFFICIAL_SHA256 in profile_spec
    assert KREA2_TURBO_OFFICIAL_SHA256 in profile_spec
    assert "family-only" in compatibility.lower()


def test_public_documentation_exposes_advanced_flowmatch_scheduler_boundary() -> None:
    readme = _read(ROOT / "README.md")
    architecture = _read(ROOT / "docs" / "ARCHITECTURE.md")
    compatibility = _read(ROOT / "docs" / "COMPATIBILITY.md")
    changelog = _read(ROOT / "CHANGELOG.md")

    for content in (readme, architecture, compatibility, changelog):
        assert "Sigmax.AdvancedFlowMatchScheduler" in content
        assert "sigmax.advanced-flowmatch-node/1" in content
    assert "exponential_mu" in readme and "direct_ratio" in readme
    assert "UNIT_FLOW" in readme
    assert "experimental" in compatibility.lower()
    assert "not a sampler" in readme.lower()


def test_public_documentation_exposes_profile_and_schedule_inspectors() -> None:
    readme = _read(ROOT / "README.md")
    architecture = _read(ROOT / "docs" / "ARCHITECTURE.md")
    compatibility = _read(ROOT / "docs" / "COMPATIBILITY.md")
    changelog = _read(ROOT / "CHANGELOG.md")

    for content in (readme, architecture, compatibility, changelog):
        assert "Sigmax.ProfileInspector" in content
        assert "Sigmax.ScheduleInspector" in content
        assert "sigmax.profile-inspector/1" in content
        assert "sigmax.schedule-inspector/1" in content
    assert "ModelSamplingFlux" in readme
    assert "fingerprint" in readme.lower()
    assert "read-only" in compatibility.lower()


def test_public_contribution_and_changelog_contract() -> None:
    contributing = _read(ROOT / "CONTRIBUTING.md")
    changelog = _read(ROOT / "CHANGELOG.md")

    assert "scripts/run_full_tests_windows.ps1" in contributing
    assert "scripts/run_full_tests_linux.sh" in contributing
    assert ".venv" in contributing
    assert ".venv-wsl" in contributing
    assert "## [Unreleased]" in changelog


def test_public_test_governance_records_activated_native_euler_h3() -> None:
    test_sop = _read(ROOT / "tests" / "TEST_SOP.md").lower()
    e2e_sop = _read(ROOT / "tests" / "E2E_TESTING_SOP.md").lower()
    ci_matrix = _read(ROOT / "tests" / "CI_TEST_MATRIX.md").lower()

    for content in (test_sop, e2e_sop, ci_matrix):
        assert "m5-01 deterministic native-euler h3" in content
    assert "partial-denoise execution is rejected" in test_sop
    assert "remaining h3 capabilities" in e2e_sop
    assert "| real comfyui h3 |" in ci_matrix


def test_public_documentation_exposes_numerical_benchmark_matrix_boundary() -> None:
    readme = _read(ROOT / "README.md")
    architecture = _read(ROOT / "docs" / "ARCHITECTURE.md")
    compatibility = _read(ROOT / "docs" / "COMPATIBILITY.md")
    specification = _read(ROOT / "docs" / "NUMERICAL_BENCHMARK_MATRIX_SPEC.md")
    ci_matrix = _read(ROOT / "tests" / "CI_TEST_MATRIX.md")

    for content in (readme, architecture, specification):
        assert "sigmax.numerical-benchmark-matrix/1" in content
        assert "23" in content
    assert "BF16" in readme and "quantized" in readme
    assert "not_evaluated" in specification
    assert "dependency-free" in compatibility
    assert "| Numerical benchmark matrix | Implemented | M7-02 |" in ci_matrix


def test_public_documentation_exposes_optional_image_benchmark_boundary() -> None:
    readme = _read(ROOT / "README.md")
    architecture = _read(ROOT / "docs" / "ARCHITECTURE.md")
    compatibility = _read(ROOT / "docs" / "COMPATIBILITY.md")
    specification = _read(ROOT / "docs" / "IMAGE_BENCHMARK_PROTOCOL_SPEC.md")
    ci_matrix = _read(ROOT / "tests" / "CI_TEST_MATRIX.md")

    for content in (readme, architecture, specification):
        assert "sigmax.image-benchmark-protocol/1" in content
        assert "supplemental_only" in content
        assert "blind" in content.lower()
    assert "not_executed" in specification
    assert "gpu_model_weights_not_approved" in specification
    assert "mathematical parity" in compatibility.lower()
    assert (
        "| Optional image benchmark protocol | Implemented; execution unapproved | M7-03 |"
        in ci_matrix
    )


def test_public_documentation_exposes_dependency_compatibility_boundary() -> None:
    readme = _read(ROOT / "README.md")
    architecture = _read(ROOT / "docs" / "ARCHITECTURE.md")
    compatibility = _read(ROOT / "docs" / "COMPATIBILITY.md")
    specification = _read(ROOT / "docs" / "DEPENDENCY_COMPATIBILITY_MATRIX_SPEC.md")
    ci_matrix = _read(ROOT / "tests" / "CI_TEST_MATRIX.md")

    for content in (readme, architecture, specification):
        assert "sigmax.dependency-compatibility-matrix/1" in content
        assert "latest" in content
        assert "unavailable" in content
    assert "Python 3.10" in compatibility and "October 2026" in compatibility
    assert "@sha256:" in specification
    assert "registry_access_denied" in specification
    assert "release/HEAD latest-host evidence implemented" in ci_matrix
    assert "official container explicitly unavailable/non-blocking" in ci_matrix


def test_public_documents_do_not_expose_internal_workspace_material() -> None:
    forbidden = (
        ".planning/",
        ".planning\\",
        "ROADMAP.md",
        "AGENTS.md",
        "reference/docs",
        "reference\\docs",
        "B:\\",
        "/mnt/b/",
    )

    for path in PUBLIC_DOCUMENTS:
        content = _read(path)
        leaked = [marker for marker in forbidden if marker.lower() in content.lower()]
        assert not leaked, f"{path.relative_to(ROOT)} exposes internal markers: {leaked}"


def test_public_relative_markdown_links_resolve() -> None:
    link_pattern = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
    broken: list[str] = []

    for path in PUBLIC_DOCUMENTS:
        for target in link_pattern.findall(_read(path)):
            target_without_fragment = target.split("#", maxsplit=1)[0].strip()
            if not target_without_fragment or "://" in target_without_fragment:
                continue
            resolved = (path.parent / target_without_fragment).resolve()
            if not resolved.exists():
                broken.append(f"{path.relative_to(ROOT)} -> {target}")

    assert not broken, f"Broken local documentation links: {broken}"
