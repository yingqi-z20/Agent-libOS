from __future__ import annotations

import hashlib
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

from agent_libos.capability.manager import CapabilityManager
from agent_libos.config import DEFAULT_CONFIG, AgentLibOSConfig
from agent_libos.models import (
    PERMITTED_EFFECTS_POLICY_PROVENANCE_KEY,
    PERMITTED_EFFECTS_POLICY_SCHEMA_VERSION,
    CapabilityRight,
    DataLabels,
    ResourceBudget,
    TaskAuthorityManifest,
    encode_permitted_effects_policy,
)
from agent_libos.models.exceptions import CapabilityDenied, NotFound, ValidationError
from agent_libos.storage import AuthorityRepository
from agent_libos.utils.ids import new_id, utc_now
from agent_libos.utils.serde import dumps


_MANIFEST_INPUT_FIELDS = frozenset(
    {
        "authorized_capabilities",
        "permitted_effects",
        "resource_budget",
        "approval_policy",
        "data_flow_policy",
        "expires_at",
        "issued_by",
        "metadata",
    }
)

_CAPABILITY_SPEC_FIELDS = frozenset(
    {
        "resource",
        "rights",
        "constraints",
        "delegable",
        "revocable",
        "expires_at",
        "uses_remaining",
        "max_delegation_depth",
    }
)


class AuthorityManifestManager:
    """Create, compile, and enforce durable process launch contracts."""

    def __init__(
        self,
        store: AuthorityRepository,
        capabilities: CapabilityManager,
        audit: Any,
        events: Any,
        images: Mapping[str, Any],
        *,
        config: AgentLibOSConfig | None = None,
    ) -> None:
        self.store = store
        self.capabilities = capabilities
        self.audit = audit
        self.events = events
        self._images = images
        self.config = config or DEFAULT_CONFIG

    def prepare_launch(
        self,
        *,
        pid: str,
        image_id: str,
        goal_ref: str | None,
        supplied: TaskAuthorityManifest | dict[str, Any] | str | None = None,
        authorized_capabilities: Iterable[dict[str, Any]] = (),
        resource_budget: ResourceBudget | dict[str, Any] | None = None,
        parent_pid: str | None = None,
        issued_by: str = "runtime.bootstrap",
    ) -> TaskAuthorityManifest:
        image = self._image(image_id)
        required = self._normalize_specs(list(getattr(image, "required_capabilities", []) or []))
        requested = self._normalize_specs(list(authorized_capabilities))
        parent = self.get_for_process(parent_pid) if parent_pid is not None else None

        payload = self._launch_payload(supplied)

        declared = self._normalize_specs(payload.get("authorized_capabilities", requested))
        if requested and payload:
            for spec in requested:
                self._require_spec_covered(declared, spec, label="launch request")
        parent_is_ceiling = parent is not None and bool(
            parent.metadata.get("transition_ceiling", parent.metadata.get("explicit"))
        )
        if parent_is_ceiling:
            for spec in declared:
                self._require_spec_covered(
                    parent.authorized_capabilities,
                    spec,
                    label="derived child authority",
                )

        selected_budget = self._resolve_resource_budget(payload, resource_budget)
        if parent_is_ceiling:
            self._require_budget_attenuated(parent.resource_budget, selected_budget)

        parent_approval_policy = dict(parent.approval_policy) if parent is not None else {}
        supplied_approval_policy = self._mapping(
            payload.get("approval_policy"),
            "approval_policy",
        )
        approval_policy = {**parent_approval_policy, **supplied_approval_policy}
        if "requestable_capabilities" in approval_policy:
            approval_policy["requestable_capabilities"] = (
                self._normalize_requestable_specs(
                    approval_policy["requestable_capabilities"]
                )
            )
        permitted_effects = self._effect_classes(
            payload["permitted_effects"]
            if "permitted_effects" in payload
            else (parent.permitted_effects if parent is not None else None)
        )
        parent_data_flow_policy = (
            self._normalize_data_flow_policy(parent.data_flow_policy)
            if parent is not None
            else self._normalize_data_flow_policy({})
        )
        data_flow_policy = (
            self._normalize_data_flow_policy(payload.get("data_flow_policy"))
            if "data_flow_policy" in payload
            else parent_data_flow_policy
        )
        expires_at = self._optional_expiry(
            payload["expires_at"]
            if "expires_at" in payload
            else (parent.expires_at if parent is not None else None),
            "expires_at",
        )
        if parent_is_ceiling:
            self._require_effects_attenuated(parent.permitted_effects, permitted_effects)
            self._require_requestable_capabilities_attenuated(parent, approval_policy)
            self._require_policy_mapping_attenuated(
                parent.approval_policy,
                approval_policy,
                label="approval_policy",
                ignored_keys={"requestable_capabilities"},
            )
        if parent is not None:
            self._require_data_flow_policy_attenuated(
                parent_data_flow_policy,
                data_flow_policy,
            )
            self._require_expiry_attenuated(parent.expires_at, expires_at)
        manifest = TaskAuthorityManifest(
            manifest_id=new_id("authm"),
            pid=pid,
            image_id=image_id,
            goal_ref=goal_ref,
            authorized_capabilities=declared,
            required_capabilities=required,
            permitted_effects=permitted_effects,
            resource_budget=selected_budget,
            approval_policy=approval_policy,
            data_flow_policy=data_flow_policy,
            expires_at=expires_at,
            issued_by=str(payload.get("issued_by") or issued_by),
            parent_manifest_id=parent.manifest_id if parent is not None else None,
            metadata={
                **self._mapping(payload.get("metadata"), "metadata"),
                "launch_authority_mode": self.config.runtime.launch_authority_mode,
                # Only a host-supplied manifest is a transition ceiling.  The
                # durable implicit manifest still denies model permission
                # requests outside its (empty) request contract, but it must
                # not erase capabilities the Host deliberately grants later.
                "explicit": supplied is not None,
                "transition_ceiling": supplied is not None or bool(parent_is_ceiling),
                PERMITTED_EFFECTS_POLICY_PROVENANCE_KEY: PERMITTED_EFFECTS_POLICY_SCHEMA_VERSION,
            },
            created_at=utc_now(),
        )
        manifest = self._insert_hashed_manifest(manifest)
        missing_required = self._missing_required_capabilities(manifest)
        self.audit.record(
            actor=manifest.issued_by,
            action="authority_manifest.bind",
            target=f"process:{pid}",
            decision={
                "manifest_id": manifest.manifest_id,
                "manifest_hash": manifest.manifest_hash,
                "image_id": image_id,
                "authorized_capabilities": len(manifest.authorized_capabilities),
                "required_capabilities": len(required),
                "missing_required_capabilities": len(missing_required),
                "parent_manifest_id": manifest.parent_manifest_id,
            },
        )
        return manifest

    def compile_root_capabilities(self, manifest: TaskAuthorityManifest) -> list[str]:
        cap_ids: list[str] = []
        for spec in self.bounded_capability_specs(manifest):
            cap = self.capabilities.issue_trusted(
                subject=manifest.pid,
                resource=str(spec["resource"]),
                rights=list(spec["rights"]),
                issued_by=f"authority_manifest:{manifest.manifest_id}",
                constraints=dict(spec.get("constraints") or {}),
                expires_at=spec.get("expires_at"),
                uses_remaining=spec.get("uses_remaining"),
                delegable=bool(spec.get("delegable", False)),
                revocable=bool(spec.get("revocable", True)),
                max_delegation_depth=spec.get("max_delegation_depth"),
                metadata={"authority_manifest_id": manifest.manifest_id},
            )
            cap_ids.append(cap.cap_id)
        return cap_ids

    def bounded_capability_specs(
        self,
        manifest: TaskAuthorityManifest,
    ) -> list[dict[str, Any]]:
        """Return declared grants with every lease capped by the manifest."""

        selected: list[dict[str, Any]] = []
        for raw_spec in manifest.authorized_capabilities:
            spec = dict(raw_spec)
            expires_at = self._bounded_expiry(
                manifest.expires_at,
                spec.get("expires_at"),
            )
            if expires_at is None:
                spec.pop("expires_at", None)
            else:
                spec["expires_at"] = expires_at
            selected.append(spec)
        return selected

    def bind_checkpoint_fork(
        self,
        *,
        source_pid: str,
        target_pid: str,
        image_id: str,
        goal_ref: str | None,
        authorized_capabilities: Iterable[dict[str, Any]],
        resource_budget: ResourceBudget | dict[str, Any] | None,
        parent_manifest_id: str | None = None,
        issued_by: str,
    ) -> TaskAuthorityManifest:
        """Bind a forked process without reissuing its already-copied capabilities."""

        source = self.get_for_process(source_pid)
        image = self._image(image_id)
        required = self._normalize_specs(list(getattr(image, "required_capabilities", []) or []))
        actual = self._normalize_specs(list(authorized_capabilities))
        source_is_ceiling = source is not None and bool(
            source.metadata.get("transition_ceiling", source.metadata.get("explicit"))
        )
        if source_is_ceiling:
            for spec in actual:
                self._require_spec_covered(
                    source.authorized_capabilities,
                    spec,
                    label="checkpoint fork authority",
                )
            declared = actual
        else:
            # Preserve the implicit manifest's empty model-request contract.
            # The copied capabilities remain usable, but do not become newly
            # requestable merely because the process crossed a checkpoint.
            declared = []
        selected_budget = self._budget_dict(resource_budget)
        if source_is_ceiling and source is not None:
            self._require_budget_attenuated(source.resource_budget, selected_budget)
        now = utc_now()
        manifest = TaskAuthorityManifest(
            manifest_id=new_id("authm"),
            pid=target_pid,
            image_id=image_id,
            goal_ref=goal_ref,
            authorized_capabilities=declared,
            required_capabilities=required,
            permitted_effects=(
                list(source.permitted_effects)
                if source is not None and source.permitted_effects is not None
                else None
            ),
            resource_budget=selected_budget,
            approval_policy=dict(source.approval_policy) if source is not None else {},
            data_flow_policy=dict(source.data_flow_policy) if source is not None else {},
            expires_at=source.expires_at if source is not None else None,
            issued_by=issued_by,
            parent_manifest_id=(
                parent_manifest_id
                if parent_manifest_id is not None
                else (source.manifest_id if source is not None else None)
            ),
            metadata={
                **(dict(source.metadata) if source is not None else {}),
                "launch_authority_mode": self.config.runtime.launch_authority_mode,
                "explicit": bool(source.metadata.get("explicit")) if source is not None else False,
                "transition_ceiling": source_is_ceiling,
                "checkpoint_fork_source_pid": source_pid,
                "checkpoint_fork_source_manifest_id": source.manifest_id if source is not None else None,
                PERMITTED_EFFECTS_POLICY_PROVENANCE_KEY: PERMITTED_EFFECTS_POLICY_SCHEMA_VERSION,
            },
            created_at=now,
        )
        manifest = self._insert_hashed_manifest(manifest)
        missing_required = self._missing_required_capabilities(manifest)
        self.audit.record(
            actor=issued_by,
            action="authority_manifest.bind_checkpoint_fork",
            target=f"process:{target_pid}",
            decision={
                "manifest_id": manifest.manifest_id,
                "manifest_hash": manifest.manifest_hash,
                "source_pid": source_pid,
                "source_manifest_id": source.manifest_id if source is not None else None,
                "authorized_capabilities": len(manifest.authorized_capabilities),
                "missing_required_capabilities": len(missing_required),
                "parent_manifest_id": manifest.parent_manifest_id,
            },
        )
        return manifest

    def get(self, manifest_id: str) -> TaskAuthorityManifest:
        manifest = self.store.get_authority_manifest(manifest_id)
        if manifest is None:
            raise NotFound(f"authority manifest not found: {manifest_id}")
        unhashed = replace(manifest, manifest_hash="")
        policy_schema_version = manifest.permitted_effects_policy_schema_version
        provenance = manifest.metadata.get(PERMITTED_EFFECTS_POLICY_PROVENANCE_KEY)
        if policy_schema_version == PERMITTED_EFFECTS_POLICY_SCHEMA_VERSION:
            valid = (
                provenance == PERMITTED_EFFECTS_POLICY_SCHEMA_VERSION
                and self._hash(unhashed) == manifest.manifest_hash
            )
        elif policy_schema_version == 1:
            # Legacy rows predate the provenance marker and persisted the
            # logical list directly. Their empty list meant unrestricted. The
            # marker absence plus the decoded v1 storage shape is required;
            # payload shape alone is never enough to select this fallback.
            valid = (
                provenance is None
                and self._legacy_hash(unhashed) == manifest.manifest_hash
            )
        else:
            valid = False
        if not valid:
            raise ValidationError(f"authority manifest hash mismatch: {manifest_id}")
        return manifest

    def get_for_process(self, pid: str | None) -> TaskAuthorityManifest | None:
        if pid is None:
            return None
        manifest = self.store.get_authority_manifest_for_process(pid)
        if manifest is None:
            return None
        return self.get(manifest.manifest_id)

    def assert_capability_request(
        self,
        pid: str,
        resource: str,
        rights: Iterable[str],
    ) -> dict[str, Any]:
        manifest = self.get_for_process(pid)
        if manifest is None:
            raise CapabilityDenied(f"{pid} has no task authority manifest")
        self._require_live(manifest)
        spec = self._normalize_spec({"resource": resource, "rights": list(rights)})
        requestable = self._normalize_requestable_specs(
            manifest.approval_policy.get("requestable_capabilities", [])
        )
        for ceiling in [*manifest.authorized_capabilities, *requestable]:
            candidate = dict(spec)
            if ceiling.get("expires_at") is not None:
                candidate["expires_at"] = ceiling["expires_at"]
            if self.capabilities.spec_covers(ceiling, candidate):
                return candidate
        raise CapabilityDenied(
            f"permission request exceeds task authority manifest: "
            f"{spec['resource']} rights={spec['rights']}"
        )

    def assert_effect(self, pid: str, effect_class: str) -> None:
        manifest = self.get_for_process(pid)
        if manifest is None:
            return
        self._require_live(manifest)
        if manifest.permitted_effects is None:
            return
        selected = str(effect_class).strip()
        if any(self._effect_matches(pattern, selected) for pattern in manifest.permitted_effects):
            return
        raise CapabilityDenied(
            f"task authority manifest {manifest.manifest_id} does not permit effect class {selected}"
        )

    def bound_capability_expiry(self, pid: str, expires_at: str | None) -> str | None:
        """Return a capability expiry constrained by the live task manifest."""

        selected_expiry = self._optional_expiry(
            expires_at,
            "capability expires_at",
        )
        manifest = self.get_for_process(pid)
        if manifest is None:
            return selected_expiry
        self._require_live(manifest)
        return self._bounded_expiry(manifest.expires_at, selected_expiry)

    def resolve_launch_resource_budget(
        self,
        *,
        supplied: TaskAuthorityManifest | dict[str, Any] | str | None,
        resource_budget: ResourceBudget | dict[str, Any] | None,
        parent_pid: str | None,
    ) -> dict[str, Any]:
        """Resolve the effective budget before a child row or reservation exists."""

        payload = self._launch_payload(supplied)
        selected = self._resolve_resource_budget(payload, resource_budget)
        parent = self.get_for_process(parent_pid) if parent_pid is not None else None
        parent_is_ceiling = parent is not None and bool(
            parent.metadata.get("transition_ceiling", parent.metadata.get("explicit"))
        )
        if parent_is_ceiling:
            self._require_budget_attenuated(parent.resource_budget, selected)
        return selected

    def assert_data_flow_labels(self, pid: str, labels: DataLabels | Any) -> None:
        """Enforce the process's inbound tenant/principal domain.

        This policy is deliberately independent from external Sink trust.  It
        can only constrain which labeled internal handoffs a process receives;
        it cannot make any external Sink trusted or reduce a Host clearance.
        """

        selected = labels if isinstance(labels, DataLabels) else DataLabels.from_object_metadata(labels)
        if selected.is_mixed_identity:
            raise CapabilityDenied(
                f"process {pid} cannot receive mixed tenant/principal data without Host reclassification"
            )
        manifest = self.get_for_process(pid)
        policy = self._normalize_data_flow_policy(
            manifest.data_flow_policy if manifest is not None else {}
        )
        for field, value in (
            ("allowed_tenants", selected.tenant),
            ("allowed_principals", selected.principal),
        ):
            if value is not None and value not in set(policy[field]):
                raise CapabilityDenied(
                    f"process {pid} data_flow_policy does not allow {field[:-1]} {value!r}"
                )

    def summary_for_process(self, pid: str) -> dict[str, Any] | None:
        manifest = self.get_for_process(pid)
        if manifest is None:
            return None
        return {
            "manifest_id": manifest.manifest_id,
            "manifest_hash": manifest.manifest_hash,
            "image_id": manifest.image_id,
            "parent_manifest_id": manifest.parent_manifest_id,
            "authorized_capabilities": manifest.authorized_capabilities,
            "required_capabilities": manifest.required_capabilities,
            "missing_required_capabilities": self._missing_required_capabilities(manifest),
            "permitted_effects": manifest.permitted_effects,
            "resource_budget": manifest.resource_budget,
            "approval_policy": manifest.approval_policy,
            "requestable_capabilities": self._normalize_requestable_specs(
                manifest.approval_policy.get("requestable_capabilities", [])
            ),
            "data_flow_policy": manifest.data_flow_policy,
            "expires_at": manifest.expires_at,
        }

    def _image(self, image_id: str) -> Any:
        image = self._images.get(image_id)
        if image is None:
            raise NotFound(f"agent image not found: {image_id}")
        return image

    def _launch_payload(
        self,
        supplied: TaskAuthorityManifest | dict[str, Any] | str | None,
    ) -> dict[str, Any]:
        if isinstance(supplied, str):
            payload = self._template_payload(self.get(supplied))
        elif isinstance(supplied, TaskAuthorityManifest):
            payload = self._template_payload(supplied)
        else:
            payload = self._mapping(supplied, "authority manifest")
        self._reject_unknown_fields(
            payload,
            _MANIFEST_INPUT_FIELDS,
            label="authority manifest",
        )
        return payload

    def _normalize_specs(self, values: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        if not isinstance(values, (list, tuple)):
            raise ValidationError("authority manifest capability collections must be lists")
        return [self._normalize_spec(value) for value in values]

    def _normalize_requestable_specs(
        self,
        values: Iterable[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Normalize model-request ceilings without implying grant semantics.

        A model permission request carries only resource, rights, and an expiry
        inherited from the matching ceiling.  Accept the canonical no-op values
        emitted by existing Host clients, but reject fields whose grant or lease
        semantics cannot be propagated through that request path.
        """

        raw_values = list(values) if isinstance(values, (list, tuple)) else values
        selected = self._normalize_specs(raw_values)
        for raw, spec in zip(raw_values, selected):
            if "constraints" in raw and raw.get("constraints") != {}:
                raise ValidationError(
                    "requestable capability constraints must be an empty object"
                )
            if spec.get("delegable") is not False:
                raise ValidationError(
                    "requestable capability delegable must be false"
                )
            if spec.get("revocable") is not True:
                raise ValidationError(
                    "requestable capability revocable must be true"
                )
            if "uses_remaining" in raw:
                raise ValidationError(
                    "requestable capability uses_remaining is not supported"
                )
            if "max_delegation_depth" in raw:
                raise ValidationError(
                    "requestable capability max_delegation_depth is not supported"
                )
        return selected

    def _normalize_data_flow_policy(self, value: Any) -> dict[str, Any]:
        selected = self._mapping(value, "data_flow_policy")
        allowed = {
            "schema_version",
            "allowed_tenants",
            "allowed_principals",
        }
        unknown = sorted(str(field) for field in selected if field not in allowed)
        if unknown:
            raise ValidationError(
                f"data_flow_policy contains unsupported fields: {unknown}"
            )
        schema_version = selected.get("schema_version", 1)
        if schema_version != 1:
            raise ValidationError(
                f"unsupported data_flow_policy schema_version: {schema_version}"
            )
        return {
            "schema_version": 1,
            "allowed_tenants": self._data_flow_identities(
                selected.get("allowed_tenants", []),
                "allowed_tenants",
            ),
            "allowed_principals": self._data_flow_identities(
                selected.get("allowed_principals", []),
                "allowed_principals",
            ),
        }

    @staticmethod
    def _data_flow_identities(value: Any, label: str) -> list[str]:
        if not isinstance(value, list):
            raise ValidationError(f"data_flow_policy.{label} must be a list")
        selected: set[str] = set()
        for raw in value:
            if not isinstance(raw, str) or not raw or raw != raw.strip():
                raise ValidationError(
                    f"data_flow_policy.{label} entries must be non-empty canonical strings"
                )
            if raw in {"*", "mixed"} or len(raw) > 256 or "\x00" in raw:
                raise ValidationError(
                    f"data_flow_policy.{label} cannot contain {raw!r}"
                )
            selected.add(raw)
        return sorted(selected)

    @staticmethod
    def _require_data_flow_policy_attenuated(
        parent: dict[str, Any],
        child: dict[str, Any],
    ) -> None:
        if child.get("schema_version") != parent.get("schema_version"):
            raise CapabilityDenied("child data_flow_policy cannot change schema_version")
        for field in ("allowed_tenants", "allowed_principals"):
            parent_values = set(parent.get(field) or ())
            child_values = set(child.get(field) or ())
            if not child_values.issubset(parent_values):
                raise CapabilityDenied(
                    f"child data_flow_policy cannot widen {field}: "
                    f"parent={sorted(parent_values)} child={sorted(child_values)}"
                )

    @staticmethod
    def _dedupe_specs(values: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        selected: dict[str, dict[str, Any]] = {}
        for value in values:
            selected[dumps(value)] = dict(value)
        return [selected[key] for key in sorted(selected)]

    def _normalize_spec(self, value: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise ValidationError("authority manifest capability entries must be objects")
        self._reject_unknown_fields(
            value,
            _CAPABILITY_SPEC_FIELDS,
            label="authority manifest capability entry",
        )
        resource = str(value.get("resource") or "").strip()
        self.capabilities.parse_resource_pattern(resource)
        rights = sorted({CapabilityRight(str(right)).value for right in value.get("rights", [])})
        if not rights:
            raise ValidationError("authority manifest capability entry requires rights")
        selected: dict[str, Any] = {
            "resource": resource,
            "rights": rights,
            "constraints": self._mapping(value.get("constraints"), "capability constraints"),
            "delegable": self._capability_boolean(value, "delegable", default=False),
            "revocable": self._capability_boolean(value, "revocable", default=True),
        }
        if value.get("expires_at") is not None:
            selected["expires_at"] = self._optional_expiry(
                value["expires_at"],
                "capability expires_at",
            )
        if value.get("uses_remaining") is not None:
            uses_remaining = value["uses_remaining"]
            if (
                isinstance(uses_remaining, bool)
                or not isinstance(uses_remaining, int)
                or uses_remaining < 1
            ):
                raise ValidationError(
                    "capability uses_remaining must be a positive integer"
                )
            selected["uses_remaining"] = uses_remaining
        max_depth = value.get("max_delegation_depth")
        if max_depth is not None:
            if isinstance(max_depth, bool) or not isinstance(max_depth, int):
                raise ValidationError("max_delegation_depth must be a non-negative integer")
            if max_depth < 0:
                raise ValidationError("max_delegation_depth must be a non-negative integer")
            selected["max_delegation_depth"] = max_depth
        return selected

    def _require_spec_covered(
        self,
        allowed: Iterable[dict[str, Any]],
        requested: dict[str, Any],
        *,
        label: str,
    ) -> None:
        if any(self.capabilities.spec_covers(parent, requested) for parent in allowed):
            return
        raise CapabilityDenied(
            f"{label} exceeds task authority manifest: {requested['resource']} rights={requested['rights']}"
        )

    def _missing_required_capabilities(
        self,
        manifest: TaskAuthorityManifest,
    ) -> list[dict[str, Any]]:
        return [
            dict(required)
            for required in manifest.required_capabilities
            if not any(
                self.capabilities.spec_covers(authorized, required)
                for authorized in manifest.authorized_capabilities
            )
        ]

    def _require_budget_attenuated(self, parent: dict[str, Any], child: dict[str, Any]) -> None:
        for key, parent_value in parent.items():
            child_value = child.get(key)
            if parent_value is not None and (
                child_value is None or child_value > parent_value
            ):
                raise CapabilityDenied(f"derived child resource budget exceeds parent manifest: {key}")

    def _require_requestable_capabilities_attenuated(
        self,
        parent: TaskAuthorityManifest,
        child_policy: dict[str, Any],
    ) -> None:
        parent_requestable = self._normalize_requestable_specs(
            parent.approval_policy.get("requestable_capabilities", [])
        )
        child_requestable = self._normalize_requestable_specs(
            child_policy.get("requestable_capabilities", [])
        )
        allowed = [*parent.authorized_capabilities, *parent_requestable]
        for spec in child_requestable:
            self._require_spec_covered(
                allowed,
                spec,
                label="derived child requestable capability",
            )

    def _require_effects_attenuated(
        self,
        parent: list[str] | None,
        child: list[str] | None,
    ) -> None:
        # An unrestricted parent may be narrowed to any concrete ceiling,
        # including deny-all. Every concrete parent rejects unrestricted
        # children, and an empty parent can only derive another empty ceiling.
        if parent is None:
            return
        if child is None or any(
            not any(self._effect_pattern_covers(parent_pattern, child_pattern) for parent_pattern in parent)
            for child_pattern in child
        ):
            raise CapabilityDenied("derived child effect ceiling exceeds parent manifest")

    @staticmethod
    def _effect_pattern_covers(parent: str, child: str) -> bool:
        if parent == "*":
            return True
        if parent == child:
            return True
        if not parent.endswith(".*"):
            return False
        parent_prefix = parent[:-1]
        if child.endswith(".*"):
            return child[:-1].startswith(parent_prefix)
        return child.startswith(parent_prefix)

    @staticmethod
    def _require_policy_mapping_attenuated(
        parent: dict[str, Any],
        child: dict[str, Any],
        *,
        label: str,
        ignored_keys: set[str] | None = None,
    ) -> None:
        ignored = ignored_keys or set()
        extra = sorted(set(child) - set(parent) - ignored)
        if extra:
            raise CapabilityDenied(
                f"derived child {label} cannot add policy keys outside parent ceiling: {extra}"
            )
        for key, value in parent.items():
            if key in ignored:
                continue
            if child.get(key) != value:
                raise CapabilityDenied(
                    f"derived child {label} cannot replace parent policy key: {key}"
                )

    @classmethod
    def _require_expiry_attenuated(cls, parent: str | None, child: str | None) -> None:
        if parent is None:
            return
        if child is None or cls._expiry_datetime(child) > cls._expiry_datetime(parent):
            raise CapabilityDenied("derived child manifest expiry exceeds parent manifest")

    @staticmethod
    def _expiry_datetime(value: str) -> datetime:
        selected = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return selected if selected.tzinfo is not None else selected.replace(tzinfo=timezone.utc)

    @classmethod
    def _bounded_expiry(
        cls,
        ceiling: str | None,
        requested: str | None,
    ) -> str | None:
        if ceiling is None:
            return requested
        if requested is None:
            return ceiling
        return (
            requested
            if cls._expiry_datetime(requested) <= cls._expiry_datetime(ceiling)
            else ceiling
        )

    @classmethod
    def _resolve_resource_budget(
        cls,
        payload: Mapping[str, Any],
        resource_budget: ResourceBudget | dict[str, Any] | None,
    ) -> dict[str, Any]:
        selected = cls._budget_dict(resource_budget)
        if "resource_budget" not in payload:
            return selected
        ceiling = cls._budget_dict(payload.get("resource_budget"))
        if resource_budget is None:
            return ceiling
        for key, ceiling_value in ceiling.items():
            if ceiling_value is None:
                continue
            current = selected.get(key)
            if current is None or ceiling_value < current:
                selected[key] = ceiling_value
        return selected

    @staticmethod
    def _budget_dict(value: ResourceBudget | dict[str, Any] | None) -> dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, ResourceBudget):
            selected = {
                name: getattr(value, name)
                for name in value.__dataclass_fields__
            }
            try:
                ResourceBudget(**selected)
            except (TypeError, ValueError) as exc:
                raise ValidationError(f"invalid resource_budget: {exc}") from exc
            return selected
        if not isinstance(value, dict):
            raise ValidationError("resource_budget must be an object")
        try:
            ResourceBudget(**value)
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"invalid resource_budget: {exc}") from exc
        return dict(value)

    @staticmethod
    def _mapping(value: Any, label: str) -> dict[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValidationError(f"{label} must be an object")
        return dict(value)

    @staticmethod
    def _reject_unknown_fields(
        value: Mapping[str, Any],
        allowed: frozenset[str],
        *,
        label: str,
    ) -> None:
        unknown = sorted(str(field) for field in value if field not in allowed)
        if unknown:
            raise ValidationError(f"{label} contains unsupported fields: {unknown}")

    @classmethod
    def _optional_expiry(cls, value: Any, label: str) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValidationError(f"{label} must be an ISO-8601 datetime string")
        selected = value.strip()
        if not selected:
            raise ValidationError(f"{label} must be non-empty")
        try:
            cls._expiry_datetime(selected)
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"{label} must be an ISO-8601 datetime") from exc
        return selected

    @staticmethod
    def _capability_boolean(
        value: Mapping[str, Any],
        field: str,
        *,
        default: bool,
    ) -> bool:
        selected = value[field] if field in value else default
        if not isinstance(selected, bool):
            raise ValidationError(
                f"authority manifest capability entry {field} must be a boolean"
            )
        return selected

    @staticmethod
    def _effect_classes(value: Any) -> list[str] | None:
        if value is None:
            return None
        if not isinstance(value, list):
            raise ValidationError("permitted_effects must be a list")
        selected: set[str] = set()
        for item in value:
            if not isinstance(item, str):
                raise ValidationError("permitted_effects entries must be strings")
            effect_class = item.strip()
            if not effect_class:
                raise ValidationError("permitted_effects entries must be non-empty strings")
            if "*" in effect_class and not (
                effect_class == "*"
                or (
                    effect_class.endswith(".*")
                    and effect_class.count("*") == 1
                    and len(effect_class) > 2
                )
            ):
                raise ValidationError(
                    "permitted_effects wildcards must be '*' or one terminal '.*'"
                )
            selected.add(effect_class)
        return sorted(selected)

    @staticmethod
    def _effect_matches(pattern: str, selected: str) -> bool:
        return pattern == "*" or pattern == selected or (pattern.endswith(".*") and selected.startswith(pattern[:-1]))

    @staticmethod
    def _template_payload(manifest: TaskAuthorityManifest) -> dict[str, Any]:
        return {
            "authorized_capabilities": manifest.authorized_capabilities,
            "permitted_effects": manifest.permitted_effects,
            "resource_budget": manifest.resource_budget,
            "approval_policy": manifest.approval_policy,
            "data_flow_policy": manifest.data_flow_policy,
            "expires_at": manifest.expires_at,
            "issued_by": manifest.issued_by,
            "metadata": manifest.metadata,
        }

    @staticmethod
    def _hash(manifest: TaskAuthorityManifest) -> str:
        payload = {
            key: value
            for key, value in manifest.__dict__.items()
            if key != "manifest_hash"
        }
        payload["permitted_effects"] = encode_permitted_effects_policy(
            manifest.permitted_effects
        )
        return hashlib.sha256(dumps(payload).encode("utf-8")).hexdigest()

    def _insert_hashed_manifest(
        self,
        manifest: TaskAuthorityManifest,
    ) -> TaskAuthorityManifest:
        selected = replace(manifest, manifest_hash=self._hash(manifest))
        self.store.insert_authority_manifest(selected)
        return selected

    @staticmethod
    def _legacy_hash(manifest: TaskAuthorityManifest) -> str:
        payload = {
            key: value
            for key, value in manifest.__dict__.items()
            if key
            not in {
                "manifest_hash",
                "permitted_effects_policy_schema_version",
            }
        }
        if manifest.permitted_effects is None:
            payload["permitted_effects"] = []
        return hashlib.sha256(dumps(payload).encode("utf-8")).hexdigest()

    @staticmethod
    def _require_live(manifest: TaskAuthorityManifest) -> None:
        if manifest.expires_at is None:
            return
        expires = datetime.fromisoformat(manifest.expires_at.replace("Z", "+00:00"))
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)
        if expires <= datetime.now(timezone.utc):
            raise CapabilityDenied(f"task authority manifest expired: {manifest.manifest_id}")
