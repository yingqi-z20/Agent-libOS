from __future__ import annotations

from pydantic import BaseModel, Field

from agent_libos.config import DEFAULT_CONFIG
from agent_libos.models.exceptions import ValidationError as LibOSValidationError
from agent_libos.tools.base import SyncAgentTool, ToolContext, ToolErrorCode, ToolExecutionError, ToolPolicy

_IMAGE_DEFAULTS = DEFAULT_CONFIG.image
_TOOL_DEFAULTS = DEFAULT_CONFIG.tools


class LoadImagePackageArgs(BaseModel):
    path: str = Field(
        description=(
            "Workspace-relative package directory containing IMAGE.yaml; registration does not exec the current process."
        )
    )
    replace: bool = Field(
        default=False,
        description="Replace an existing image with the same id; this requires exact image admin authority.",
    )


class LoadImagePackageOutput(BaseModel):
    image_id: str
    name: str
    version: str
    source_path: str
    replaced: bool
    default_tools: list[str]
    package_jit_tools: list[str]
    boot_kind: str
    artifact_id: str
    package_sha256: str
    required_capabilities_count: int
    required_modules_count: int


class CommitCheckpointToImageArgs(BaseModel):
    checkpoint_id: str = Field(
        description="Checkpoint id whose reconstructable internal state becomes an immutable image artifact."
    )
    image_id: str = Field(description="Target AgentImage id; a new id requires exact image write authority.")
    name: str = Field(description="Human-readable image name.")
    version: str = Field(default="v0", description="Image version.")
    replace: bool = Field(
        default=False,
        description="Replace an existing target image; this requires exact image admin authority.",
    )
    metadata: dict[str, object] = Field(default_factory=dict, description="Optional image metadata.")


class CommitCheckpointToImageOutput(BaseModel):
    image_id: str
    name: str
    version: str
    replaced: bool
    boot_kind: str
    artifact_id: str
    artifact_sha256: str
    required_capabilities_count: int
    required_modules_count: int


class LoadImagePackageTool(SyncAgentTool[LoadImagePackageArgs]):
    name = "load_image_package"
    description = (
        "Read an AgentImage package directory from the workspace and register it for later process launch or exec. "
        "Registration alone does not change the current process image, tool table, memory, or capabilities. "
        "The filesystem primitive enforces file read authority. Registering a new id requires exact image write authority; "
        "replace=true requires exact image admin authority."
    )
    args_schema = LoadImagePackageArgs
    output_schema = LoadImagePackageOutput
    policy = ToolPolicy(
        side_effects=True,
        idempotent=False,
        declared_permissions={"filesystem.read", "image.write", "image.admin"},
        timeout_s=_TOOL_DEFAULTS.standard_timeout_s,
    )
    tags = ["image", "registry", "package", "side_effect"]

    def run(self, args: LoadImagePackageArgs, ctx: ToolContext) -> LoadImagePackageOutput:
        runtime = ctx.runtime
        if runtime is None:
            raise ToolExecutionError("Runtime is unavailable.", code=ToolErrorCode.EXECUTION_ERROR)
        try:
            result = runtime.image_registry.register_from_workspace_package(
                ctx.pid,
                args.path,
                replace=args.replace,
            )
        except LibOSValidationError as exc:
            raise ToolExecutionError(
                str(exc),
                code=ToolErrorCode.VALIDATION_ERROR,
                details={"path": args.path},
            ) from exc
        image = result.image
        return LoadImagePackageOutput(
            image_id=image.image_id,
            name=image.name,
            version=image.version,
            source_path=result.source or args.path,
            replaced=result.replaced,
            default_tools=list(image.default_tools),
            package_jit_tools=list(image.metadata.get("package_jit_tools", [])),
            boot_kind=image.boot.get("kind", "fresh"),
            artifact_id=str(image.boot.get("artifact_id", "")),
            package_sha256=str(image.boot.get("package_sha256", "")),
            required_capabilities_count=len(image.required_capabilities),
            required_modules_count=len(image.required_modules),
        )


class CommitCheckpointToImageTool(SyncAgentTool[CommitCheckpointToImageArgs]):
    name = "commit_checkpoint_to_image"
    description = (
        "Publish a process checkpoint as a checkpoint-derived AgentImage for later launch or exec. "
        "This does not restore the checkpoint or change the current process. The image captures reconstructable "
        "state only; it neither clones external provider state nor grants external capabilities. "
        "A new id requires exact image write authority; replace=true requires exact image admin authority."
    )
    args_schema = CommitCheckpointToImageArgs
    output_schema = CommitCheckpointToImageOutput
    policy = ToolPolicy(
        side_effects=True,
        idempotent=False,
        declared_permissions={"checkpoint.read", "image.write", "image.admin"},
        timeout_s=_TOOL_DEFAULTS.standard_timeout_s,
    )
    tags = ["image", "checkpoint", "commit", "self_evolution", "high_risk"]

    def run(self, args: CommitCheckpointToImageArgs, ctx: ToolContext) -> CommitCheckpointToImageOutput:
        runtime = ctx.runtime
        if runtime is None:
            raise ToolExecutionError("Runtime is unavailable.", code=ToolErrorCode.EXECUTION_ERROR)
        try:
            result = runtime.image_registry.commit_from_checkpoint(
                actor=ctx.pid,
                checkpoint_id=args.checkpoint_id,
                image_id=args.image_id,
                name=args.name,
                version=args.version,
                replace=args.replace,
                metadata=dict(args.metadata),
                require_capability=True,
            )
        except LibOSValidationError as exc:
            raise ToolExecutionError(str(exc), code=ToolErrorCode.VALIDATION_ERROR) from exc
        image = result.image
        return CommitCheckpointToImageOutput(
            image_id=image.image_id,
            name=image.name,
            version=image.version,
            replaced=result.replaced,
            boot_kind=image.boot.get("kind", "fresh"),
            artifact_id=str(image.boot.get("artifact_id", "")),
            artifact_sha256=str(image.boot.get("artifact_sha256", "")),
            required_capabilities_count=len(image.required_capabilities),
            required_modules_count=len(image.required_modules),
        )
