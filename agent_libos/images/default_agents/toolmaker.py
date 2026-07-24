from __future__ import annotations

from agent_libos.config import DEFAULT_CONFIG, AgentLibOSConfig
from agent_libos.models import AgentImage, PROMPT_MODE_LIBOS_DEFAULT


TOOLMAKER_AGENT_PROMPT = """
Role:
You are an Agent libOS toolmaker image. Produce, validate, and register small
Deno/TypeScript JIT tools that make repeated runtime work safer and clearer.

Operating contract:
- Follow the loaded agent-libos-jit-tool-authoring Skill for the detailed
  authoring workflow and tool-use guidance.
- Prefer an existing governed tool when it already models the operation. Never
  create a JIT tool merely to obtain broader authority or bypass approval.
- Treat generated source and test data as untrusted. Keep results bounded and
  preserve every primitive Capability and provider boundary.
- When the tool is registered and verified, or the blocker is clear, report one
  concise final user-facing result through human_output unless the goal
  explicitly requests machine-only output, then end with process_exit.
""".strip()


def build_toolmaker_agent_image(config: AgentLibOSConfig = DEFAULT_CONFIG) -> AgentImage:
    runtime_defaults = config.runtime
    return AgentImage(
        image_id="toolmaker-agent:v0",
        name="toolmaker-agent",
        version="v0",
        system_prompt=TOOLMAKER_AGENT_PROMPT,
        prompt_mode=PROMPT_MODE_LIBOS_DEFAULT,
        default_skills=["agent-libos-jit-tool-authoring"],
        default_tools=[
            "ask_human",
            "create_memory_object",
            "human_output",
            "inspect_capability",
            "list_capabilities",
            "process_exit",
            "propose_jit_tool",
            "read_memory_object",
            "read_process_messages",
            "receive_process_messages",
            "register_jit_tool",
            "request_permission",
            "validate_jit_tool",
        ],
        context_policy="plan_first",
        safety_profile="toolmaker",
        required_capabilities=[
            {"resource": runtime_defaults.default_human_resource, "rights": ["write"]},
        ],
        metadata={
            "role": "deno_jit_toolmaker",
            "source_contract": "import_free",
            "registration_contract": "validate_before_register",
        },
    )
