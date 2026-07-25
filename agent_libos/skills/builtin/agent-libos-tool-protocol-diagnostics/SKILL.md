---
name: agent-libos-tool-protocol-diagnostics
description: Round-trip a small top-level JSON object through echo to isolate model/tool argument and result plumbing. Use for semantic transport diagnostics; not for domain validation, authorization, provider health, JIT registration, arbitrary persistence, or application work.
allowed-tools: echo
---
# Diagnose tool plumbing

Use this Skill to answer a narrow question: can this model-facing tool-call path carry a small JSON object to `echo` and return its values? It is deliberately weaker than an end-to-end test of any real tool.

## Tool guide

### `echo`

- Input is a top-level JSON object. The schema permits extra fields and performs no field-specific semantic validation.
- Output is the object accepted by the Echo tool. Compare JSON value types and nesting, not textual wire bytes, whitespace, number spelling, or key order.
- Use a tiny, non-sensitive sentinel containing only the types at issue, for example strings, numbers, booleans, null, lists, and nested objects. Global argument and normalized-result persistence limits apply, and result normalization can serialize the same logical value into structured data and textual content.

`action` is a reserved top-level protocol key and is the one important exception to "unchanged." Before tool dispatch, the model tool-call adapter trims the function `name`, parses `arguments` as an object, removes any top-level `action` argument, and injects the selected function name as the internal `action`. The real function name wins over an `action` supplied in arguments; only a missing function name may use that key as a legacy fallback. Consequently, never use a top-level `action` field as an Echo round-trip sentinel. Nested keys named `action` are ordinary JSON data.

`arguments` supplied as `null`, absent, or an empty string normalize to `{}`. A JSON string must decode to an object; arrays, scalars, booleans, and malformed JSON are rejected before `echo` runs. A nonempty function name is normally required.

## Recommended workflow

1. State the exact suspected layer and property, such as nested-array preservation or boolean-vs-string coercion.
2. Call `echo` once through the same model-facing provider path with the smallest top-level object that tests that property. Avoid `action`, credentials, tokens, personal data, large blobs, and domain records.
3. Compare the returned JSON semantically and record whether `echo` actually dispatched. If parsing or name normalization fails first, classify it as a protocol-frame failure rather than an Echo mismatch.
4. After the isolated check, call the real target tool with a small schema-valid, safe case. Its schema, visibility, authority, approval, primitive, provider, side effects, and output bounds are independent.

For process-local JIT tools, distinguish two model projections precisely:

- In **direct** projection, invoke the registered JIT tool's actual function name with arguments matching its stored input schema.
- In **multiplexed** projection, invoke the model-facing pseudo-function `run_jit_tool` with exactly `{"tool_name":"registered_name","arguments":{...}}`. The field is `tool_name`, not `name`. A direct call to the registered JIT name is rejected in this mode, and the nested `arguments` object is validated against that JIT tool's schema.

`run_jit_tool` is not a real process tool: it is never installed in the process tool table, cannot be called through `runtime.tools.call`, and must not appear in a Skill's `allowed-tools`. It only routes model calls to a process-local JIT tool installed in the process table under multiplexed mode. Multiplexed projection deliberately omits automatic catalog-derived JIT names/schemas and loaded-Skill JIT catalog/source entries; it does not erase a name or contract explicitly written in trusted Image/Skill instructions. The exact name and contract must therefore already be known from those instructions, retained task-local authoring evidence, or an exact loaded-snapshot resource path supplied by trusted instructions/evidence. `read_skill_resource` can read that known path because multiplexing limits discovery rather than loaded-resource access; it cannot list paths. If no such evidence exists, do not guess a name, arguments, or filename; re-establish the contract through the authorized authoring workflow or stop.

Echo success does not prove JIT availability. After an Echo transport check, validate JIT plumbing by invoking the real registered tool in direct mode or the exact multiplexer envelope in multiplexed mode, first with a small valid case and then with a deliberately schema-invalid case that must be rejected before execution.

## Failure and recovery

- Missing/empty function name without a fallback, malformed JSON, and non-object arguments fail in protocol normalization before Echo execution. An Echo result cannot diagnose those failures because no Echo result exists; use provider/repair/audit evidence.
- Unknown or hidden tool names are projection or process-tool-table failures. Echo success says nothing about them.
- An oversized argument is rejected before the normal tool-called event. An oversized read-only Echo result fails without a persisted ToolResult. Reduce the diagnostic to the minimum sentinel rather than raising limits or transferring payloads through Echo.
- Successful Echo calls still create normal audit/event/result evidence. Echo is not a secret-free scratch channel or proof of arbitrary application persistence.
- Echo performs no permission grant, external communication, filesystem access, provider request, or domain-state mutation. It cannot verify authorization, endpoint credentials, provider health, retries, idempotency, application semantics, JIT sandbox behavior, syscalls, or output-schema enforcement.
- Do not repeat Echo after the tested semantic property is already known. Repetition adds audit and context volume without improving domain evidence.

## Completion evidence

A protocol diagnostic is complete only when it records:

- the exact model-facing function name and JSON object used, excluding secrets and the reserved top-level `action` key;
- whether the call reached `echo` or failed earlier during frame normalization;
- a semantic expected-versus-observed comparison, including JSON types and nesting;
- argument/result truncation or size-limit status;
- for JIT, the projection mode and either the direct registered name or exact `run_jit_tool` envelope, followed by real-tool valid-case output and invalid-input rejection;
- the remaining layers not tested by Echo.

Stop after isolating the transport property. Domain completion must come from the actual target tool and its authoritative readback, not from `echo`.
