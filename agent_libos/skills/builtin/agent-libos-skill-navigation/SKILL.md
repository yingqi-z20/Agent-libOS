---
name: agent-libos-skill-navigation
description: Discover, activate, inspect, and unload Agent Skills. Use when a task needs tools or domain guidance that are not currently visible, or when loaded Skill context should be examined or removed.
allowed-tools: discover_skills activate_skill read_skill_resource unload_skill
---
# Navigate Skills

## Workflow

1. Check the available built-in Skill catalog and activate the smallest Skill matching the current intent.
2. Use `discover_skills` when the exact Skill is unknown or a registered workspace/global Skill may be relevant.
3. After activation, inspect the newly visible tool schemas before calling them.
4. Use `read_skill_resource` only for a resource named by an already loaded Skill. Unload guidance that is no longer useful.

## Boundaries and safety

- Activating a built-in Skill changes model visibility and instructions only. It never grants Capability authority.
- Do not activate adjacent Skills merely to bypass a denied operation. Inspect or request the exact missing authority instead.
- Registered Skills can add static or JIT tools and retain their normal trust and Capability checks.

## Verify

Confirm the requested Skill is active and the intended tools, not unrelated categories, became visible.
