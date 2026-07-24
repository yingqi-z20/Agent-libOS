---
name: agent-libos-agent-images
description: Register AgentImage packages, publish checkpoint-derived images, or replace the current process image. Use when creating reusable runtime configurations or deliberately changing process code/tool/prompt identity.
allowed-tools: load_image_package commit_checkpoint_to_image exec_process
---
# Manage AgentImages

## Workflow

1. Register a workspace image package when an audited definition should be available for later launch or exec.
2. Commit a checkpoint when reconstructable runtime state should become an immutable image artifact.
3. Use `exec_process` only with an exact target image ID already known from trusted context or returned by a preceding registration/publication, and only when the current PID should adopt it.
4. For registration/publication, verify the returned boot kind, tools, capability counts, and hashes. For exec, verify the reported new image and active tools.

## Boundaries and safety

- Registration and checkpoint commit do not change the current process. Exec does.
- These tools cannot inspect an arbitrary pre-existing image. Do not invent its contract from an ID.
- Images define requirements and visibility but never grant missing capabilities; preserve options cannot amplify authority.
- Provider state is neither packaged nor cloned. Replacing an existing image requires exact admin authority.

## Verify

After publication verify immutable identifiers/hashes. After exec, verify the new image and visible tools; if available, use Object Memory and authority inspection separately to check requested preservation.
