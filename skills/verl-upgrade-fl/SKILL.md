---
name: verl-upgrade-fl
version: 0.1.0
description: >
  Merge upstream verl releases into the verl-FL fork, resolve conflicts preserving
  the multi-chip platform abstraction layer, run lint and CPU tests, then open a PR.
triggers:
  - /verl-upgrade-fl
  - upgrade verl
  - merge upstream verl
  - sync verl upstream
parameters:
  - name: upstream_repo
    type: string
    required: false
    default: "https://github.com/volcengine/verl.git"
    description: Upstream verl repository URL
  - name: fork_folder
    type: string
    required: false
    default: ""
    description: Local path to verl-FL clone. If empty, clones from origin.
  - name: target_tag
    type: string
    required: false
    default: ""
    description: >
      Upstream release tag to merge (e.g. v0.8.0). If empty, auto-detects the
      latest release tag.
  - name: fork_repo
    type: string
    required: false
    default: "physics31415926/verl-FL"
    description: GitHub owner/repo for the fork (used for PR creation)
---

# verl-upgrade-fl

Merge upstream [verl](https://github.com/volcengine/verl) releases into the
**verl-FL** fork, resolve conflicts while preserving the multi-chip platform
abstraction layer, and submit a PR with full test validation.

## When to Use

- A new upstream verl release is available and verl-FL needs to catch up.
- User says "upgrade verl", "merge upstream verl", "sync verl upstream", etc.

## Background

verl-FL extends upstream verl with:

- **Multi-chip platform abstraction** (`verl/plugin/platform/`) — hardware-agnostic
  device management for NVIDIA, Ascend NPU, MetaX, etc.
- **FL-specific backends** — `vllm-plugin-fl`, `te-fl` integrations
- **Device proxy layer** (`verl/utils/device.py`) — delegates `torch.cuda.*` calls
  through `get_platform()` so non-CUDA chips work transparently

Conflicts concentrate in device/platform-related files where both sides evolve
independently.

## Critical Rules

1. **Never replace `get_platform()` delegation with `torch.cuda.*` hardcoding.**
   The platform abstraction is the core value of verl-FL.
2. **Preserve verl-FL's `is_device_available()`, `get_visible_devices_keyword()`,
   `get_resource_name()`** and other platform-aware functions.
3. **Keep upstream's new functions/parameters** — merge them into the platform
   delegation pattern (e.g. wrap new upstream functions with
   `_get_platform_manager()` calls).
4. **Incremental verification** — after resolving each conflict file, verify no
   conflict markers remain before moving on.
5. **Run ruff lint+format before committing** — the repo uses ruff as pre-commit.

## Upgrade Procedure

### Step 1: Setup

```
1. Clone verl-FL (or use existing clone at `fork_folder`)
2. Add upstream remote if not present:
     git remote add upstream <upstream_repo>
3. Fetch upstream tags:
     git fetch upstream --tags
4. Determine target tag:
   - If `target_tag` is set → use it
   - Otherwise → pick the latest vX.Y.Z tag from upstream
5. Identify current base version from `verl/version/version`
6. Create merge branch:
     git checkout -b merge-upstream-<tag> main
```

### Step 2: Merge

```
git merge <tag> --no-edit
```

If auto-merge succeeds completely → skip to Step 4.
Otherwise → proceed to Step 3.

### Step 3: Resolve Conflicts

Identify conflict files:

```
git diff --check | grep "leftover conflict marker"
```

For each conflicted file, apply the appropriate resolution strategy:

| File Pattern | Strategy |
|---|---|
| `verl/version/version` | Take upstream version |
| `verl/utils/device.py` | Keep verl-FL's `get_platform()` delegation; add upstream's new functions wrapped in `_get_platform_manager()` calls |
| `verl/utils/profiler/` | Keep verl-FL's `get_platform().profiler_start/stop()` with enable/rank/step guards |
| `verl/workers/rollout/sglang_rollout/` | Use upstream's generic `visible_devices_keyword`; keep verl-FL's `is_device_available()` |
| `tests/special_sanity/check_device_api_usage.py` | Combine both sides' whitelist entries |
| `verl/workers/fsdp_workers.py` | Usually auto-resolves; verify no markers |
| Other files | Evaluate case-by-case: prefer upstream logic + verl-FL platform hooks |

After resolving each file:

```
git diff --check <file>   # verify no conflict markers (CRLF warnings are OK)
```

### Step 4: Verify & Commit Merge

```
# Ensure no conflict markers remain anywhere
git diff --check

# Stage and commit
git add -A
git commit --no-edit   # uses the auto-generated merge message
```

### Step 5: Lint

```
ruff check --fix <conflicted_files>
ruff format <conflicted_files>
```

If ruff made changes, commit them:

```
git add -A
git commit -m "[misc] style: fix ruff lint after upstream merge"
```

### Step 6: Test

Run CPU tests in this order. Stop and fix if any fail.

```python
# 1. Device API sanity check
python tests/special_sanity/check_device_api_usage.py -d verl

# 2. Platform abstraction tests
python tests/plugin/test_device_on_cpu.py
python tests/plugin/test_platform_abstraction.py

# 3. Engine and FL env tests
python tests/plugin/test_engine_registry_on_cpu.py
python tests/plugin/test_fl_env_manager_on_cpu.py

# 4. Utility tests (all *_on_cpu.py files)
python -m pytest tests/utils/ -k "on_cpu" -x

# Skip: test_timeout_decorator_cpu.py (Unix signals, incompatible with Windows)
```

Common issues to watch for:

- **Missing proxy functions in `device.py`**: if a test fails with `ImportError`
  for a function from `verl.utils.device`, check if upstream added new functions
  that need platform delegation wrappers. Add them following the existing pattern:
  ```python
  def new_function(*args, **kwargs):
      return _get_platform_manager().new_function(*args, **kwargs)
  ```

### Step 7: Push & PR

```
git push origin merge-upstream-<tag>
```

Create PR via `gh`:

```
gh pr create \
  --repo <fork_repo> \
  --base main \
  --head merge-upstream-<tag> \
  --title "[misc] chore: merge upstream verl <tag>" \
  --body "<conflict resolution table>"
```

PR body should include:

- Upstream tag merged
- Table of conflicted files with resolution summary
- Test results summary

### Step 8: Post-merge Cleanup

After PR is merged:

```
git checkout main
git pull origin main
git branch -d merge-upstream-<tag>
```

## Conflict Resolution Principles

1. **Version file** → always take upstream
2. **Device/platform files** → upstream logic + verl-FL abstraction layer
3. **Test whitelists** → union of both sides
4. **Worker files** → upstream structure + verl-FL device hooks
5. **Profiler files** → verl-FL's platform-aware profiling

When in doubt: start from upstream's version, then re-inject verl-FL's
platform-specific customizations. This is safer than patching upstream changes
into verl-FL's version.

## Examples

**Example 1: Standard release upgrade**
```
User: "upgrade verl to v0.8.0"
Actions:
  1. Fetch upstream tags, confirm v0.8.0 exists
  2. Create branch merge-upstream-v0.8.0
  3. git merge v0.8.0 --no-edit
  4. Resolve conflicts (device.py, version, etc.)
  5. ruff check+format, commit
  6. Run CPU tests, fix any import errors
  7. Push, create PR
Result: PR ready for review with all CPU tests passing
```

**Example 2: Auto-detect latest release**
```
User: "sync verl upstream"
Actions:
  1. Fetch upstream tags
  2. Detect latest: v0.9.0 (current fork base: v0.8.0)
  3. Same merge workflow as above
Result: Fork updated to latest upstream release
```
