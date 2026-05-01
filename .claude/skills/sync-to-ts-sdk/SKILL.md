---
name: sync-to-ts-sdk
description: Sync Python SDK changes to TypeScript SDK and create a PR.
argument-hint: "[PR_URL] [--no-commit]"
disable-model-invocation: true
allowed-tools: Bash, Read, Write, Edit, Glob, Grep, Task
---

# Sync Python SDK Changes to TypeScript SDK

This skill analyzes changes from the Python SDK (`twilio-agent-connect-python`) and generates corresponding changes for the TypeScript SDK (`twilio-agent-connect-typescript`), then creates a PR.

## Input Sources

The skill accepts the following arguments (can be combined):

1. **GitHub PR URL**: `https://github.com/twilio/twilio-agent-connect-python/pull/123`
2. **No commit flag**: `--no-commit` (makes and stages changes, but skips commit/push/PR)

**Examples:**

- `/sync-to-ts-sdk` - Current branch, creates PR
- `/sync-to-ts-sdk --no-commit` - Current branch, stages changes only
- `/sync-to-ts-sdk https://github.com/.../pull/123` - From PR URL, creates PR
- `/sync-to-ts-sdk https://github.com/.../pull/123 --no-commit` - From PR URL, stages changes only

## Configuration

**Repositories:**

- Python SDK (source): Current working directory (this repo)
- TypeScript SDK (target): Clone to `~/.claude/cache/sync-to-ts-sdk/twilio-agent-connect-typescript` (user's home directory)
- GitHub org: `twilio`

## Determine Input Mode

Parse arguments to determine the source of changes and mode:

```
Arguments: $ARGUMENTS (the full argument string)

1. Check for NO_COMMIT flag:
   If arguments contain "--no-commit":
     -> NO_COMMIT: true
   Else:
     -> NO_COMMIT: false

2. Check for PR URL:
   If arguments contain a URL matching "github.com" and "/pull/":
     -> PR URL MODE: Fetch changes from the specified PR
     -> Extract: org, repo, PR number from URL
   Else:
     -> CURRENT BRANCH MODE: Use current branch in local repo
```

**Parsing Examples:**

- `""` → Current branch, NO_COMMIT=false
- `"--no-commit"` → Current branch, NO_COMMIT=true
- `"https://github.com/.../pull/123"` → PR URL mode, NO_COMMIT=false
- `"https://github.com/.../pull/123 --no-commit"` → PR URL mode, NO_COMMIT=true
- `"--no-commit https://github.com/.../pull/123"` → PR URL mode, NO_COMMIT=true

## Workflow

### Phase 0: Preflight — Verify GitHub CLI

Before doing anything else, verify that the `gh` CLI is installed and authenticated with access to the target repo:

```bash
gh repo view twilio/twilio-agent-connect-typescript --json name --jq '.name'
```

- If this command succeeds (prints `twilio-agent-connect-typescript`), proceed to Phase 1.
- If it fails for **any reason** (gh not installed, not authenticated, no repo access, network error, etc.), **STOP the skill immediately** and tell the user:
  > This skill requires the GitHub CLI (`gh`) to be installed and authenticated with access to `twilio/twilio-agent-connect-typescript`.
  >
  > Run `gh auth status` to check your authentication, or `gh auth login` to authenticate.

Do not continue to any subsequent phase if this check fails.

### Phase 1: Setup TypeScript SDK Repository

Clone or update the TypeScript SDK in the user's cache directory. **Always hard reset to remote main** to ensure a clean state, but warn the user first if there are uncommitted changes.

**Step 1a: Define the cache directory and check if repo exists**

```bash
TS_SDK_DIR="$HOME/.claude/cache/sync-to-ts-sdk/twilio-agent-connect-typescript"
echo "TS_SDK_DIR=$TS_SDK_DIR"
[ -d "$TS_SDK_DIR" ] && echo "REPO_EXISTS=true" || echo "REPO_EXISTS=false"
```

**Step 1b: If repo exists, check for uncommitted changes**

```bash
cd "$TS_SDK_DIR" && git status --porcelain
```

**If the output is non-empty** (there are uncommitted changes):

1. Show the user the git status output and the full repo path
2. Use `AskUserQuestion` to ask:
   - Question: "The TypeScript SDK repo at `<path>` has uncommitted changes (shown above). Continuing will discard them. Do you want to proceed?"
   - Options: "Yes, discard and continue" / "No, abort"
3. **If user chooses "No, abort"**: STOP the skill immediately and inform them that the sync was cancelled.

**Step 1c: Update or clone the repository**

Only proceed with this step if either:
- The repo doesn't exist (REPO_EXISTS=false), OR
- The repo has no uncommitted changes, OR
- The user approved discarding changes

**If repo exists (update and reset):**

```bash
cd "$TS_SDK_DIR"
git fetch origin
git checkout main
git reset --hard origin/main
git clean -fd
```

**If repo doesn't exist (clone fresh):**

```bash
mkdir -p "$HOME/.claude/cache/sync-to-ts-sdk"
gh repo clone twilio/twilio-agent-connect-typescript "$TS_SDK_DIR"
```

### Phase 2: Analyze Python SDK Changes

**If PR URL MODE:**

Fetch PR details and diff using GitHub CLI:

```bash
# Get PR metadata
gh pr view <PR_NUMBER> --repo twilio/twilio-agent-connect-python --json title,body,headRefName,baseRefName,files,url

# Get the diff
gh pr diff <PR_NUMBER> --repo twilio/twilio-agent-connect-python

# Get list of changed files
gh pr view <PR_NUMBER> --repo twilio/twilio-agent-connect-python --json files --jq '.files[].path'
```

Store:

- `SOURCE_BRANCH`: PR head branch name
- `SOURCE_PR_URL`: The PR URL
- `SOURCE_PR_TITLE`: PR title
- `SOURCE_PR_NUMBER`: PR number

**If CURRENT BRANCH MODE:**

Get the current branch name and changes vs main:

```bash
# Navigate to git root first
cd "$(git rev-parse --show-toplevel)"

# Get current branch
git rev-parse --abbrev-ref HEAD

# Get list of changed files
git diff --name-only main...HEAD

# Get detailed diff
git diff main...HEAD
```

Store:

- `SOURCE_BRANCH`: Current branch name
- `SOURCE_PR_URL`: Check if PR exists with `gh pr view --json url` (may be empty)
- `SOURCE_PR_TITLE`: Empty or from existing PR
- `SOURCE_PR_NUMBER`: Empty or from existing PR

### Phase 3: Explore TypeScript SDK and Create Plan

**First, explore the TypeScript SDK to understand its structure:**

1. **Read the TypeScript SDK's CLAUDE.md** at `$TS_SDK_DIR/.claude/CLAUDE.md` for project-specific guidance
2. **Launch Explore agents** to find equivalent modules and understand patterns used in the TypeScript SDK
3. **Search for similar type/function names** to find where concepts are implemented

**Then, for each changed Python file, determine:**

- What type of change (new file, modification, deletion)
- Which TypeScript file(s) it maps to (based on exploration)
- What the semantic change is (new endpoint, new model field, bug fix, etc.)

### Phase 4: Present Plan and Get User Approval

**IMPORTANT:** Before making any changes to the TypeScript SDK, present a detailed plan to the user and wait for approval.

Present the plan in this format:

```markdown
# Sync Plan: Python → TypeScript SDK

## Source

- Branch: `<branch-name>`
- PR: <url or N/A>
- Changed files: <count>

## Proposed Changes

### 1. <Python File Path>

- **Change type:** <new file | modified | deleted>
- **TypeScript target:** `<typescript file path>`
- **What will change:**
  - <bullet point description of each change>
  - <e.g., "Add new `getProfile` method with `traitGroups` parameter">
  - <e.g., "Add `ProfileResponse` type with `id`, `createdAt`, `traits` fields">

### 2. <Next Python File>

...

## Files to Create/Modify in TypeScript SDK

| Action | File                            |
| ------ | ------------------------------- |
| Modify | `<path discovered via explore>` |
| Create | `<path discovered via explore>` |

## Ready to proceed?

Reply **yes** to implement these changes, or provide feedback to adjust the plan.
```

**Wait for the user to reply with approval before proceeding to Phase 5.**

If the user provides feedback or requests changes to the plan, incorporate their feedback and present an updated plan.

### Phase 5: Implement TypeScript Changes

**Only proceed with this phase after user approval.**

1. Create a new branch in the TypeScript SDK:

```bash
cd "$TS_SDK_DIR"

# Create branch based on source
TS_BRANCH="sync/${SOURCE_BRANCH}"

git checkout -b "$TS_BRANCH"
```

2. For each Python change, generate the equivalent TypeScript code following the conventions and patterns discovered in Phase 3.

3. Create or update tests to cover the new/changed functionality.

### Phase 6: Verify Changes

Run verification checks using sub-agents to preserve context. First read `$TS_SDK_DIR/.claude/CLAUDE.md` for the correct commands, then launch each check in a sub-agent:

1. **Formatting** - Launch a sub-agent to run the formatter
2. **Linting** - Launch a sub-agent to run the linter
3. **Type checking** - Launch a sub-agent to run the type checker
4. **Tests** - Launch a sub-agent to run the test suite

Review the results from each sub-agent and fix any issues found before proceeding to Phase 7.

### Phase 7: Create PR in TypeScript SDK

1. Stage all changes:

```bash
git add .
```

**If NO_COMMIT is true:**
Stop here. Output a message noting:

- Changes are staged in branch `sync/${SOURCE_BRANCH}`
- Path: `~/.claude/cache/sync-to-ts-sdk/twilio-agent-connect-typescript`
- User can inspect with `git status` and `git diff --staged`

**If NO_COMMIT is false:**
Continue with commit and PR creation:

2. Commit changes:

```bash
git commit -m "$(cat <<'EOF'
Sync: <summary of Python changes>

Synced from Python SDK branch: ${SOURCE_BRANCH}
Python PR: ${SOURCE_PR_URL}
EOF
)"

git push -u origin "$TS_BRANCH"
```

3. Read the PR template from the TypeScript SDK and create PR:

**IMPORTANT:** Before creating the PR, read the PR template file at:

```
~/.claude/cache/sync-to-ts-sdk/twilio-agent-connect-typescript/.github/PULL_REQUEST_TEMPLATE.md
```

Use the template structure to create the PR body:

- Fill in the Summary section with the sync details and Python SDK reference
- Check the appropriate "Type of Change" checkbox based on the Python changes
- Check "Python SDK PR created" in the SDK Parity section and include the source PR link
- Add a footer noting this was generated by the `/sync-to-ts-sdk` skill

```bash
gh pr create \
  --draft \
  --title "Synced from Python: <summary>" \
  --body "<PR body following the template format>"
```

### Phase 8: Output Report

Generate a summary report with the following structure:

**Report Template:**

```
# SDK Sync Report

## Source (Python SDK)

- Mode: <PR URL | Current Branch>
- Branch: `<branch-name>`
- PR: <url or N/A>
- Changed files: <count>

## Changes Detected

| Python File        | Change Type | TypeScript Target | Status |
| ------------------ | ----------- | ----------------- | ------ |
| <python file path> | Modified    | <ts file path>    | Synced |

## Detailed Changes

### 1. <File Name>

**Python change:**
(show diff)

**TypeScript equivalent:**
(show diff)

## Target (TypeScript SDK)

- Branch: `sync/<branch-name>`
- PR: <url or N/A (N/A if NO COMMIT mode)>
```

## Execution

Now proceed through each phase:

1. **Determine Input Mode**: Parse arguments (PR URL vs current branch, --no-commit flag)
2. **Phase 0**: Verify `gh` CLI access to target repo; STOP if it fails
3. **Phase 1**: Setup/update TypeScript SDK repository in cache (check for uncommitted changes first)
4. **Phase 2**: Analyze Python SDK changes from PR URL or current branch vs main
5. **Phase 3**: Explore TypeScript SDK and create a detailed plan
6. **Phase 4**: Present the plan to the user and wait for approval
7. **Phase 5**: After approval, create branch and implement TypeScript changes
8. **Phase 6**: Verify changes using sub-agents (format, lint, type check, test)
9. **Phase 7**: Stage changes; if not --no-commit, commit and create PR
10. **Phase 8**: Output final sync report
