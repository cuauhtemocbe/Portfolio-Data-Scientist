# CLAUDE.md

Instruction guide for Claude Code when working in this repository.

---

## Available Skills

This repository includes the following Claude Code skills:

- **`/spec-driven-dev`**: Complete spec-driven development workflow (idea → spec → plan → tasks → implementation)
- **`/user-stories`**: Write and publish user stories with Gherkin acceptance criteria to GitHub/GitLab Issues
- **`/commit-writer`**: Generate conventional commits following project standards
- **`/testing`**: TDD workflow with mutation testing, coverage targets, and test quality validation
- **`/sonar-check`**: Code quality analysis with SonarQube (MCP-first or Docker fallback)
- **`/trivy-scan`**: Security scanning for vulnerabilities, secrets, IaC misconfigurations, and licenses

Use these skills proactively when relevant to the work at hand.

`meta-projects` (`.claude/skills/`) is the canonical source of truth for
these 6 skills. Pull fresh copies from there — `scripts/08_sync_skills.sh`
in that repo keeps every registered repo's copy in sync with it.

---

## Recommended Development Workflow

### For New Features

1. **Spec-Driven Development**: Use `/spec-driven-dev` to transform ideas into structured specifications
   - Phase 1 (Specify): Refine idea → create spec in `specs/{feature}.md`
   - Phase 2 (Plan): Create implementation plan → `specs/{feature}-plan.md`
   - Phase 3 (Tasks): Break into tasks → create GitHub/GitLab Issues with `/user-stories`
   - Phase 4 (Implement): Execute tasks with TDD using `/testing`

2. **Implementation**: Follow TDD cycle for each task
   - Write tests first (use Gherkin scenarios from user stories)
   - Run tests and verify they fail
   - Implement the feature
   - Run tests and verify they pass

3. **Quality Gates**: Before committing
   - Run linter and formatter
   - Run `/sonar-check` for code quality validation
   - Run `/trivy-scan` for security scanning
   - Ensure all quality gates pass

4. **Commit**: Use `/commit-writer` to generate conventional commits

5. **Memory**: Save learnings to Engram (see Memory Protocol below)

### For Bug Fixes

1. Write a failing test that reproduces the bug
2. Fix the bug
3. Verify test passes
4. Run quality gates (`/sonar-check`, `/trivy-scan`)
5. Commit with `/commit-writer`
6. Save bugfix to memory with `mem_save`

---

## Issue Tracking Integration

### GitHub Issues

Use the `/user-stories` skill to create issues. It auto-detects GitHub and uses MCP integration:

```bash
# The skill handles this automatically, but MCP must be configured
# If not configured, the skill will guide you to set it up
```

### GitLab Issues

Use the `/user-stories` skill (auto-detects GitLab) or use `glab` directly:

```bash
cat > /tmp/issue.md << 'EOF'
## User Story

```
As <role>
I want <action>
```

## Acceptance Criteria

```gherkin
Feature: ...
EOF

glab issue create --repo owner/repo \
  --title "Issue title" \
  --label "feature,sprint-1" \
  --description "$(cat /tmp/issue.md)"
```

---

## Memory (Engram)

You have access to persistent memory via MCP tools (mem_save, mem_search, mem_session_summary, etc.).

- Save proactively after significant work — do not wait to be asked.
- After any compaction or context reset, call `mem_context` to recover the state from previous sessions before continuing.

### When to save
- Finished a bugfix → mem_save (type: bugfix)
- Made an architecture or technology decision → mem_save (type: decision, topic_key: "architecture/xxx")
- Discovered a gotcha or non-obvious pattern → mem_save (type: discovery)
- Configured something non-trivial → mem_save (type: config)
- Identified a project or user preference → mem_save (type: preference)

### When starting a session
1. Call mem_context to review recent history (fast and cheap)
2. If relevant context is missing, call mem_search with keywords for the current topic

### When ending a session
Call mem_session_summary with structure:
- Goal: what was being attempted
- Accomplished: what was completed
- Discoveries: important findings
- Files: relevant modified files

### In case of compaction
If you see a reset or context compaction message:
1. IMMEDIATELY call mem_session_summary with the content of the compacted summary
2. Then call mem_context to recover additional context
Do not skip step 1. Without it, everything done before the compaction is lost from memory.

### Key Learnings pattern
When finishing significant work, include at the end of your response:
## Key Learnings:
1. [learning]
2. [other learning]
Engram will extract and save these items automatically via mem_capture_passive.

### Progressive search
1. First call mem_context (reviews recent session history)
2. If not found, call mem_search with relevant keywords
3. If a match is found, us

---

## Development Guidelines

### Quality Standards

- **Test Coverage**: Use `/testing` skill to maintain coverage targets and mutation scores
- **Code Quality**: Use `/sonar-check` to validate Quality Gates (complexity, duplication, maintainability)
- **Security**: Use `/trivy-scan` to detect vulnerabilities, secrets, and misconfigurations
- **Commit Messages**: Use `/commit-writer` for conventional commits with proper body and co-authoring

### TDD Cycle (use `/testing` skill)

1. **Red**: Write failing tests (use Gherkin scenarios from `/user-stories` as guide)
2. **Green**: Implement minimum code to pass tests
3. **Refactor**: Improve code while keeping tests green
4. **Verify**: Run quality gates (`/sonar-check`, `/trivy-scan`)
5. **Commit**: Generate commit with `/commit-writer`
6. **Remember**: Save learnings with `mem_save`

### Before Merging

- [ ] All tests passing (unit, integration, E2E)
- [ ] Test coverage meets targets (use `/testing` for guidance)
- [ ] Mutation score acceptable (use `/testing` for interpretation)
- [ ] SonarQube Quality Gate passed (use `/sonar-check`)
- [ ] Security scan clean (use `/trivy-scan`)
- [ ] Linter and formatter run
- [ ] Commit messages follow conventions (use `/commit-writer`)
- [ ] User stories updated/closed with evidence (use `/user-stories`)

---

## Architecture and Design Rules (Project-Specific Example)

**Note**: This section is project-specific. Adapt it to your architecture and tech stack.

### Example: Pragmatic Hexagonal Architecture

**Layering:** Strict separation between:
- **Presentation** (`api/`): HTTP handlers, controllers, DTOs
- **Business Logic** (`domain/`, `services/`): Core domain models, use cases
- **Infrastructure** (`infrastructure/`): Database, external APIs, adapters

**Structural Typing (Go style):** 
- Use `interface` in domain to define contracts
- Avoid complex inheritance or `abstract class`
- Leverage TypeScript's structural Duck Typing

**KISS Principle:** Flat, easy-to-read and test code. Prefer pure functions over classes when possible.

**Configuration Example:**
```ts
// config.ts
import { z } from "zod";

const ConfigSchema = z.object({
  DATABASE_URL: z.string().url(),
  PORT: z.coerce.number().default(3000),
});

export const config = ConfigSchema.parse(process.env); // implicit singleton
```

---

## Adapting This Template

This `CLAUDE.example.md` is a starting point. To use it in your project:

1. **Copy to `CLAUDE.md`** in your project root (or `.claude/CLAUDE.md`)
2. **Customize Architecture section** with your specific rules (layering, patterns, conventions)
3. **Adjust Quality Gates** (coverage targets, mutation scores, SonarQube thresholds)
4. **Configure Issue Tracker** (GitHub vs GitLab, labels, project board links)
5. **Add Project Context** (domain knowledge, key constraints, team conventions)
6. **Remove what you don't use** (if you don't use SonarQube, remove that section)

The skills (`/spec-driven-dev`, `/user-stories`, `/testing`, etc.) are universal and work across projects. The CLAUDE.md file is where you add project-specific context.

---

## Spec-Driven Development Workflow

When starting a new feature or application, use the `/spec-driven-dev` skill:

### Phase 1: SPECIFY
- Refine the idea through clarifying questions
- Create structured spec: `specs/{feature-name}.md`
- Define objective, requirements, architecture, testing strategy, success criteria
- Get approval before advancing

### Phase 2: PLAN
- Create implementation plan: `specs/{feature-name}-plan.md`
- Break into components, identify dependencies, estimate effort
- Document risks and assumptions
- Get approval before advancing

### Phase 3: TASKS
- Break plan into discrete, testable tasks
- Create GitHub/GitLab Issues with `/user-stories`
- Link issues to spec files
- Get approval before advancing

### Phase 4: IMPLEMENT
- Execute tasks one-by-one using TDD (`/testing`)
- Verify against acceptance criteria
- Update task status and link commits
- Close issues with evidence when complete

**Key principle**: Specs are source of truth, code is regenerable. Keep specs updated when requirements change.

---

## Testing Guidelines

Use the `/testing` skill for comprehensive testing guidance:

- **TDD Workflow**: Red → Green → Refactor cycle
- **Coverage Targets**: Risk-based targets (critical: 90-100%, core: 80-90%, utility: 70-80%)
- **Mutation Testing**: Interpretation of mutation scores and how to improve
- **Test Quality**: Assertion strength, async patterns, E2E best practices
- **Debugging**: When tests fail, how to diagnose and fix

See `/testing` skill for detailed guidance and examples.

---

## Security Scanning

Use the `/trivy-scan` skill to detect security issues before they reach production:

**What it scans:**
- **Vulnerabilities**: OS packages, language dependencies (npm, pip, go.mod, etc.)
- **Secrets**: API keys, passwords, tokens accidentally committed
- **IaC Misconfigurations**: Kubernetes, Docker, Terraform security issues
- **Licenses**: License compliance for dependencies

**When to run:**
- Before committing (catches secrets before they enter git history)
- Before merging PRs (validates dependencies are secure)
- On schedule (detect newly discovered CVEs in existing dependencies)

**Workflow:**
```bash
# The skill handles this automatically:
# 1. Detects what scanners are needed (based on project files)
# 2. Runs scans (filesystem, secrets, IaC, licenses)
# 3. Triages by severity (CRITICAL > HIGH > MEDIUM > LOW)
# 4. Provides remediation guidance
```

**Ignore workflow**: Use `.trivyignore` for accepted risks (document why)

See `/trivy-scan` skill for detailed usage and examples.

---

## Code Quality Analysis

Use the `/sonar-check` skill to validate code quality metrics:

**What it analyzes:**
- **Code Smells**: Complexity, duplication, maintainability issues
- **Security Hotspots**: Potential security vulnerabilities
- **Bugs**: Probable bugs detected by static analysis
- **Coverage**: Test coverage gaps
- **Technical Debt**: Estimated time to fix all issues

**Quality Gates** (customize per project):
- Maintainability Rating: A or B
- Reliability Rating: A
- Security Rating: A
- Coverage: >= 80% (adjust based on risk)
- Duplication: < 3%

**Workflow:**
```bash
# The skill uses MCP-first approach:
# 1. Ping SonarQube MCP to check availability
# 2. If available: use fast MCP tools for analysis
# 3. If not: fall back to Docker + sonar-scanner CLI
# 4. Triage issues by severity (BLOCKER > CRITICAL > MAJOR > MINOR)
```

**Scoped analysis**: For fix-loop iterations, analyze only changed code (faster feedback)

See `/sonar-check` skill for setup and detailed usage.

---

## User Stories and Issue Management

Use the `/user-stories` skill to write and manage user stories:

**What it does:**
- Write user stories in domain language (not technical implementation)
- Validate stories with **INVEST** criteria (Independent, Negotiable, Valuable, Estimable, Small, Testable)
- Write acceptance criteria in **Gherkin** format (Given/When/Then)
- Split large stories using **SPIDR** (Spike, Paths, Interfaces, Data, Rules)
- Publish to GitHub/GitLab Issues with proper formatting and labels
- Close issues with evidence (changes, tests, links to PRs)

**Fundamental rule**: Every acceptance criterion MUST have an automated test. No exceptions.

**When to use:**
- Planning new features (write stories before coding)
- Refining backlog (split large stories, add missing criteria)
- Publishing to issue tracker (create issues with proper format)
- Closing completed work (document what was done with evidence)

See `/user-stories` skill for templates, examples, and INVEST/SPIDR guidance.
