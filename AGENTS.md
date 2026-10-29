# AGENTS.md

Guidance for AI coding agents working in this repository.

## Commit Messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/) style used by the Rust/cargo ecosystem (`cargo-release` and `git-cliff` can derive versions and changelogs from it).

Format:

```
<type>(<scope>): <subject>

<body>
```

### Types

- `feat` — new feature (SemVer minor)
- `fix` — bug fix (SemVer patch)
- `docs` — documentation only
- `refactor` — code change that neither fixes a bug nor adds a feature
- `perf` — performance improvement
- `test` — adding or correcting tests
- `build` — build system or dependency changes (`Cargo.toml`, Makefile, `rust-toolchain.toml`)
- `ci` — CI/CD configuration changes
- `chore` — maintenance tasks that do not touch source or test files

### Rules

1. The subject line is imperative mood ("add", not "added" or "adds"), lowercase, no trailing period, ~50 characters or fewer.
2. Scope is optional; use the crate or module name (e.g. `reader`, `meta`, `extension`) when the change is contained.
3. The body explains **why**, not what — the diff already shows what changed. Include context such as the motivation, the quirk in the Milvus backup format being handled, or trade-offs considered.
4. Wrap body lines at 72 characters.
5. Breaking changes must note `BREAKING CHANGE:` in the footer or use `!` after the type (`feat!:`).
6. One logical change per commit; do not mix unrelated fixes.
7. Reference issues in the footer (`Refs #12`, `Fixes #34`) when applicable.
