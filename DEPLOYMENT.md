# Deployment Guide

This project uses **GitHub tags** for versioning. Each tag creates a versioned release that users can install directly.

---

## Release Checklist

- [ ] `VERSION` file updated to the new version
- [ ] Git tag created and pushed

---

## How to Release a New Version

### 1. Update the `VERSION` file

```bash
echo "0.2.0" > VERSION
git add VERSION
git commit -m "Bump version to 0.2.0"
```

### 2. Create and push the tag

```bash
git tag -a v0.2.0 -m "Release v0.2.0"
git push origin v0.2.0
```

### 3. Verify the release

```bash
pip install "git+https://github.com/myndfire/pydanticai-fluent.git@v0.2.0"
```

---

## Versioning Guidelines

| Bump Type | When to Use | Example |
|---|---|---|
| **Patch** (`0.1.0` -> `0.1.1`) | Bug fixes, performance improvements | Fix import error, reduce latency |
| **Minor** (`0.1.0` -> `0.2.0`) | New features, backward-compatible changes | Add new memory provider, new tool pattern |
| **Major** (`0.1.0` -> `1.0.0`) | Breaking API changes | Rename `ManagedAgent` methods, remove deprecated features |

---

## Current Release

- **v0.1.0** — Initial versioned release
