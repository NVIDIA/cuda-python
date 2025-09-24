# 🔒 Prevent Accidental Pushes to `main`/`master` — With Emergency Override

This guide sets up a **global Git pre‑push hook** that blocks pushes to protected branches (`main` and `master`) unless you explicitly set an environment variable override:

```
BREAK_GLASS_MAIN_PUSH=1
```

---

## 1. Create a Global Git Hooks Directory
```bash
mkdir -p ~/.git-hooks
```

---

## 2. Create the Pre‑Push Hook

Before adding this new hook, check if `~/.git-hooks/pre-push` already exists.
If it does, **do not overwrite it** — instead, merge the branch‑protection logic into your current hook so both sets of checks run.

Open your existing hook:
```bash
my-favorite-text-editor ~/.git-hooks/pre-push
```

Paste in:
```bash
#!/bin/sh
branch="$(git symbolic-ref --short HEAD)"

if [ "$branch" = "main" ] || [ "$branch" = "master" ]; then
    if [ "$BREAK_GLASS_MAIN_PUSH" = "1" ]; then
        echo "⚠️ Override used: pushing to '$branch'..."
        # allow push
    else
        echo "❌ Push to '$branch' is disabled locally."
        echo "   If you REALLY need to, run:"
        echo "      BREAK_GLASS_MAIN_PUSH=1 git push origin $branch"
        exit 1
    fi
fi
```

---

## 3. Make the Hook Executable
```bash
chmod +x ~/.git-hooks/pre-push
```

---

## 4. Tell Git to Use It Globally
```bash
git config --global core.hooksPath ~/.git-hooks
```

---

## 5. Test It
1. Switch to `main`:
    ```bash
    git checkout main
    ```
2. Try to push:
    ```bash
    git push origin main
    ```
   ➡ Should be **blocked**.

3. Use the override (emergency only):
    ```bash
    BREAK_GLASS_MAIN_PUSH=1 git push origin main
    ```
   ➡ Allowed with a warning.

---
