# 🔒 Prevent Accidental Pushes to `main`/`master` — With Emergency Override

This guide sets up a **global Git pre‑push hook** that blocks pushes to protected branches (`main` and `master`) unless you explicitly use the override flag:

```
--break_glass_to_push_main
```

---

## 1. Create a Global Git Hooks Directory
```bash
mkdir -p ~/.git-hooks
```

---

## 2. Create the Pre‑Push Hook

Before adding this new hook, check if ~/.git-hooks/pre-push already exists.
If it does, do not overwrite it — instead, add the branch‑protection logic to your current hook so both sets of checks run.

Open your existing hook:
```bash
vim ~/.git-hooks/pre-push
```

Paste in:
```bash
#!/bin/sh
branch="$(git symbolic-ref --short HEAD)"
override_flag="--break_glass_to_push_main"

if [ "$branch" = "main" ] || [ "$branch" = "master" ]; then
    case "$*" in
        *"$override_flag"* )
            echo "⚠️ Override used: pushing to '$branch'..."
            ;;
        *)
            echo "❌ Push to '$branch' is disabled locally."
            echo "   If you **really** need to, use: git push $override_flag"
            exit 1
            ;;
    esac
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
    git push
    ```
   ➡  Should be **blocked**.

3. Use the override (emergency only):
    ```bash
    git push --break_glass_to_push_main
    ```
   ➡  Allowed with a warning.

---
