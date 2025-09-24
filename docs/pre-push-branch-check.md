# 🔒 Prevent Accidental Pushes to `main`/`master` — With Emergency Override

This guide shows you how to install a **Git pre‑push hook** that blocks pushes to branches (`main` or `master`) unless you explicitly set a noisy environment variable:

```
BREAK_GLASS_MAIN_PUSH=1
```

You can install this hook **globally** (affecting all your repos) or **per repo** (only in the specific repo you choose).
Pick the option that best fits your workflow.

---

## 🛠 The Hook Script

Both installation methods use the same script:

```bash
#!/bin/sh
branch="$(git symbolic-ref --short HEAD)"

if [ "$branch" = "main" ] || [ "$branch" = "master" ]; then
    if [ "$BREAK_GLASS_MAIN_PUSH" = "1" ]; then
        echo "⚠️ Override used: pushing to '$branch'..."
    else
        echo "❌ Push to '$branch' is disabled locally."
        echo "   If you REALLY need to, run:"
        echo "      BREAK_GLASS_MAIN_PUSH=1 git push origin $branch"
        exit 1
    fi
fi
```

---

## Option 1 — Install Globally (All Repos)

This will protect every repo on your machine by default.

1. Create a global hooks directory:
    ```bash
    mkdir -p ~/.git-hooks
    ```

2. Create the pre‑push hook:
    ```bash
    vim ~/.git-hooks/pre-push
    ```
    Paste the script above.

3. Make it executable:
    ```bash
    chmod +x ~/.git-hooks/pre-push
    ```

4. Tell Git to use it globally:
    ```bash
    git config --global core.hooksPath ~/.git-hooks
    ```

---

## Option 2 — Install Per Repo (Only One Project)

This will protect only the repo you set it up in.

1. Go to your repo:
    ```bash
    cd /path/to/your-repo
    ```

2. Create the pre‑push hook:
    ```bash
    vim .git/hooks/pre-push
    ```
    Paste the script above.

3. Make it executable:
    ```bash
    chmod +x .git/hooks/pre-push
    ```

---

## ✅ Testing

1. Try pushing to `main` without override:
    ```bash
    git push origin main
    ```
    ➡ Should be **blocked**.

2. Try with override:
    ```bash
    BREAK_GLASS_MAIN_PUSH=1 git push origin main
    ```
    ➡ Allowed with warning.

---
