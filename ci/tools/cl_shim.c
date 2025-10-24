#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Simple helper to append into a dynamically-growing buffer */
static void ensure_capacity(char **buf, size_t *cap, size_t need) {
    if (*cap >= need) return;
    while (*cap < need) *cap *= 2;
    *buf = (char*)realloc(*buf, *cap);
}

/* Does this argument need quoting? (space, tab or double-quote) */
static int needs_quotes(const char *s) {
    for (; *s; ++s) {
        if (*s == ' ' || *s == '\t' || *s == '"') return 1;
    }
    return 0;
}

/* Append a single argument to the command line, quoting/escaping as needed.
   This implements a simple quoting strategy sufficient for most cases:
   - wrap in double quotes if contains space, tab or a double-quote
   - escape internal double-quotes with backslash
   Note: For full generality on Windows you may want to implement the exact
   CreateProcess parsing/escaping rules; this is pragmatic and works for normal
   compiler paths and file names. */
static void append_arg(char **buf, size_t *len, size_t *cap, const char *arg) {
    int quote = needs_quotes(arg);
    if (quote) {
        size_t need = *len + 3 + strlen(arg); /* space + quotes + content */
        ensure_capacity(buf, cap, need + 1);
        (*buf)[(*len)++] = ' ';
        (*buf)[(*len)++] = '"';
        for (const char *p = arg; *p; ++p) {
            if (*p == '"') {
                (*buf)[(*len)++] = '\\';
                (*buf)[(*len)++] = '"';
            } else {
                (*buf)[(*len)++] = *p;
            }
        }
        (*buf)[(*len)++] = '"';
        (*buf)[*len] = '\0';
    } else {
        size_t need = *len + 1 + strlen(arg);
        ensure_capacity(buf, cap, need + 1);
        (*buf)[(*len)++] = ' ';
        strcpy(*buf + *len, arg);
        *len += strlen(arg);
        (*buf)[*len] = '\0';
    }
}

int main(int argc, char **argv) {
    /* Determine target compiler path from CL_EXE environment variable.
       If not set, fall back to "cl.exe" (will rely on PATH). */
    const char *cl_env = getenv("CL_EXE");
    const char *cl_path = cl_env && cl_env[0] ? cl_env : "cl.exe";

    /* Build command line: sccache "<cl_path>" arg1 arg2 ... */
    size_t cap = 1024;
    char *cmd = (char*)malloc(cap);
    if (!cmd) {
        fprintf(stderr, "out of memory\n");
        return 1;
    }
    cmd[0] = '\0';
    size_t len = 0;
    /* start with 'sccache' */
    strcpy(cmd, "sccache");
    len = strlen(cmd);

    /* append the compiler path (quoted if necessary) */
    append_arg(&cmd, &len, &cap, cl_path);

    /* append the rest of the args (skip argv[0]) */
    for (int i = 1; i < argc; ++i) {
        append_arg(&cmd, &len, &cap, argv[i]);
    }

    /* Use CreateProcess to run the command. We pass NULL as lpApplicationName
       and the built command as lpCommandLine; this lets Windows parse it.
       We inherit the current environment and current working directory. */
    STARTUPINFOA si;
    PROCESS_INFORMATION pi;
    ZeroMemory(&si, sizeof(si));
    si.cb = sizeof(si);
    ZeroMemory(&pi, sizeof(pi));

    /* CreateProcessA expects a mutable buffer for lpCommandLine */
    if (!CreateProcessA(NULL, cmd, NULL, NULL, TRUE, 0, NULL, NULL, &si, &pi)) {
        DWORD err = GetLastError();
        fprintf(stderr, "CreateProcess failed: %lu\n", err);
        free(cmd);
        return 1;
    }

    /* Wait for completion and return the child exit code */
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD exit_code = 1;
    if (!GetExitCodeProcess(pi.hProcess, &exit_code)) {
        exit_code = 1;
    }

    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    free(cmd);
    return (int)exit_code;
}
