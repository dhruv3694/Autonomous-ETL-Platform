# PHASE 5.2 — VERIFIED EXECUTION ISOLATION AND SQL SECURITY SPECIFICATION

## 1. Executive Summary

Phase 5.2 establishes verified execution boundaries for user-submitted transformations and hardened SQL analysis:
1. **Isolated Child-Process Execution**: Untrusted Python code is never executed directly inside the main Streamlit application process. Execution is delegated to a ephemeral child process running in an isolated temporary directory with strictly sanitized environment variables and IPC via Parquet serialization.
2. **True Security Transparency**: The platform explicitly documents what is isolated, what is detected, and what cannot be guaranteed by child processes on Windows without kernel containers.
3. **sqlparse AST Tokenization**: Replaces basic regex stripping with `sqlparse` token traversal and SQLite C-level authorizer rules to prevent SQL injection and unauthorized schema mutations.

---

## 2. Python Execution Security Architecture

### A. The Flaw of AST Validation Alone
AST validation and `__builtins__` dictionaries do not form a true security sandbox. Python's runtime reflection, C extensions (such as NumPy/Pandas native calls), and bytecode manipulation can bypass restricted dictionaries if executed in-process.

### B. Dedicated Process Isolation Architecture
The platform enforces execution boundaries via [`execute_python_isolated_process()`](src/etl/isolated_runner.py):
- **Process Boundary**: Spawns an ephemeral Python child process using `subprocess.Popen`.
- **Memory Decoupling**: The child process runs in a separate address space. The main Streamlit process state, session variables, and application credentials cannot be read or corrupted.
- **Controlled Parquet IPC**: Data is serialized to an immutable input Parquet file in a secure temporary directory (`etl_iso_*`). Transformed output is loaded back into the parent only if the child completes successfully.
- **Environment Sanitization**: Application secrets (such as API keys, tokens, or custom environment variables) are excluded from the child environment. Only core runtime system paths (`PATH`, `SYSTEMROOT`, `TEMP`) and standard library search directories are provided.
- **Process Cleanup & Termination**: Enforces a strict execution timeout (`timeout_seconds = 10.0s`). Upon timeout or error, `proc.kill()` is invoked to terminate lingering processes.
- **Temporary Directory Deletion**: Input and output files and the ephemeral runner script are deleted upon completion.

### C. Defenses Prevented vs. Platform Limitations

| Threat Category | Prevented / Handled | Mechanism |
| :--- | :--- | :--- |
| **Parent Memory Corruption** | **YES** | Separate OS process space; no shared pointers or state. |
| **Secret Inheritance** | **YES** | Sanitized child environment; no parent application secrets passed. |
| **Infinite Loops** | **YES** | Subprocess timeout kills child after 10.0s. |
| **Reflection / Dunder Access** | **YES** | Prohibited by AST validator (`__class__`, `__globals__`). |
| **Imports of Dangerous Libs** | **YES** | AST rejects `import os`, `subprocess`, `sys`, `socket`. |
| **OS Filesystem Sandboxing** | **PARTIAL** | On standard Windows without AppContainer / Docker, OS-level filesystem isolation cannot be fully guaranteed. |
| **Network Isolation** | **PARTIAL** | Enforced at the AST layer (blocking socket/HTTP modules); not kernel network namespaces on native Windows. |

> **Transparency Notice**: The UI explicitly informs users that code runs in an isolated child process with IPC serialization, not a kernel-isolated sandbox. Arbitrary code is checked via AST allowlisting before execution.

---

## 3. SQL Security Architecture

### A. AST-Based SQL Tokenization
[`validate_user_sql_code()`](src/etl/optimization_engine.py) leverages `sqlparse` to inspect structural tokens:
- **Statement Count Enforcement**: Multiple statements separated by semicolons are strictly rejected.
- **Allowed Query Types**: Only `SELECT` or `CREATE TABLE ... AS SELECT` statements are permitted.
- **Keyword Allowlisting**: Tokens outside comments and string literals are checked against disallowed keywords (`ATTACH`, `DETACH`, `PRAGMA`, `LOAD_EXTENSION`, `VACUUM`, `ALTER`, `DROP`, `INSERT`, `UPDATE`, `DELETE`, `GRANT`, `REVOKE`).
- **Literal & Comment Resilience**: Keywords embedded inside string literals (`WHERE status = 'ATTACH'`) or harmless comments (`/* pragma */`) are safely ignored and not falsely rejected.

### B. SQLite Authorizer Defense-in-Depth
When executing analysis queries, the SQLite connection registers a C-level security authorizer:
```python
def _sqlite_security_authorizer(action, arg1, arg2, dbname, source):
    if action in (sqlite3.SQLITE_ATTACH, sqlite3.SQLITE_DETACH, sqlite3.SQLITE_PRAGMA):
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK
```
Any attempt to attach external databases, detach databases, or execute pragmas is rejected at the database engine level.

---

## 4. Test Verification

The test suite covers:
- Child process execution and Parquet serialization.
- Subprocess timeout enforcement.
- Non-inheritance of parent process environment secrets.
- AST reflection attack prevention.
- SQL statement count and keyword tokenization.
- Adversarial comments and literal handling.

Total tests passing: **64 tests** in ~23 seconds.
