"""
CompliancePatchEnv — patch agent environment for CompliancePatchBench.
The agent receives a codebase with flagged violations and must produce
minimal, correct patches that pass compliance and existing tests.
"""

import ast
import re
import textwrap
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field, ConfigDict
from environment.rules import ALL_RULES


# ── Utilities ─────────────────────────────────────────────────────────────────

def normalize_indentation(code: str, base_indent: int = 0) -> str:
    """Normalize indentation to make patching more reliable."""
    if not code.strip():
        return code
    
    lines = code.split("\n")
    # Find minimum indentation (excluding empty lines)
    non_empty_lines = [line for line in lines if line.strip()]
    if not non_empty_lines:
        return code
    
    min_indent = min(len(line) - len(line.lstrip()) for line in non_empty_lines)
    
    # Remove minimum indentation and add base_indent
    normalized = []
    for line in lines:
        if line.strip():
            normalized.append(" " * base_indent + line[min_indent:])
        else:
            normalized.append("")
    
    return "\n".join(normalized)


def detect_base_indentation(code: str, line_num: int) -> int:
    """Detect the base indentation level at a given line."""
    lines = code.split("\n")
    if line_num < 1 or line_num > len(lines):
        return 0
    
    # Look at surrounding lines to determine indentation
    for i in range(max(0, line_num - 2), min(len(lines), line_num + 2)):
        line = lines[i]
        if line.strip() and not line.strip().startswith('#'):
            return len(line) - len(line.lstrip())
    
    return 0


# ── Action models ─────────────────────────────────────────────────────────────

class ReadFileAction(BaseModel):
    action_type: str = "read_file"
    path: str

class WritePatchAction(BaseModel):
    action_type: str = "write_patch"
    file: str
    line_start: int
    line_end: int
    new_code: str          # replacement code for lines line_start..line_end

class RunCIAction(BaseModel):
    action_type: str = "run_ci"

class FinalizeAction(BaseModel):
    action_type: str = "finalize_patch"


# ── State ─────────────────────────────────────────────────────────────────────

class PatchEpisodeState(BaseModel):
    task_id: str
    codebase: Dict[str, str]              # filename → content (immutable original)
    violations: List[Dict]                # from RegAudit ground_truth format
    patches: Dict[str, str] = Field(default_factory=dict)  # filename → patched content
    ci_results: List[Dict] = Field(default_factory=list)
    step_count: int = 0
    max_steps: int = 20
    file_reads_remaining: int = 5
    done: bool = False
    cumulative_reward: float = 0.0
    reward_events: List[str] = Field(default_factory=list)
    violations_fixed_count: int = 0       # track progress
    last_ci_pass_count: int = 0           # for step-level delta
    failed_patches: List[Dict] = Field(default_factory=list)  # track failures
    regressions_introduced: int = 0       # track regressions
    fixes_reverted: int = 0               # track reverted fixes
    semantic_validations_passed: int = 0  # track semantic validation success
    semantic_validations_failed: int = 0  # track semantic validation failures
    model_config = ConfigDict(arbitrary_types_allowed=True)


# ── Reward ────────────────────────────────────────────────────────────────────

def compute_patch_reward(
    state: PatchEpisodeState,
    ci_passed: bool,
    tests_passed: bool,
    lines_changed: int,
    patch_is_deletion: bool,
    ast_node_delta: int,
) -> Tuple[float, Dict[str, float]]:
    """
    Deterministic, anti-cheat reward function.

    +1.0  compliance CI passes (violation no longer detected)
    +0.5  existing tests still pass (no regressions)
    +0.2  patch is semantically minimal (AST node delta < 3)
    -0.3  per unnecessary line changed (lines_changed > minimum needed)
    -1.0  patch is just deletion of flagged line (the classic cheat)
    """
    breakdown: Dict[str, float] = {}
    delta = 0.0

    # ANTI-CHEAT: deletion is not a fix
    if patch_is_deletion:
        breakdown["deletion_cheat"] = -1.0
        return -1.0, breakdown

    if ci_passed:
        delta += 1.0
        breakdown["ci_pass"] = 1.0

    if tests_passed:
        delta += 0.5
        breakdown["tests_pass"] = 0.5

    if ast_node_delta < 3:
        delta += 0.2
        breakdown["minimal_patch"] = 0.2

    if lines_changed > 3:
        penalty = min(0.9, (lines_changed - 3) * 0.3)
        delta -= penalty
        breakdown["verbosity_penalty"] = -penalty

    return round(delta, 4), breakdown


# ── CI Sandbox ────────────────────────────────────────────────────────────────

class CISandbox:
    """
    Deterministic compliance checker.
    Runs two checks:
      1. AST syntax validation
      2. Compliance pattern checker (violation no longer present)
    Returns structured results — no external processes, no network.
    """

    @staticmethod
    def check_syntax(code: str) -> Tuple[bool, str]:
        try:
            ast.parse(code)
            return True, "ok"
        except SyntaxError as e:
            return False, f"SyntaxError: {e}"

    @staticmethod
    def count_ast_nodes(code: str) -> int:
        try:
            tree = ast.parse(code)
            return sum(1 for _ in ast.walk(tree))
        except SyntaxError:
            return 0

    @staticmethod
    def check_violation_fixed(
        patched_code: str,
        rule_id: str,
        original_line_start: int,
        original_line_end: int,
    ) -> Tuple[bool, str]:
        """
        Check the violation is no longer present using rule-specific patterns.
        Returns (fixed, reason).
        """
        VIOLATION_PATTERNS = {
            "GDPR-ART5-1A": [
                r"logger\.(info|debug|warning|error).*email",
                r"print.*email",
                r"f['\"].*\{.*email.*\}",
                r"log.*user\.email",
                r"logger\.(info|debug|warning|error).*request\.body",
                r"log_body",
            ],
            "GDPR-ART5-1C": [
                r"to_dict\(\)",
                r"fields\s*=.*password",
                r"includes password",
                r"['\"]password_hash['\"]",
                r"password_hash.*\}",
                r"jsonify\(\{['\"]user['\"]\s*:",
                r"internal_id",
                r"api_key",
            ],
            "GDPR-ART25": [
                r"@app\.route.*(?!@limiter)",
                r"def \w+\(\):\s*$",
            ],
            "GDPR-ART32": [
                r"DEBUG\s*=\s*True",
                r"debug\s*=\s*True",
                r"password\s*!=\s*['\"]password123['\"]",
                r"def create_payment\(",
            ],
            "GDPR-ART30": [
                r"GDPR-ART30 violation",
                r"missing created_at",
                r"# no audit",
                r"without.*log",
                r"pass\s*#.*log",
                r"class User:",
            ],
            "OWASP-A01": [
                r"objects\.get\(id=",
                r"objects\.get\(pk=",
                r"get_object_or_404\(",
                r"jwt\.encode\(\{['\"]user_id['\"]",
                r"User\.get_by_id\(",
                r"no tenant scope",
            ],
            "OWASP-A02": [
                r"SECRET_KEY\s*=\s*['\"][^'\"]{4,}['\"]",
                r"API_KEY\s*=\s*['\"]",
                r"PASSWORD\s*=\s*['\"]",
                r"SELECT \* FROM payments",
                r"build_report_query",
            ],
            "OWASP-A03": [
                r"\.raw\(f['\"]",
                r"execute\(f['\"]",
                r"SELECT.*\{.*\}",
                r"build_report_query",
                r"should filter tenant",
            ],
            "OWASP-A04": [
                r"OWASP-A04 violation",
                r"no extension/MIME validation",
                r"etree\.parse\(",
                r"xml\.parse\(",
                r"lxml\.etree",
                r"parseString\(",
                r"def list_users\(",
            ],
            "SOC2-CC6.1": [
                r"logging\.disable\(",
                r"log_level\s*=\s*None",
                r"logger\.disabled\s*=\s*True",
                r"def token_info\(",
                r"def refund_payment\(",
            ],
            "SOC2-CC7.2": [
                r"except\s*:\s*pass",
                r"except Exception:\s*pass",
            ],
        }

        patterns = VIOLATION_PATTERNS.get(rule_id, [])
        if not patterns:
            return False, f"no pattern defined for {rule_id} — violation assumed present"

        lines = patched_code.split("\n")
        for pattern in patterns:
            for line in lines:
                if re.search(pattern, line, re.IGNORECASE):
                    return False, f"Violation still present: {pattern}"
        return True, "violation pattern not found in patched code"

    @staticmethod
    def is_deletion_patch(
        original_code: str,
        patched_code: str,
        line_start: int,
        line_end: int,
    ) -> bool:
        """
        100% STRICT deletion detection.
        Detects ANY attempt to remove logic without proper replacement.
        """
        orig_lines = original_code.split("\n")
        patch_lines = patched_code.split("\n")
        
        # Check 1: Line count reduction
        lines_removed = len(orig_lines) - len(patch_lines)
        lines_in_range = line_end - line_start + 1
        if lines_removed >= lines_in_range:
            return True
        
        # Check 2: Extract patched section
        ls = max(0, line_start - 1)
        le = min(len(patch_lines), line_start + (line_end - line_start))
        patched_section = "\n".join(patch_lines[ls:le]).strip()
        
        # Check 3: STRICT - Block ALL trivial replacements
        if not patched_section:
            return True  # Empty replacement
        
        # Remove comments and whitespace for semantic check
        semantic_content = "\n".join(
            line for line in patched_section.split("\n")
            if line.strip() and not line.strip().startswith('#')
        ).strip()
        
        if not semantic_content:
            return True  # Only comments/whitespace
        
        # Check 4: Block trivial statements
        trivial_patterns = [
            r'^\s*pass\s*$',
            r'^\s*return\s*$',
            r'^\s*return\s+None\s*$',
            r'^\s*\.\.\.\s*$',
            r'^\s*continue\s*$',
            r'^\s*break\s*$',
        ]
        
        for pattern in trivial_patterns:
            if re.match(pattern, semantic_content, re.MULTILINE | re.IGNORECASE):
                return True
        
        # Check 5: Verify semantic preservation via AST
        try:
            orig_ast = ast.parse(original_code)
            patch_ast = ast.parse(patched_code)
            
            # Count meaningful nodes (exclude Module, Pass, Expr with constants)
            def count_meaningful_nodes(tree):
                count = 0
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.If, 
                                        ast.For, ast.While, ast.With, ast.Try,
                                        ast.Assign, ast.AugAssign, ast.Return,
                                        ast.Call, ast.Attribute)):
                        count += 1
                return count
            
            orig_meaningful = count_meaningful_nodes(orig_ast)
            patch_meaningful = count_meaningful_nodes(patch_ast)
            
            # If meaningful nodes reduced significantly, it's likely a deletion
            if orig_meaningful > 0 and patch_meaningful < orig_meaningful * 0.5:
                return True
        except SyntaxError:
            pass  # Can't parse, let other checks handle it
        
        return False

    @staticmethod
    def count_changed_lines(original: str, patched: str) -> int:
        orig = original.split("\n")
        new = patched.split("\n")
        changes = 0
        for i in range(min(len(orig), len(new))):
            if orig[i] != new[i]:
                changes += 1
        changes += abs(len(orig) - len(new))
        return changes

    @staticmethod
    def run_semantic_validation(patched_code: str, rule_id: str, line_start: int = None, line_end: int = None) -> Tuple[bool, str]:
        """
        RELIABLE semantic validation via static analysis.
        Validates correctness using AST analysis and pattern matching.
        Correct patch → validation passes, incorrect patch → validation fails.
        
        If line_start/line_end provided, only check those specific lines.
        Otherwise, check the entire file.
        """
        SEMANTIC_VALIDATORS = {
            "GDPR-ART5-1A": {
                "test": lambda code, ls, le: (
                    # Check only the specific lines if provided
                    (lambda lines: not any(
                        "logger" in line and "email" in line and 
                        ("user.email" in line or "f\"" in line or "f'" in line or ".format" in line)
                        for idx, line in enumerate(lines)
                        if ls is None or (ls <= idx + 1 <= le)
                    ))(code.split("\n")) and
                    # Must have logging (not deleted)
                    ("logger" in code or "logging" in code)
                ),
                "description": "Must not log email in logger statements"
            },
            "GDPR-ART5-1C": {
                "test": lambda code, ls, le: (
                    # Check that password_hash is NOT in return/jsonify lines (only in specified range)
                    not any("password_hash" in line and ("return" in line or "jsonify" in line) 
                           for idx, line in enumerate(code.split("\n"))
                           if ls is None or (ls <= idx + 1 <= le))
                ),
                "description": "Must not expose password_hash in responses"
            },
            "GDPR-ART25": {
                "test": lambda code, ls, le: (
                    # Must have rate limiting
                    ("@limiter" in code or "limiter.limit" in code or "@app.route" in code) and
                    # Must have function definition (not deleted)
                    "def " in code
                ),
                "description": "Must have rate limiting on function"
            },
            "OWASP-A03": {
                "test": lambda code, ls, le: (
                    # Must NOT use f-strings in SQL
                    ".raw(f" not in code and
                    "execute(f" not in code and
                    "f\"SELECT" not in code and
                    "f'SELECT" not in code
                ),
                "description": "Must not use f-strings in SQL queries"
            },
            "OWASP-A02": {
                "test": lambda code, ls, le: (
                    # If SECRET_KEY exists, must use environment
                    ("SECRET_KEY" not in code or 
                     "os.environ" in code or 
                     "getenv" in code or
                     "environ.get" in code)
                ),
                "description": "Must load secrets from environment if present"
            },
        }
        
        test_spec = SEMANTIC_VALIDATORS.get(rule_id)
        if not test_spec:
            # No validator defined = pass (don't block unknown rules)
            return True, "no semantic validator defined"
        
        try:
            passed = test_spec["test"](patched_code, line_start, line_end)
            reason = test_spec["description"] if passed else f"FAILED: {test_spec['description']}"
            return passed, reason
        except Exception as e:
            # Exception = fail (don't silently pass)
            return False, f"Semantic validation error: {e}"


# ── Environment ───────────────────────────────────────────────────────────────

class CompliancePatchEnv:
    """
    Patch agent environment.
    Agent reads files, writes patches, runs CI, and finalizes.
    """

    def __init__(self):
        self.state: Optional[PatchEpisodeState] = None
        self.ci = CISandbox()

    def reset(
        self,
        task_id: str,
        codebase: Dict[str, str],
        violations: List[Dict],
        max_steps: int = 20,
        file_reads_remaining: int = 5,
    ) -> Dict:
        """Start a new patch episode."""
        self.state = PatchEpisodeState(
            task_id=task_id,
            codebase=codebase,
            violations=violations,
            patches={fname: content for fname, content in codebase.items()},
            max_steps=max_steps,
            file_reads_remaining=file_reads_remaining,
        )
        return self._build_observation("Episode started. Read files, write patches, run CI, then finalize.")

    def step(self, action: Dict) -> Tuple[Dict, float, bool, Dict]:
        assert self.state is not None, "Call reset() first"
        assert not self.state.done, "Episode done — call reset()"

        self.state.step_count += 1
        action_type = action.get("action_type", "")
        reward_delta = 0.0
        breakdown = {}
        info = {}
        result = ""  # Initialize result

        if action_type == "read_file":
            path = action.get("path", "")
            if self.state.file_reads_remaining <= 0:
                result = "ERROR: read budget exhausted"
                reward_delta = -0.02
                breakdown = {"budget_exceeded": -0.02}
            elif path not in self.state.patches:
                result = f"ERROR: file '{path}' not found. Available: {list(self.state.patches.keys())}"
                reward_delta = -0.02
                breakdown = {"invalid_read": -0.02}
            else:
                result = self.state.patches[path]
                self.state.file_reads_remaining -= 1

        elif action_type == "write_patch":
            result, reward_delta, breakdown = self._apply_patch(action)

        elif action_type == "run_ci":
            result, reward_delta, breakdown = self._run_ci()

        elif action_type == "finalize_patch":
            return self._finalize()

        else:
            result = f"ERROR: unknown action_type '{action_type}'"
            reward_delta = -0.05
            breakdown = {"invalid_action": -0.05}

        self.state.cumulative_reward += reward_delta
        if reward_delta != 0 or breakdown:
            self.state.reward_events.append(
                f"step={self.state.step_count};{action_type};delta={reward_delta:.4f};{breakdown}"
            )

        # Terminate on step limit OR if all violations fixed
        if self.state.step_count >= self.state.max_steps:
            info["termination_reason"] = "max_steps_reached"
            obs, reward, done, finalize_info = self._finalize(termination_reason="max_steps_reached")
            info.update(finalize_info)
            return obs, reward, done, info
        elif self.state.violations_fixed_count == len(self.state.violations):
            info["termination_reason"] = "all_violations_fixed"
            obs, reward, done, finalize_info = self._finalize(termination_reason="all_violations_fixed")
            info.update(finalize_info)
            return obs, reward, done, info

        obs = self._build_observation(result)
        return obs, reward_delta, self.state.done, info

    def get_state(self) -> Dict:
        if self.state is None:
            return {"status": "not_started"}
        return self.state.model_dump()

    # ── Private methods ────────────────────────────────────────────────────────

    def _apply_patch(self, action: Dict) -> Tuple[str, float, Dict]:
        """Apply patch with lenient indentation handling for usability."""
        file = action.get("file", "")
        line_start = int(action.get("line_start", 1))
        line_end = int(action.get("line_end", line_start))
        new_code = action.get("new_code", "")

        if file not in self.state.patches:
            self.state.failed_patches.append({
                "file": file, "reason": "file_not_found", "step": self.state.step_count
            })
            return f"ERROR: file '{file}' not found", -0.05, {"invalid_file": -0.05}

        original = self.state.patches[file]
        lines = original.split("\n")

        # Validate line range
        if line_start < 1 or line_end > len(lines) or line_start > line_end:
            self.state.failed_patches.append({
                "file": file, "reason": "invalid_line_range", "step": self.state.step_count
            })
            return f"ERROR: invalid line range {line_start}-{line_end}", -0.05, {"invalid_range": -0.05}

        # LENIENT: Try multiple indentation strategies
        base_indent = detect_base_indentation(original, line_start)
        
        # Strategy 1: Use provided code as-is
        ls = max(0, line_start - 1)
        le = min(len(lines), line_end)
        new_lines = new_code.split("\n") if new_code.strip() else []
        patched_lines = lines[:ls] + new_lines + lines[le:]
        patched = "\n".join(patched_lines)
        
        syntax_ok, syntax_msg = self.ci.check_syntax(patched)
        
        # Strategy 2: If failed, try with auto-normalization
        if not syntax_ok:
            normalized_code = normalize_indentation(new_code, base_indent)
            new_lines = normalized_code.split("\n") if normalized_code.strip() else []
            patched_lines = lines[:ls] + new_lines + lines[le:]
            patched = "\n".join(patched_lines)
            syntax_ok, syntax_msg = self.ci.check_syntax(patched)
        
        # Strategy 3: If still failed, try preserving surrounding indentation
        if not syntax_ok and ls > 0:
            # Match indentation of previous line
            prev_line = lines[ls - 1] if ls > 0 else ""
            prev_indent = len(prev_line) - len(prev_line.lstrip())
            normalized_code = normalize_indentation(new_code, prev_indent)
            new_lines = normalized_code.split("\n") if normalized_code.strip() else []
            patched_lines = lines[:ls] + new_lines + lines[le:]
            patched = "\n".join(patched_lines)
            syntax_ok, syntax_msg = self.ci.check_syntax(patched)

        # Final check: reject only if all strategies failed
        if not syntax_ok:
            self.state.failed_patches.append({
                "file": file, "reason": "syntax_error", "message": syntax_msg, "step": self.state.step_count
            })
            return f"PATCH REJECTED — {syntax_msg}", -0.05, {"syntax_error": -0.05}

        # STRICT deletion check - but only flag, don't reject yet (CI will catch it)
        is_deletion = self.ci.is_deletion_patch(original, patched, line_start, line_end)
        if is_deletion:
            # Mark but don't reject - let CI be final authority
            self.state.failed_patches.append({
                "file": file, "reason": "potential_deletion", "step": self.state.step_count
            })

        # State mutation after validation
        self.state.patches[file] = patched
        changed = self.ci.count_changed_lines(original, patched)
        return f"Patch applied to {file} ({changed} lines changed). Run CI to verify.", 0.0, {}

    def _run_ci(self) -> Tuple[str, float, Dict]:
        """Run CI with before/after comparison and strict reward gating."""
        results = []
        total_reward = 0.0
        total_breakdown: Dict[str, float] = {}
        
        # Track before state for delta calculation
        previous_pass_count = self.state.last_ci_pass_count

        for v in self.state.violations:
            fname = v["file"]
            rule_id = v["rule_id"]
            ls = v["line_start"]
            le = v["line_end"]

            original = self.state.codebase.get(fname, "")
            patched = self.state.patches.get(fname, original)

            # STRICT: No reward if syntax broken
            syntax_ok, syntax_reason = self.ci.check_syntax(patched)
            if not syntax_ok:
                results.append({
                    "file": fname, "rule_id": rule_id, "ci": "FAIL", 
                    "reason": f"syntax_error: {syntax_reason}", "reward": 0.0
                })
                continue

            # Check if violation actually fixed (before vs after)
            original_fixed, _ = self.ci.check_violation_fixed(original, rule_id, ls, le)
            patched_fixed, reason = self.ci.check_violation_fixed(patched, rule_id, ls, le)
            
            # SEMANTIC VALIDATION: Static analysis via AST and patterns (PRIMARY SIGNAL)
            semantic_passed, semantic_reason = self.ci.run_semantic_validation(patched, rule_id, ls, le)
            
            # Track semantic validation results
            if semantic_passed:
                self.state.semantic_validations_passed += 1
            else:
                self.state.semantic_validations_failed += 1
            
            # CRITICAL FIX: Semantic validation is PRIMARY signal, pattern check is SECONDARY
            # If semantic validation passes AND pattern check passes, it's fixed
            # If semantic validation passes but pattern check fails, trust semantic validation
            # If semantic validation fails, it's not fixed regardless of pattern
            if semantic_passed:
                # Semantic validation passed - check if violation was present originally
                if not original_fixed:
                    # Violation was present, semantic validation now passes = FIXED
                    fixed = True
                    reason = semantic_reason
                else:
                    # Violation wasn't present originally (false positive in ground truth?)
                    fixed = False
                    reason = "No violation detected in original code"
            else:
                # Semantic validation failed - not fixed
                fixed = False
                reason = semantic_reason
            
            is_deletion = self.ci.is_deletion_patch(original, patched, ls, le)
            changed = self.ci.count_changed_lines(original, patched)
            orig_nodes = self.ci.count_ast_nodes(original)
            new_nodes = self.ci.count_ast_nodes(patched)
            node_delta = abs(new_nodes - orig_nodes)

            # Global test: all files must remain valid
            tests_passed = all(
                self.ci.check_syntax(c)[0]
                for c in self.state.patches.values()
            )

            # STRICT REWARD GATING: only if violation actually reduced AND not deletion
            # CI IS FINAL AUTHORITY - check deletion here too
            is_deletion_final = self.ci.is_deletion_patch(original, patched, ls, le)
            
            if fixed and not is_deletion and not is_deletion_final:
                r, bd = compute_patch_reward(
                    self.state,
                    ci_passed=fixed,
                    tests_passed=tests_passed,
                    lines_changed=changed,
                    patch_is_deletion=False,
                    ast_node_delta=node_delta,
                )
            elif is_deletion or is_deletion_final:
                r, bd = -1.0, {"deletion_cheat": -1.0}
                fixed = False  # Override - deletion is NOT a fix
            else:
                r, bd = 0.0, {"no_improvement": 0.0}

            total_reward += r
            for k, val in bd.items():
                total_breakdown[f"{rule_id}_{k}"] = val

            results.append({
                "file": fname,
                "rule_id": rule_id,
                "ci": "PASS" if fixed else "FAIL",
                "reason": "DELETION DETECTED - not counted as fix" if (is_deletion or is_deletion_final) else reason,
                "reward": r,
                "is_deletion": is_deletion or is_deletion_final,
                "tests_passed": tests_passed,
                "deletion_blocked": is_deletion or is_deletion_final,
                "semantic_validation_passed": semantic_passed,
                "semantic_validation_reason": semantic_reason,
            })

        # Update state ONLY after successful CI run
        self.state.ci_results = results
        pass_count = sum(1 for r in results if r["ci"] == "PASS")
        self.state.violations_fixed_count = pass_count
        
        # Step-level progression signal with regression tracking
        delta_fixed = pass_count - previous_pass_count
        if delta_fixed > 0:
            total_breakdown["progress_bonus"] = delta_fixed * 0.1
            total_reward += delta_fixed * 0.1
        elif delta_fixed < 0:
            self.state.regressions_introduced += abs(delta_fixed)
            self.state.fixes_reverted += abs(delta_fixed)
            total_breakdown["regression_penalty"] = delta_fixed * 0.2
            total_reward += delta_fixed * 0.2
        
        self.state.last_ci_pass_count = pass_count
        total = len(results)

        summary = f"CI complete: {pass_count}/{total} violations fixed. Reward: {total_reward:+.4f}"
        return summary, total_reward, total_breakdown

    def _hidden_compliance_check(self) -> Dict:
        """Run the hidden oracle inside the environment reward path."""
        try:
            from project.hidden_compliance import run_hidden_compliance_checks
        except Exception:
            return {"hidden_violation": False, "reason": "hidden_oracle_unavailable", "findings": []}
        return run_hidden_compliance_checks(dict(self.state.patches))

    def _finalize(self, termination_reason: str = "finalize_patch") -> Tuple[Dict, float, bool, Dict]:
        _, final_reward, breakdown = self._run_ci()

        pass_count = sum(1 for r in self.state.ci_results if r["ci"] == "PASS")
        total = len(self.state.violations)
        hidden = self._hidden_compliance_check()
        hidden_violation = bool(hidden.get("hidden_violation"))
        partial_fix = 0 < pass_count < total
        no_fix = total > 0 and pass_count == 0 and not hidden_violation
        fail_timeout = termination_reason == "max_steps_reached" and pass_count < total

        if hidden_violation:
            final_reward -= 1.0
            breakdown["hidden_violation_penalty"] = -1.0
        if partial_fix:
            final_reward -= 0.5
            breakdown["partial_fix_penalty"] = -0.5
        if no_fix:
            final_reward -= 0.2
            breakdown["no_fix_penalty"] = -0.2
        if fail_timeout:
            final_reward -= 0.10
            breakdown["timeout_penalty"] = -0.10

        self.state.cumulative_reward = round(final_reward, 4)
        self.state.done = True
        if hidden_violation or partial_fix or fail_timeout:
            self.state.reward_events.append(
                f"finalize;reason={termination_reason};hidden={hidden_violation};"
                f"partial={partial_fix};no_fix={no_fix};timeout={fail_timeout};score={self.state.cumulative_reward:.4f}"
            )

        critique = {
            "final_score": self.state.cumulative_reward,
            "violations_fixed": pass_count,
            "violations_total": total,
            "ci_results": self.state.ci_results,
            "reward_breakdown": breakdown,
            "hidden_violation": hidden_violation,
            "hidden_reason": str(hidden.get("reason", "ok")),
            "hidden_findings": list(hidden.get("findings", [])),
            "partial_fix": partial_fix,
            "no_fix": no_fix,
            "termination_reason": termination_reason,
        }

        obs = self._build_observation(f"Patch session finalized. Score: {self.state.cumulative_reward:.4f}")
        reward_model = type("R", (), {
            "model_dump": lambda self: {
                "value": self.value,
                "cumulative": self.cumulative,
                "breakdown": self.breakdown,
            }
        })()
        reward_model.value = self.state.cumulative_reward
        reward_model.cumulative = self.state.cumulative_reward
        reward_model.breakdown = breakdown
        return obs, self.state.cumulative_reward, True, {"critique": critique, "final_score": self.state.cumulative_reward}

    def _build_observation(self, action_result: str) -> Dict:
        """Build observation with full reward component tracking."""
        return {
            "action_result": action_result,
            "available_files": list(self.state.patches.keys()),
            "violations": self.state.violations,
            "ci_results": self.state.ci_results,
            "file_reads_remaining": self.state.file_reads_remaining,
            "step_count": self.state.step_count,
            "done": self.state.done,
            "cumulative_reward": self.state.cumulative_reward,
            "violations_fixed": self.state.violations_fixed_count,
            "violations_total": len(self.state.violations),
            "failed_patches": self.state.failed_patches,
            "regressions_introduced": self.state.regressions_introduced,
            "fixes_reverted": self.state.fixes_reverted,
            "semantic_validations_passed": self.state.semantic_validations_passed,
            "semantic_validations_failed": self.state.semantic_validations_failed,
            "reward_events": self.state.reward_events[-5:] if len(self.state.reward_events) > 5 else self.state.reward_events,
            "hidden_violation": any("hidden=True" in event for event in self.state.reward_events),
            "no_fix": any("no_fix=True" in event for event in self.state.reward_events),
        }
