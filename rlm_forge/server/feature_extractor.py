"""Semi-automatic feature extraction: discovers (source, test) pairs and removes features."""

import ast
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class ExtractedFeature:
    """Represents a feature removed from a repo for training."""

    source_path: str
    test_path: str
    original_content: str
    num_tests: int
    difficulty: str
    task_description: str


# Curated fallback pairs — known-good (repo, source, test) triples
# AMENDMENT 1: python-slugify test file is test.py at root, NOT test/test_slugify.py
CURATED_PAIRS = [
    {
        "repo_url": "https://github.com/un33k/python-slugify",
        "source_file": "slugify/slugify.py",
        "test_file": "test.py",
        "test_command": "pytest test.py -v",
        "difficulty": "easy",
    },
    {
        "repo_url": "https://github.com/python-humanize/humanize",
        "source_file": "src/humanize/number.py",
        "test_file": "tests/test_number.py",
        "test_command": "pytest tests/test_number.py -v",
        "difficulty": "medium",
    },
    {
        "repo_url": "https://github.com/python-humanize/humanize",
        "source_file": "src/humanize/time.py",
        "test_file": "tests/test_time.py",
        "test_command": "pytest tests/test_time.py -v",
        "difficulty": "medium",
    },
]


class FeatureExtractor:
    """Discovers and extracts (source, test) pairs from Python repos."""

    def discover_pairs(self, repo_path: str) -> list[dict]:
        """Auto-discover (source, test) pairs via filename pattern matching."""
        pairs = []
        test_files = self._find_test_files(repo_path)

        for test_file in test_files:
            source_file = self._match_source_file(repo_path, test_file)
            if source_file and self._verify_import(repo_path, test_file, source_file):
                num_tests = self._count_tests(os.path.join(repo_path, test_file))
                source_loc = self._count_lines(os.path.join(repo_path, source_file))

                # Filter by complexity sweet spot
                if 3 <= num_tests <= 50 and 30 <= source_loc <= 500:
                    pairs.append(
                        {
                            "source_path": source_file,
                            "test_path": test_file,
                            "num_tests": num_tests,
                            "source_loc": source_loc,
                        }
                    )

        # Sort by best fit (prefer 5-20 tests, 50-300 LOC)
        pairs.sort(
            key=lambda p: abs(p["num_tests"] - 12) + abs(p["source_loc"] - 150)
        )
        return pairs

    def _find_test_files(self, repo_path: str) -> list[str]:
        """Find all test files in the repo."""
        test_files = []
        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
            for f in files:
                if f.endswith(".py") and (
                    f.startswith("test_") or f.endswith("_test.py")
                ):
                    rel = os.path.relpath(os.path.join(root, f), repo_path)
                    test_files.append(rel)
        return test_files

    def _match_source_file(
        self, repo_path: str, test_file: str
    ) -> Optional[str]:
        """Given test_foo.py, find foo.py in common source locations."""
        test_basename = os.path.basename(test_file)

        if test_basename.startswith("test_"):
            source_name = test_basename[5:]  # Remove "test_" prefix
        elif test_basename.endswith("_test.py"):
            source_name = test_basename[:-8] + ".py"
        else:
            return None

        # Search common source locations
        search_dirs = ["src", "lib", "."]

        # Also try package directories (dirs with __init__.py)
        try:
            for item in os.listdir(repo_path):
                item_path = os.path.join(repo_path, item)
                if os.path.isdir(item_path) and os.path.exists(
                    os.path.join(item_path, "__init__.py")
                ):
                    search_dirs.append(item)
        except Exception:
            pass

        for search_dir in search_dirs:
            if search_dir == ".":
                candidate = source_name
            else:
                candidate = os.path.join(search_dir, source_name)

            if os.path.exists(os.path.join(repo_path, candidate)):
                return candidate

            # Also search subdirectories of src/
            src_dir = os.path.join(repo_path, search_dir)
            if os.path.isdir(src_dir):
                for sub in os.listdir(src_dir):
                    sub_candidate = os.path.join(search_dir, sub, source_name)
                    if os.path.exists(os.path.join(repo_path, sub_candidate)):
                        return sub_candidate

        return None

    def _verify_import(
        self, repo_path: str, test_file: str, source_file: str
    ) -> bool:
        """Check if test_file likely imports from source_file (basic heuristic)."""
        try:
            base_name = os.path.splitext(os.path.basename(source_file))[0]
            test_content = open(os.path.join(repo_path, test_file)).read()
            return base_name in test_content
        except Exception:
            return False

    def _count_tests(self, test_file_path: str) -> int:
        """Count test functions/methods in a test file using AST."""
        try:
            with open(test_file_path) as f:
                tree = ast.parse(f.read())
            count = 0
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name.startswith("test_"):
                        count += 1
            return count
        except Exception:
            return 0

    def _generate_stub(self, original_content: str) -> str:
        """Generate a stub module with correct function/class signatures but broken implementations.

        Parses the original source with AST to extract all top-level function
        and class definitions, then generates a stub that:
        - Has the same imports (so dependencies resolve)
        - Has the same function/class names with correct signatures
        - Returns None/raises NotImplementedError for all functions
        """
        try:
            tree = ast.parse(original_content)
        except SyntaxError:
            return "# Stub: original file could not be parsed\n"

        lines = ["# STUB: This file needs to be reimplemented.\n"]
        lines.append("# All functions return None — tests will fail.\n\n")

        # Preserve imports from the original
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                lines.append(ast.get_source_segment(original_content, node) + "\n")

        lines.append("\n")

        # Generate stub functions/classes
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.FunctionDef):
                # Extract the full signature from source using body start line
                func_lines = original_content.splitlines()
                # Signature spans from the def line to the line before the body
                body_start = node.body[0].lineno  # 1-indexed
                sig_lines = func_lines[node.lineno - 1 : body_start - 1]
                signature = "\n".join(sig_lines)
                if not signature.rstrip().endswith(":"):
                    signature = signature.rstrip() + ":"
                lines.append(f"{signature}\n")
                lines.append("    return None\n\n")

            elif isinstance(node, ast.ClassDef):
                lines.append(f"class {node.name}:\n")
                lines.append("    pass\n\n")

            elif isinstance(node, ast.Assign):
                # Preserve top-level variable assignments
                segment = ast.get_source_segment(original_content, node)
                if segment:
                    lines.append(segment + "\n")

        return "".join(lines)

    def _patch_init_files(self, repo_path: str, removed_source: str) -> None:
        """Remove imports of the deleted module from __init__.py files.

        When a module like `package/number.py` is removed, the package's
        `__init__.py` may do `from package.number import ...` which would
        crash the entire package import. We comment out those lines.
        """
        module_base = os.path.splitext(os.path.basename(removed_source))[0]
        source_dir = os.path.dirname(removed_source)

        # Check __init__.py in the same directory as the removed file
        init_path = os.path.join(repo_path, source_dir, "__init__.py")
        if not os.path.exists(init_path):
            return

        try:
            with open(init_path, "r") as f:
                lines = f.readlines()

            patched = []
            in_multiline_import = False
            for line in lines:
                # Detect imports referencing the removed module
                if in_multiline_import:
                    patched.append(f"# [RLM-FORGE REMOVED] {line}")
                    if ")" in line:
                        in_multiline_import = False
                elif f".{module_base}" in line and ("import" in line or "from" in line):
                    patched.append(f"# [RLM-FORGE REMOVED] {line}")
                    if "(" in line and ")" not in line:
                        in_multiline_import = True
                elif f'"{module_base}"' in line or f"'{module_base}'" in line:
                    # Catch __all__ references
                    patched.append(line)
                else:
                    patched.append(line)

            with open(init_path, "w") as f:
                f.writelines(patched)
        except Exception:
            pass

    def _count_lines(self, file_path: str) -> int:
        try:
            with open(file_path) as f:
                return sum(1 for _ in f)
        except Exception:
            return 0

    def extract_feature(
        self, repo_path: str, source_path: str, test_path: str
    ) -> ExtractedFeature:
        """Remove source file and create the ExtractedFeature."""
        full_source = os.path.join(repo_path, source_path)
        full_test = os.path.join(repo_path, test_path)

        # Save original content
        with open(full_source, "r") as f:
            original_content = f.read()

        # Count tests
        num_tests = self._count_tests(full_test)

        # Replace the source file with a stub that has correct signatures
        # but wrong implementations. This ensures:
        # - Other modules can still import from it (no cascading ImportErrors)
        # - Tests FAIL (not ERROR), giving a better reward signal
        # - The agent's job is to write the correct implementation
        stub = self._generate_stub(original_content)
        with open(full_source, "w") as f:
            f.write(stub)

        # Generate task description
        task_description = (
            f"The file `{source_path}` has been replaced with a broken stub. "
            f"{num_tests} tests in `{test_path}` are now failing. "
            f"Your task is to explore the repository, understand the expected behavior "
            f"from the tests and other code, and rewrite `{source_path}` with a correct "
            f"implementation so that all tests pass.\n\n"
            f"Available tools:\n"
            f"  read_file(path) - Read a file from the repo\n"
            f"  list_dir(path='.') - List directory contents\n"
            f"  search(pattern, path='.') - Grep for a pattern\n"
            f"  write_file(path, content) - Write/create a file\n"
            f"  run_tests(test_path=None) - Run pytest on a test file\n"
            f"  spawn_agent(scope, mission, budget=5) - Explore a directory scope\n"
            f"  FINAL() - Signal that your implementation is complete\n\n"
            f"Call FINAL() when you believe your implementation is complete."
        )

        return ExtractedFeature(
            source_path=source_path,
            test_path=test_path,
            original_content=original_content,
            num_tests=num_tests,
            difficulty="medium",
            task_description=task_description,
        )
