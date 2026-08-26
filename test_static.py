"""
test_static.py -- catch cross-file mistakes without importing torch.

    python tests/test_static.py

Why this exists. Most of this repo cannot be imported on a machine without a
deep-learning framework, and even with one, a wrong flag name or a renamed
function only shows up minutes into a run -- or, worse, in the shell script that
nobody runs until the day of the deadline. These checks are pure AST work, so
they run anywhere in under a second:

  1. every source file parses and compiles
  2. every `module.attribute` reference between files in this repo resolves to
     something that file actually defines
  3. every command-line flag used in scripts/*.sh and README.md exists in the
     argparse parser of the script it is passed to
  4. the documented default sub-frame count agrees across files

Check 3 earned its keep immediately: the pipeline script was calling
`src/infer.py --out`, but infer.py declares `--csv`.
"""

from __future__ import annotations

import ast
import os
import re
import sys
from typing import Dict, Set

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
SRC = os.path.join(ROOT, "src")

# Attributes reached through a module object that are not module-level names,
# e.g. dataclass fields accessed as W.Window(...).anchor are not checked here.
BUILTIN_OK = {"__file__", "__name__", "__doc__"}


def source_files() -> Dict[str, str]:
    out = {}
    for f in sorted(os.listdir(SRC)):
        if f.endswith(".py"):
            out[f[:-3]] = os.path.join(SRC, f)
    return out


def parse(path: str) -> ast.Module:
    with open(path, "r", encoding="utf-8") as fh:
        return ast.parse(fh.read(), filename=path)


def module_level_names(tree: ast.Module) -> Set[str]:
    """Top-level names a module exposes: defs, classes, assignments, imports."""
    names: Set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for al in node.names:
                names.add(al.asname or al.name.split(".")[0])
        elif isinstance(node, ast.If):  # e.g. definitions under a version guard
            for sub in node.body + node.orelse:
                if isinstance(sub, (ast.FunctionDef, ast.ClassDef)):
                    names.add(sub.name)
                elif isinstance(sub, ast.Assign):
                    for t in sub.targets:
                        if isinstance(t, ast.Name):
                            names.add(t.id)
    return names


def local_aliases(tree: ast.Module, local: Set[str]) -> Dict[str, str]:
    """alias -> local module name, for `import pfio` / `import windows as W`."""
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for al in node.names:
                base = al.name.split(".")[0]
                if base in local:
                    out[al.asname or base] = base
    return out


# --------------------------------------------------------------------------- #


def test_everything_parses_and_compiles():
    for name, path in source_files().items():
        src = open(path, "r", encoding="utf-8").read()
        compile(src, path, "exec")  # raises SyntaxError on failure
    for extra in ("tests/test_core.py", "tests/test_data_numpy.py",
                  "tests/test_static.py"):
        p = os.path.join(ROOT, extra)
        if os.path.isfile(p):
            compile(open(p, encoding="utf-8").read(), p, "exec")


def test_cross_module_references_resolve():
    """The check that catches a rename in one file breaking another.

    Covers both spellings of a cross-file reference:
      import pfio            ->  pfio.project_points(...)
      from pfio import X      ->  X used bare
    """
    files = source_files()
    exported = {n: module_level_names(parse(p)) for n, p in files.items()}
    problems = []
    for name, path in files.items():
        tree = parse(path)
        aliases = local_aliases(tree, set(files))

        # form 1: attribute access through a module alias
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Name)):
                continue
            mod = aliases.get(node.value.id)
            if mod is None or node.attr in BUILTIN_OK:
                continue
            if node.attr not in exported[mod]:
                problems.append(
                    f"{name}.py:{node.lineno} uses {node.value.id}.{node.attr}, "
                    f"but {mod}.py does not define {node.attr!r}")

        # form 2: from-imports of names that must exist in the source module
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.module not in files:
                continue
            for al in node.names:
                if al.name != "*" and al.name not in exported[node.module]:
                    problems.append(
                        f"{name}.py:{node.lineno} imports {al.name!r} from "
                        f"{node.module}.py, which does not define it")
    assert not problems, "\n    " + "\n    ".join(problems)


def _flags_of(script: str) -> Set[str]:
    """Every flag string declared by argparse in src/<script>.py."""
    path = os.path.join(SRC, f"{script}.py")
    if not os.path.isfile(path):
        return set()
    flags: Set[str] = set()
    for node in ast.walk(parse(path)):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    if arg.value.startswith("-"):
                        flags.add(arg.value)
    return flags


def _invocations(text: str):
    """Find `python[3] src/<name>.py <rest>` spans, tolerating \\ line breaks."""
    joined = re.sub(r"\\\s*\n\s*", " ", text)
    for m in re.finditer(r"python3?\s+src/(\w+)\.py((?:\s+[^\n|&;]*)?)", joined):
        yield m.group(1), m.group(2)


def test_documented_flags_exist():
    """Every --flag passed to a script in the shell pipeline or the README must
    be declared by that script. This is the check that found src/infer.py being
    called with --out when it declares --csv."""
    texts = {}
    for rel in ("scripts/run_octopus.sh", "README.md"):
        p = os.path.join(ROOT, rel)
        if os.path.isfile(p):
            texts[rel] = open(p, encoding="utf-8").read()
    assert texts, "no scripts/README found to check"

    problems, checked = [], 0
    for rel, text in texts.items():
        for script, rest in _invocations(text):
            flags = _flags_of(script)
            if not flags:
                problems.append(f"{rel}: calls src/{script}.py, which does not exist")
                continue
            for flag in re.findall(r"(?<![\w-])--[a-z][a-z0-9_-]*", rest):
                checked += 1
                if flag not in flags:
                    problems.append(
                        f"{rel}: src/{script}.py is passed {flag}, which it does "
                        f"not declare (it has: {', '.join(sorted(flags))})")
    assert checked > 5, f"only {checked} flags checked -- the regex probably broke"
    assert not problems, "\n    " + "\n    ".join(problems)


def _choices_of(script: str) -> Dict[str, Set[str]]:
    """flag -> allowed values, for add_argument(..., choices=...)."""
    path = os.path.join(SRC, f"{script}.py")
    if not os.path.isfile(path):
        return {}
    tree = parse(path)
    # choices are often a module-level tuple, e.g. choices=CROP_MODES
    consts: Dict[str, Set[str]] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, (ast.Tuple, ast.List)):
            vals = {e.value for e in node.value.elts
                    if isinstance(e, ast.Constant) and isinstance(e.value, str)}
            if vals:
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        consts[t.id] = vals

    out: Dict[str, Set[str]] = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            continue
        flag = next((a.value for a in node.args
                     if isinstance(a, ast.Constant)
                     and isinstance(a.value, str) and a.value.startswith("--")), None)
        if flag is None:
            continue
        for kw in node.keywords:
            if kw.arg != "choices":
                continue
            if isinstance(kw.value, (ast.Tuple, ast.List, ast.Set)):
                out[flag] = {e.value for e in kw.value.elts
                             if isinstance(e, ast.Constant)}
            elif isinstance(kw.value, ast.Name) and kw.value.id in consts:
                out[flag] = consts[kw.value.id]
            elif (isinstance(kw.value, ast.Attribute)
                  and kw.value.attr in ("WRENCH_FRAMES", "CROP_MODES", "BACKBONES")):
                # defined in another local module; resolve it there
                for other in source_files():
                    o = {}
                    for nd in parse(os.path.join(SRC, f"{other}.py")).body:
                        if (isinstance(nd, ast.Assign)
                                and isinstance(nd.value, (ast.Tuple, ast.List))):
                            for t in nd.targets:
                                if isinstance(t, ast.Name):
                                    o[t.id] = {e.value for e in nd.value.elts
                                               if isinstance(e, ast.Constant)}
                    if kw.value.attr in o:
                        out[flag] = o[kw.value.attr]
                        break
    return out


def test_documented_flag_values_are_valid_choices():
    """A flag can exist and still be given a value argparse will reject. This
    check reads `choices=` and validates the literal values the pipeline passes.
    Written after `--crop_mode center` was tried by hand: the flag is real, the
    value is not (it is one of full/fixed/fk_centered)."""
    problems, checked = [], 0
    for rel in ("scripts/run_octopus.sh", "README.md"):
        p = os.path.join(ROOT, rel)
        if not os.path.isfile(p):
            continue
        text = open(p, encoding="utf-8").read()
        for script, rest in _invocations(text):
            allowed = _choices_of(script)
            if not allowed:
                continue
            toks = rest.split()
            for i, tok in enumerate(toks[:-1]):
                if tok in allowed:
                    val = toks[i + 1]
                    if val.startswith("-") or "$" in val:
                        continue  # a shell variable; value not knowable statically
                    checked += 1
                    if val not in allowed[tok]:
                        problems.append(
                            f"{rel}: src/{script}.py {tok} {val!r} is not a valid "
                            f"choice ({', '.join(sorted(allowed[tok]))})")
    assert checked > 0, "no choice-constrained flags were exercised"
    assert not problems, "\n    " + "\n    ".join(problems)


def test_subframe_default_is_consistent():
    """K is quoted in prose in several places; keep the code the single truth."""
    tree = parse(os.path.join(SRC, "train.py"))
    default = None
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "--subframes"):
            for kw in node.keywords:
                if kw.arg == "default" and isinstance(kw.value, ast.Constant):
                    default = kw.value.value
    assert default == 4, f"train.py --subframes default is {default}, docs say 4"
    sh = os.path.join(ROOT, "scripts/run_octopus.sh")
    if os.path.isfile(sh):
        text = open(sh, encoding="utf-8").read()
        assert "K=${K:-4}" in text, "run_octopus.sh no longer defaults K to 4"


def test_no_leftover_debug_statements():
    """A cheap guard against shipping a breakpoint or a stray TODO marker."""
    bad = []
    for name, path in source_files().items():
        for i, line in enumerate(open(path, encoding="utf-8"), 1):
            s = line.strip()
            if s.startswith("breakpoint(") or "pdb.set_trace" in s:
                bad.append(f"{name}.py:{i}: {s}")
            if "FIXME" in s or "XXX" in s:
                bad.append(f"{name}.py:{i}: {s}")
    assert not bad, "\n    " + "\n    ".join(bad)


def main() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = []
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as exc:
            failed.append(fn.__name__)
            print(f"FAIL  {fn.__name__}:{exc}")
        except Exception as exc:  # noqa: BLE001
            failed.append(fn.__name__)
            print(f"ERROR {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(fns) - len(failed)}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
