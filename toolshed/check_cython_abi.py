# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


"""
Tool to check for Cython ABI changes in a given package.

Cython must be installed in your venv to run this script.

There are different types of ABI changes, some of which are covered by this tool:

- cdef function signatures (capsule strings) — covered here
- cdef class struct size (tp_basicsize) — covered here
- cdef struct / ctypedef struct field layout — covered here (via .pxd parsing)
- cdef class vtable layout / method reordering — not covered, and this one fails
  as silent UB rather than an import-time error
- Fused specialization ordering — partially covered (reorders manifest as
  capsule-name deltas, but the mapping is non-obvious)

The workflow is basically:

1) Build and install a "clean" upstream version of the package.

2) Generate ABI files from the package by running (in the same venv in which the
   package is installed), where `package_name` is the import path to the package,
   e.g. `cuda.bindings`:

    python check_cython_abi.py generate <package_name> <dir>

3) Checkout a version with the changes to be tested, and build and install.

4) Check the ABI against the previously generated files by running:

    python check_cython_abi.py check <package_name> <dir>
"""

import ctypes
import importlib
import json
import sys
import sysconfig
from io import StringIO
from pathlib import Path

from Cython.Compiler import Parsing
from Cython.Compiler.Scanning import FileSourceDescriptor, PyrexScanner
from Cython.Compiler.Symtab import ModuleScope
from Cython.Compiler.TreeFragment import StringParseContext

EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX")
ABI_SUFFIX = ".abi.json"


_pycapsule_get_name = ctypes.pythonapi.PyCapsule_GetName
_pycapsule_get_name.restype = ctypes.c_char_p
_pycapsule_get_name.argtypes = [ctypes.py_object]


def get_capsule_name(v: object) -> str:
    return _pycapsule_get_name(v).decode("utf-8")


def short_stem(name: str) -> str:
    return name.split(".", 1)[0]


def get_package_path(package_name: str) -> Path:
    package = importlib.import_module(package_name)
    return Path(package.__file__).parent


def import_from_path(root_package: str, root_dir: Path, path: Path) -> object:
    path = path.relative_to(root_dir)
    parts = [root_package] + list(path.parts[:-1]) + [short_stem(path.name)]
    return importlib.import_module(".".join(parts))


def so_path_to_abi_path(so_path: Path, build_dir: Path, abi_dir: Path) -> Path:
    abi_name = f"{short_stem(so_path.name)}{ABI_SUFFIX}"
    return abi_dir / so_path.parent.relative_to(build_dir) / abi_name


def abi_path_to_so_path(abi_path: Path, build_dir: Path, abi_dir: Path) -> Path:
    so_name = f"{short_stem(abi_path.name)}{EXT_SUFFIX}"
    return build_dir / abi_path.parent.relative_to(abi_dir) / so_name


def is_cython_module(module: object) -> bool:
    # This is kind of quick-and-dirty, but seems to work
    return hasattr(module, "__pyx_capi__")


######################################################################################
# STRUCTS


def get_cdef_classes(module: object) -> dict:
    """Extract cdef class (extension type) basicsize from a compiled Cython module."""
    result = {}
    module_name = module.__name__
    for name in sorted(dir(module)):
        obj = getattr(module, name, None)
        if isinstance(obj, type) and getattr(obj, "__module__", None) == module_name and hasattr(obj, "__basicsize__"):
            result[name] = {"basicsize": obj.__basicsize__}
    return result


def _format_base_type_name(bt: object) -> str:
    """Format a Cython base type AST node into a type name string."""
    cls = type(bt).__name__
    if cls == "CSimpleBaseTypeNode":
        return bt.name
    if cls == "CComplexBaseTypeNode":
        inner = _format_base_type_name(bt.base_type)
        return _unwrap_declarator(inner, bt.declarator)[0]
    return cls


def _unwrap_declarator(type_str: str, decl: object) -> tuple[str, str]:
    """Unwrap nested Cython declarator nodes to get (type_string, field_name)."""
    cls = type(decl).__name__
    if cls == "CNameDeclaratorNode":
        return type_str, decl.name
    if cls == "CPtrDeclaratorNode":
        return _unwrap_declarator(f"{type_str}*", decl.base)
    if cls == "CReferenceDeclaratorNode":
        return _unwrap_declarator(f"{type_str}&", decl.base)
    if cls == "CArrayDeclaratorNode":
        dim = getattr(decl, "dimension", None)
        size = getattr(dim, "value", "") if dim is not None else ""
        return _unwrap_declarator(f"{type_str}[{size}]", decl.base)
    return type_str, ""


def _extract_fields_from_cvardef(node: object) -> list:
    """Extract [type, name] pairs from a CVarDefNode."""
    results = []
    for d in node.declarators:
        type_str, name = _unwrap_declarator(_format_base_type_name(node.base_type), d)
        if name:
            results.append([type_str, name])
    return results


def _collect_cvardef_fields(node: object) -> list:
    """Recursively collect CVarDefNode fields, skipping nested struct/class/func defs."""
    fields = []
    if type(node).__name__ == "CVarDefNode":
        fields.extend(_extract_fields_from_cvardef(node))
    skip = ("CStructOrUnionDefNode", "CClassDefNode", "CFuncDefNode")
    for attr_name in getattr(node, "child_attrs", []):
        child = getattr(node, attr_name, None)
        if child is None:
            continue
        if isinstance(child, list):
            for item in child:
                if hasattr(item, "child_attrs") and type(item).__name__ not in skip:
                    fields.extend(_collect_cvardef_fields(item))
        elif hasattr(child, "child_attrs") and type(child).__name__ not in skip:
            fields.extend(_collect_cvardef_fields(child))
    return fields


def _collect_structs_from_tree(node: object) -> dict:
    """Walk a Cython AST and collect struct/class field definitions."""
    result = {}
    cls = type(node).__name__

    if cls == "CStructOrUnionDefNode":
        fields = []
        for attr in node.attributes:
            if type(attr).__name__ == "CVarDefNode":
                fields.extend(_extract_fields_from_cvardef(attr))
        if fields:
            result[node.name] = {"fields": fields}

    elif cls == "CClassDefNode":
        fields = _collect_cvardef_fields(node.body)
        if fields:
            result[node.class_name] = {"fields": fields}

    for attr_name in getattr(node, "child_attrs", []):
        child = getattr(node, attr_name, None)
        if child is None:
            continue
        if isinstance(child, list):
            for item in child:
                if hasattr(item, "child_attrs"):
                    result.update(_collect_structs_from_tree(item))
        elif hasattr(child, "child_attrs"):
            result.update(_collect_structs_from_tree(child))

    return result


class _PxdParseContext(StringParseContext):
    """Parse context that resolves includes via real paths and ignores unknown cimports."""

    def find_module(
        self,
        module_name,
        from_module=None,  # noqa: ARG002
        pos=None,  # noqa: ARG002
        need_pxd=1,  # noqa: ARG002
        absolute_fallback=True,  # noqa: ARG002
        relative_import=False,  # noqa: ARG002
    ):
        return ModuleScope(module_name, parent_module=None, context=self)


def parse_pxd_structs(pxd_path: Path) -> dict:
    """Parse struct and cdef class field definitions from a .pxd file.

    Uses Cython's own parser (in .pxd mode) for reliable extraction.
    cimport lines in the top-level file are stripped since they are
    unresolvable without the full compilation context; included files
    are handled via a lenient context that returns dummy scopes.

    Returns a dict mapping struct/class name to {"fields": [[type, name], ...]}.
    """
    text = pxd_path.read_text(encoding="utf-8")

    # Strip cimport lines (unresolvable without full compilation context)
    lines = text.splitlines()
    cleaned = "\n".join("" if (" cimport " in ln or ln.lstrip().startswith("cimport ")) else ln for ln in lines)

    name = pxd_path.stem
    context = _PxdParseContext(name, include_directories=[str(pxd_path.parent)])
    code_source = FileSourceDescriptor(str(pxd_path))
    scope = context.find_module(name, pos=(code_source, 1, 0), need_pxd=False)

    scanner = PyrexScanner(
        StringIO(cleaned),
        code_source,
        source_encoding="UTF-8",
        scope=scope,
        context=context,
        initial_pos=(code_source, 1, 0),
    )
    tree = Parsing.p_module(scanner, pxd=1, full_module_name=name)
    tree.scope = scope

    return _collect_structs_from_tree(tree)


def get_structs(module: object) -> dict:
    # Extract cdef class basicsize from compiled module (primary)
    structs = get_cdef_classes(module)
    so_path = Path(module.__file__)

    # Parse neighboring .pxd file for struct/class field layout (fallback complement)
    if so_path is not None:
        pxd_path = so_path.parent / f"{short_stem(so_path.name)}.pxd"
        if pxd_path.is_file():
            pxd_structs = parse_pxd_structs(pxd_path)
            for name, info in pxd_structs.items():
                if name in structs:
                    structs[name].update(info)
                else:
                    structs[name] = info

    return dict(sorted(structs.items()))


def _report_field_changes(name: str, expected_fields: list, found_fields: list) -> None:
    """Print detailed field-level differences for a struct."""
    expected_dict = {f[1]: f[0] for f in expected_fields}
    found_dict = {f[1]: f[0] for f in found_fields}

    for field_name, field_type in expected_dict.items():
        if field_name not in found_dict:
            print(f"  Struct {name}: removed field '{field_name}'")
        elif found_dict[field_name] != field_type:
            print(
                f"  Struct {name}: field '{field_name}' type changed from '{field_type}' to '{found_dict[field_name]}'"
            )
    for field_name in found_dict:
        if field_name not in expected_dict:
            print(f"  Struct {name}: added field '{field_name}'")

    expected_common = [f[1] for f in expected_fields if f[1] in found_dict]
    found_common = [f[1] for f in found_fields if f[1] in expected_dict]
    if expected_common != found_common:
        print(f"  Struct {name}: fields were reordered")


def check_structs(expected: dict, found: dict) -> tuple[bool, bool]:
    has_errors = False
    has_allowed_changes = False

    for name, expected_info in expected.items():
        if name not in found:
            print(f"  Missing struct/class: {name}")
            has_errors = True
            continue
        found_info = found[name]

        if "basicsize" in expected_info:
            if "basicsize" not in found_info:
                print(f"  Struct {name}: basicsize no longer available")
                has_errors = True
            elif found_info["basicsize"] != expected_info["basicsize"]:
                print(
                    f"  Struct {name}: basicsize changed from {expected_info['basicsize']} to {found_info['basicsize']}"
                )
                has_errors = True

        if "fields" in expected_info:
            if "fields" not in found_info:
                print(f"  Struct {name}: field information no longer available")
                has_errors = True
            elif found_info["fields"] != expected_info["fields"]:
                _report_field_changes(name, expected_info["fields"], found_info["fields"])
                has_errors = True

    for name in found:
        if name not in expected:
            print(f"  Added struct/class: {name}")
            has_allowed_changes = True

    return has_errors, has_allowed_changes


######################################################################################
# FUNCTIONS


def get_functions(module: object) -> dict:
    pyx_capi = module.__pyx_capi__
    return {k: get_capsule_name(pyx_capi[k]) for k in sorted(pyx_capi.keys())}


def check_functions(expected: dict[str, str], found: dict[str, str]) -> tuple[bool, bool]:
    has_errors = False
    has_allowed_changes = False
    for k, v in expected.items():
        if k not in found:
            print(f"  Missing symbol: {k}")
            has_errors = True
        elif found[k] != v:
            print(f"  Changed symbol: {k}: expected {v}, got {found[k]}")
            has_errors = True
    for k, v in found.items():
        if k not in expected:
            print(f"  Added symbol: {k}")
            has_allowed_changes = True
    return has_errors, has_allowed_changes


######################################################################################
# MAIN


def compare(expected: dict, found: dict) -> tuple[bool, bool]:
    has_errors = False
    has_allowed_changes = False

    for func, name in [(check_functions, "functions"), (check_structs, "structs")]:
        errors, allowed_changes = func(expected[name], found[name])
        has_errors |= errors
        has_allowed_changes |= allowed_changes

    return has_errors, has_allowed_changes


def module_to_json(module: object) -> dict:
    """
    Extracts information about a Cython-compiled .so into JSON-serializable information.
    """
    return {"functions": get_functions(module), "structs": get_structs(module)}


def check(package: str, abi_dir: Path) -> bool:
    build_dir = get_package_path(package)

    has_errors = False
    has_allowed_changes = False
    for abi_path in Path(abi_dir).glob(f"**/*{ABI_SUFFIX}"):
        so_path = abi_path_to_so_path(abi_path, build_dir, abi_dir)
        if so_path.is_file():
            try:
                module = import_from_path(package, build_dir, so_path)
            except ImportError:
                print(f"Failed to import module for {so_path.relative_to(build_dir)}")
                has_errors = True
                continue
            if is_cython_module(module):
                found_json = module_to_json(module)
                with open(abi_path, encoding="utf-8") as f:
                    expected_json = json.load(f)
                print(f"Checking module: {so_path.relative_to(build_dir)}")
                check_errors, check_allowed_changes = compare(expected_json, found_json)
                has_errors |= check_errors
                has_allowed_changes |= check_allowed_changes
            else:
                print(f"Module no longer has an exposed ABI or is no longer Cython: {so_path.relative_to(build_dir)}")
                has_errors = True
        else:
            print(f"No module found for {abi_path.relative_to(abi_dir)}")
            has_errors = True

    for so_path in Path(build_dir).glob(f"**/*{EXT_SUFFIX}"):
        module = import_from_path(package, build_dir, so_path)
        if hasattr(module, "__pyx_capi__"):
            abi_path = so_path_to_abi_path(so_path, build_dir, abi_dir)
            if not abi_path.is_file():
                print(f"New module added {so_path.relative_to(build_dir)}")
                has_allowed_changes = True

    print()
    if has_errors:
        print("ERRORS FOUND")
        return True
    elif has_allowed_changes:
        print("Allowed changes found.")
    else:
        print("No changes found.")
    return False


def generate(package: str, abi_dir: Path) -> bool:
    if abi_dir.is_dir():
        print(f"ABI directory {abi_dir} already exists. Please remove it before regenerating.")
        return True

    build_dir = get_package_path(package)
    for so_path in Path(build_dir).glob(f"**/*{EXT_SUFFIX}"):
        try:
            module = import_from_path(package, build_dir, so_path)
        except ImportError:
            print(f"Failed to import module: {so_path.relative_to(build_dir)}")
            continue
        if is_cython_module(module):
            print(f"Generating ABI from {so_path.relative_to(build_dir)}")
            abi_path = so_path_to_abi_path(so_path, build_dir, abi_dir)
            abi_path.parent.mkdir(parents=True, exist_ok=True)
            with open(abi_path, "w", encoding="utf-8") as f:
                json.dump(module_to_json(module), f, indent=2)

    return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="check_cython_abi", description="Checks for changes in the Cython ABI of a given package"
    )

    subparsers = parser.add_subparsers()

    gen_parser = subparsers.add_parser("generate", help="Regenerate the ABI files")
    gen_parser.set_defaults(func=generate)
    gen_parser.add_argument("package", help="Python package to collect data from")
    gen_parser.add_argument("dir", help="Output directory to save data to")

    check_parser = subparsers.add_parser("check", help="Check the API against existing ABI files")
    check_parser.set_defaults(func=check)
    check_parser.add_argument("package", help="Python package to collect data from")
    check_parser.add_argument("dir", help="Input directory to read data from")

    args = parser.parse_args()
    if hasattr(args, "func"):
        if args.func(args.package, Path(args.dir)):
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)
