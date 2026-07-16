#!/usr/bin/env python3
"""Utility: verify inspect.getsource works before JIT is triggered."""
import os
import sys
import inspect
import json


def inspect_function(module_path, module_name, func_name):
    """Inspect a function's source retrievability."""
    sys.path.insert(0, module_path)
    try:
        mod = __import__(module_name)
        fn = getattr(mod, func_name, None)
        if fn is None:
            return {"status": "FAIL", "reason": f"Function {func_name} not found in {module_name}"}
    except Exception as e:
        return {"status": "FAIL", "reason": str(e)}

    result = {
        "module": getattr(fn, '__module__', 'N/A'),
        "qualname": getattr(fn, '__qualname__', 'N/A') if hasattr(fn, '__qualname__') else 'JitCallableWrapper',
        "file": None,
        "source_file": None,
        "getsource_ok": False,
        "source_first_200_chars": None,
        "wrapped_type": type(fn).__name__,
    }

    try:
        result["file"] = inspect.getfile(fn)
    except Exception as e:
        result["file_error"] = str(e)

    try:
        result["source_file"] = inspect.getsourcefile(fn)
    except Exception as e:
        result["source_file_error"] = str(e)

    # JitCallableWrapper: try original_func first
    orig_func = getattr(fn, '_original_func', fn)
    try:
        source = inspect.getsource(orig_func)
        result["getsource_ok"] = True
        result["source_len"] = len(source)
        result["source_first_200_chars"] = source[:200]
    except Exception as e:
        result["getsource_error"] = str(e)
        if orig_func is not fn:
            result["note"] = "JitCallableWrapper inspected via _original_func"

    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--module-path", required=True)
    parser.add_argument("--module-name", required=True)
    parser.add_argument("--func-name", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    result = inspect_function(args.module_path, args.module_name, args.func_name)
    print(json.dumps(result, indent=2))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)

    sys.exit(0 if result.get("getsource_ok") else 1)
