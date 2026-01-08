#!/usr/bin/env python3
"""
Parse Criterion benchmark output and generate markdown tables.

Usage:
    cat bench_output.txt | python3 format_benchmarks.py > CI_Bench_Results.md
    # or
    python3 format_benchmarks.py < bench_output.txt > CI_Bench_Results.md
"""

import sys
import re
from collections import defaultdict

# Regex to strip ANSI escape codes
ANSI_ESCAPE = re.compile(r'\x1b\[[0-9;]*m')


def strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text."""
    return ANSI_ESCAPE.sub('', text)


def parse_time(value: str, unit: str) -> float:
    """Convert time value to microseconds."""
    val = float(value)
    if unit == 'ns':
        return val / 1000.0
    if unit in ('us', 'µs'):
        return val
    if unit == 'ms':
        return val * 1000.0
    if unit == 's':
        return val * 1000000.0
    return val


def format_time(val_us: float) -> str:
    """Format microseconds to human-readable string."""
    if val_us < 1.0:
        return f"{val_us * 1000:.2f} ns"
    elif val_us < 1000.0:
        return f"{val_us:.2f} µs"
    elif val_us < 1000000.0:
        return f"{val_us / 1000:.2f} ms"
    else:
        return f"{val_us / 1000000:.2f} s"


def main():
    # Data structure: data[group][expression][variant] = time_us
    data = defaultdict(lambda: defaultdict(dict))

    # Regex for benchmark name: e.g. "1_parse/symb_anafis/Normal PDF"
    # Groups: 1=Group, 2=Variant, 3=Expression
    name_pattern = re.compile(r'^(\w+)/([^\s/]+)/(.+)$')

    # Regex for time line: e.g. "time:   [2.6508 us 2.6534 us 2.6561 us]"
    time_pattern = re.compile(
        r'^time:\s+\[[\d.]+\s+\w+\s+([\d.]+)\s+(\w+)\s+[\d.]+\s+\w+\]'
    )

    # Lines to ignore (Criterion comparison output from future runs)
    ignore_patterns = [
        re.compile(r'^change:\s+\['),        # change: [-1.2% +0.5% +2.1%]
        re.compile(r'^Performance has'),      # Performance has improved/regressed
        re.compile(r'^Found \d+ outliers'),   # Found N outliers
        re.compile(r'^\d+\s+\(\d+'),          # Outlier detail lines
        re.compile(r'^Benchmarking'),         # Benchmarking progress lines
        re.compile(r'^Warning:'),             # Warning lines
        re.compile(r'^Gnuplot'),              # Gnuplot not found
        re.compile(r'^Running'),              # Running benches/...
        re.compile(r'^Compiling'),            # Compiling crate
        re.compile(r'^Downloading'),          # Downloading crates
        re.compile(r'^Downloaded'),           # Downloaded crate
        re.compile(r'^Updating'),             # Updating crates.io
        re.compile(r'^Locking'),              # Locking packages
        re.compile(r'^Finished'),             # Finished build
        re.compile(r'^running \d+ tests'),    # Test output
        re.compile(r'^test .* \.\.\. ignored'),  # Test ignored
        re.compile(r'^test result:'),         # Test result line
    ]

    current_benchmark = None
    
    # Metadata to extract
    metadata = {
        "date": None,
        "commit": None,
        "version": None,
        "rust_version": None,
    }

    # Read all lines from stdin
    input_lines = sys.stdin.readlines()
    
    # First pass: extract metadata from header lines
    for line in input_lines:
        line_clean = strip_ansi(line).strip()
        
        if line_clean.startswith("Date:"):
            metadata["date"] = line_clean[5:].strip()
        elif line_clean.startswith("Commit:"):
            metadata["commit"] = line_clean[7:].strip()
        elif "Compiling symb_anafis v" in line_clean:
            # Extract version from "Compiling symb_anafis v0.5.1"
            match = re.search(r'symb_anafis v([\d.]+)', line_clean)
            if match:
                metadata["version"] = match.group(1)
        elif "Rust" in line_clean and "compatible" in line_clean:
            # Extract from "Locking 167 packages to latest Rust 1.92.0 compatible versions"
            match = re.search(r'Rust ([\d.]+)', line_clean)
            if match:
                metadata["rust_version"] = match.group(1)
    
    # Second pass: parse benchmark data
    for line in input_lines:
        line = strip_ansi(line).strip()

        # Skip empty lines
        if not line:
            continue

        # Skip ignored patterns
        if any(pat.match(line) for pat in ignore_patterns):
            continue

        # Check for benchmark name match
        name_match = name_pattern.match(line)
        if name_match:
            group, variant, expr = name_match.groups()
            # Normalize names: remove trailing slashes
            variant = variant.rstrip('/')
            expr = expr.rstrip('/')
            current_benchmark = (group, variant, expr)
            continue

        # Check for time match
        time_match = time_pattern.match(line)
        if time_match and current_benchmark:
            val, unit = time_match.groups()
            time_us = parse_time(val, unit)

            group, variant, expr = current_benchmark
            data[group][expr][variant] = time_us

    # Generate output
    output_lines = []

    # Header with metadata
    output_lines.append("# CI Benchmark Results")
    output_lines.append("")
    
    # Add metadata table if we have any info
    if any(metadata.values()):
        if metadata["version"]:
            output_lines.append(f"**SymbAnaFis Version:** {metadata['version']}  ")
        if metadata["date"]:
            output_lines.append(f"**Date:** {metadata['date']}  ")
        if metadata["commit"]:
            short_commit = metadata["commit"][:12] if metadata["commit"] else ""
            output_lines.append(f"**Commit:** `{short_commit}`  ")
        if metadata["rust_version"]:
            output_lines.append(f"**Rust:** {metadata['rust_version']}  ")
        output_lines.append("")
    
    output_lines.append("> Auto-generated from Criterion benchmark output")
    output_lines.append("")

    # Order of groups to print
    groups_order = [
        "1_parse",
        "2_diff",
        "3_diff_simplified",
        "4_simplify",
        "5_compile",
        "6_eval_1000pts",
        "7_full_pipeline",
        "eval_methods_1000pts",
        "eval_scaling",
        "multi_expr_batch",
        "eval_api_comparison",
        "large_expr_100",
        "large_expr_300"
    ]

    readable_groups = {
        "1_parse": "1. Parsing (String → AST)",
        "2_diff": "2. Differentiation",
        "3_diff_simplified": "3. Differentiation + Simplification",
        "4_simplify": "4. Simplification Only",
        "5_compile": "5. Compilation",
        "6_eval_1000pts": "6. Evaluation (1000 points)",
        "7_full_pipeline": "7. Full Pipeline",
        "eval_methods_1000pts": "Parallel: Evaluation Methods (1k pts)",
        "eval_scaling": "Parallel: Scaling (Points)",
        "multi_expr_batch": "Parallel: Multi-Expression Batch",
        "eval_api_comparison": "Parallel: API Comparison (10k pts)",
        "large_expr_100": "Large Expressions (100 terms)",
        "large_expr_300": "Large Expressions (300 terms)"
    }

    # Variant Display Names
    variant_map = {
        "symb_anafis": "SymbAnaFis",
        "symbolica": "Symbolica",
        "symb_anafis_light": "SymbAnaFis (Light)",
        "symb_anafis_full": "SymbAnaFis (Full)",
        "raw": "SA (Raw)",
        "simplified": "SA (Simplified)",
        "compiled_raw": "SA (Raw)",
        "compiled_simplified": "SA (Simplified)",
        "compiled_loop": "Compiled Loop",
        "eval_batch": "Eval Batch (SIMD)",
        "tree_walk_evaluate": "Tree Walk",
        "loop_evaluate": "Loop",
        "sequential_loops": "Sequential Loops",
        "eval_batch_per_expr": "Eval Batch (per expr)",
        "evaluate_parallel": "Evaluate Parallel",
        "eval_f64": "Eval F64 (SIMD)",
        "eval_f64_per_expr": "Eval F64 (per expr)",
        "symb_anafis_diff_only": "SA (Diff Only)",
        "symb_anafis_diff+simplify": "SA (Diff+Simp)",
        "symb_anafis_raw": "SA (Raw)",
        "symb_anafis_simplified": "SA (Simplified)",
        # Trailing slash variants from large_expr benchmarks
        "symb_anafis_raw/": "SA (Raw)",
        "symb_anafis_simplified/": "SA (Simplified)",
        "symbolica/": "Symbolica",
    }

    # Configuration per group:
    # (Preferred Columns, Primary Comparison Pair, Swap Rows/Cols)
    group_config = {
        "1_parse": (["symb_anafis", "symbolica"], ("symbolica", "symb_anafis"), False),
        "2_diff": (["symb_anafis_light", "symbolica"], ("symbolica", "symb_anafis_light"), False),
        "5_compile": (["simplified", "symbolica"], ("symbolica", "simplified"), False),
        "6_eval_1000pts": (["compiled_simplified", "symbolica"], ("symbolica", "compiled_simplified"), False),
        "7_full_pipeline": (["symb_anafis", "symbolica"], ("symbolica", "symb_anafis"), False),
        "eval_methods_1000pts": (["compiled_loop", "tree_walk_evaluate"], ("tree_walk_evaluate", "compiled_loop"), False),
        "eval_scaling": (["eval_batch", "loop_evaluate"], ("loop_evaluate", "eval_batch"), False),
        "large_expr_100": (
            ["symb_anafis", "symbolica", "symb_anafis_diff_only", "symb_anafis_diff+simplify", 
             "symb_anafis_simplified", "symb_anafis_raw"],
            ("symbolica", "symb_anafis"),
            True
        ),
        "large_expr_300": (
            ["symb_anafis", "symbolica", "symb_anafis_diff_only", "symb_anafis_diff+simplify",
             "symb_anafis_simplified", "symb_anafis_raw"],
            ("symbolica", "symb_anafis"),
            True
        ),
    }

    for group in groups_order:
        if group not in data:
            continue

        output_lines.append(f"## {readable_groups.get(group, group)}")
        output_lines.append("")

        # Get configuration for this group
        columns, comparison, swap_rows = group_config.get(group, (None, None, False))

        # Special handling for large_expr tables
        if group.startswith("large_expr_"):
            # Build simplified table: Operation | SymbAnaFis | Symbolica | Speedup
            output_lines.append("| Operation | SymbAnaFis | Symbolica | Speedup |")
            output_lines.append("| :--- | :---: | :---: | :---: |")
            
            # Collect data by operation
            # data[group][library][operation] structure
            group_data = data[group]
            
            # Parse operation
            sa_parse = group_data.get("symb_anafis", {}).get("1_parse")
            sy_parse = group_data.get("symbolica", {}).get("1_parse")
            if sa_parse and sy_parse:
                speedup = sy_parse / sa_parse
                output_lines.append(f"| Parse | **{format_time(sa_parse)}** | {format_time(sy_parse)} | **SA** ({speedup:.2f}x) |")
            
            # Diff (no simplify) - best of SA diff_only
            sa_diff = group_data.get("symb_anafis_diff_only", {}).get("2_diff")
            sy_diff = group_data.get("symbolica", {}).get("2_diff")
            if sa_diff and sy_diff:
                speedup = sy_diff / sa_diff
                output_lines.append(f"| Diff (no simplify) | **{format_time(sa_diff)}** | {format_time(sy_diff)} | **SA** ({speedup:.2f}x) |")
            
            # Diff+Simplify - SA only
            sa_diff_simp = group_data.get("symb_anafis_diff+simplify", {}).get("2_diff")
            if sa_diff_simp:
                output_lines.append(f"| Diff+Simplify | {format_time(sa_diff_simp)} | — | — |")
            
            # Compile (simplified)
            sa_compile = group_data.get("symb_anafis_simplified", {}).get("3_compile")
            sy_compile = group_data.get("symbolica", {}).get("3_compile")
            if sa_compile and sy_compile:
                speedup = sy_compile / sa_compile
                output_lines.append(f"| Compile (simplified) | **{format_time(sa_compile)}** | {format_time(sy_compile)} | **SA** ({speedup:.2f}x) |")
            
            # Eval 1000pts (simplified)
            sa_eval = group_data.get("symb_anafis_simplified", {}).get("4_eval_1000pts")
            sy_eval = group_data.get("symbolica", {}).get("4_eval_1000pts")
            if sa_eval and sy_eval:
                if sa_eval < sy_eval:
                    speedup = sy_eval / sa_eval
                    output_lines.append(f"| Eval 1000pts (simplified) | **{format_time(sa_eval)}** | {format_time(sy_eval)} | **SA** ({speedup:.2f}x) |")
                else:
                    speedup = sa_eval / sy_eval
                    output_lines.append(f"| Eval 1000pts (simplified) | {format_time(sa_eval)} | **{format_time(sy_eval)}** | **SY** ({speedup:.2f}x) |")
            
            output_lines.append("")
            output_lines.append("---")
            output_lines.append("")
            continue

        # Standard table processing for non-large_expr groups
        # Prepare Data View
        view_data = defaultdict(dict)

        if swap_rows:
            # Pivot data: rows become variants, columns become expressions
            for expr in data[group]:
                for var in data[group][expr]:
                    view_data[var][expr] = data[group][expr][var]
        else:
            view_data = data[group]

        # Determine Row Keys
        row_keys = sorted(view_data.keys())

        # Determine Columns
        available_cols = set()
        for r in row_keys:
            for c in view_data[r]:
                available_cols.add(c)

        if not columns:
            columns = sorted(list(available_cols))
        else:
            # Make a copy so we don't modify the config
            columns = list(columns)
            extras = [c for c in sorted(list(available_cols)) if c not in columns]
            columns.extend(extras)

        # Build Table Header
        header_cols = [variant_map.get(c, c) for c in columns]
        if swap_rows:
            first_col_name = "Operation"
        elif group == "eval_scaling":
            first_col_name = "Points"
        else:
            first_col_name = "Expression"

        header = f"| {first_col_name} | " + " | ".join(header_cols) + " | Speedup |"
        separator = "| :--- | " + " | ".join([":---:" for _ in columns]) + " | :---: |"

        output_lines.append(header)
        output_lines.append(separator)

        for row_key in row_keys:
            # For swapped rows (large expr), clean up row_key
            display_row = row_key
            if swap_rows:
                display_row = re.sub(r'^\d+_', '', row_key).capitalize()
                if "eval" in row_key:
                    display_row = "Evaluate"
                if "compile" in row_key:
                    display_row = "Compile"
                if "diff" in row_key:
                    display_row = "Diff"
                if "parse" in row_key:
                    display_row = "Parse"

            row = f"| {display_row} |"

            col_times = []
            for col in columns:
                if col in view_data[row_key]:
                    col_times.append(view_data[row_key][col])
                else:
                    col_times.append(None)

            valid_times = [t for t in col_times if t is not None]
            min_time = min(valid_times) if valid_times else 0

            for i, t in enumerate(col_times):
                if t is None:
                    row += " - |"
                else:
                    fmt = format_time(t)
                    if t == min_time and len(valid_times) > 1:
                        row += f" **{fmt}** |"
                    else:
                        row += f" {fmt} |"

            # Speedup Logic
            speedup_str = "-"
            if comparison:
                base_col, cand_col = comparison
                if base_col in view_data[row_key] and cand_col in view_data[row_key]:
                    t_base = view_data[row_key][base_col]
                    t_cand = view_data[row_key][cand_col]

                    if t_cand > 0:
                        winner = variant_map.get(cand_col, cand_col)
                        ratio = t_base / t_cand

                        if t_base < t_cand:
                            winner = variant_map.get(base_col, base_col)
                            ratio = t_cand / t_base

                        speedup_str = f"**{winner}** ({ratio:.2f}x)"

            row += f" {speedup_str} |"
            output_lines.append(row)

        output_lines.append("")
        output_lines.append("---")
        output_lines.append("")

    # Print all output
    print("\n".join(output_lines))


if __name__ == "__main__":
    main()
