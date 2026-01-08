
import sys
import re
from collections import defaultdict

def parse_time(value, unit):
    val = float(value)
    if unit == 'ns': return val / 1000.0
    if unit == 'us' or unit == 'µs': return val
    if unit == 'ms': return val * 1000.0
    if unit == 's': return val * 1000000.0
    return val

def format_time(val_us):
    if val_us < 1.0:
        return f"{val_us*1000:.2f} ns"
    elif val_us < 1000.0:
        return f"{val_us:.2f} µs"
    elif val_us < 1000000.0:
        return f"{val_us/1000:.2f} ms"
    else:
        return f"{val_us/1000000:.2f} s"

def main():
    # Data structure: data[group][expression][variant] = time_us
    data = defaultdict(lambda: defaultdict(dict))
    
    # Regex for detailed Criterion output (not simple bar output)
    # Example line: 1_parse/symb_anafis/Normal PDF time:   [2.6508 us 2.6534 us 2.6561 us]
    # But sometimes the output is split across lines.
    # We will look for the "time: [ ... median ... ]" line and associate it with the last seen benchmark name.
    
    # Regex to catch the benchmark name line
    # Matches: "1_parse/symb_anafis/Normal PDF"
    # Groups: 1=Group, 2=Variant, 3=Expression
    name_pattern = re.compile(r'^(\w+)/([\w_]+)/(.+)$')
    
    # Regex to catch the time line (after strip())
    # Matches: "time:   [2.6508 us 2.6534 us 2.6561 us]"
    # We allow optional space at start just in case, but strip() removed them.
    time_pattern = re.compile(r'^time:\s+\[[\d.]+\s+\w+\s+([\d.]+)\s+(\w+)\s+[\d.]+\s+\w+\]')

    current_benchmark = None
    
    # Read from stdin
    input_lines = sys.stdin.readlines()
    
    for line in input_lines:
        line = line.strip()
        
        # Check for benchmark name match
        name_match = name_pattern.match(line)
        if name_match:
            current_benchmark = name_match.groups()
            continue

        # Check for time match
        time_match = time_pattern.match(line)
        # Also try matching roughly if exact match fails
        if not time_match and "time:" in line:
             pass

        if time_match and current_benchmark:
            val, unit = time_match.groups()
            time_us = parse_time(val, unit)
            
            group, variant, expr = current_benchmark
            data[group][expr][variant] = time_us

    # Order of groups to print
    groups_order = [
        "1_parse",
        "2_diff",
        "3_diff_simplified", 
        "4_simplify",
        "5_compile",
        "6_eval_1000pts",
        "7_full_pipeline"
    ]
    
    readable_groups = {
        "1_parse": "1. Parsing (String → AST)",
        "2_diff": "2. Differentiation",
        "3_diff_simplified": "3. Differentiation + Simplification",
        "4_simplify": "4. Simplification Only",
        "5_compile": "5. Compilation",
        "6_eval_1000pts": "6. Evaluation (1000 points)",
        "7_full_pipeline": "7. Full Pipeline"
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
        
        # Parallel variants
        "compiled_loop": "Compiled Loop",
        "eval_batch": "Eval Batch (SIMD)",
        "tree_walk_evaluate": "Tree Walk",
        "loop_evaluate": "Loop",
        "sequential_loops": "Sequential Loops",
        "eval_batch_per_expr": "Eval Batch (per expr)",
        "evaluate_parallel": "Evaluate Parallel",
        "eval_f64": "Eval F64 (SIMD)",

        # Large Expr specific variants (mapped for cleaner columns)
        "1_parse": "Parse",
        "2_diff": "Diff",
        "3_compile": "Compile",
        "4_eval_1000pts": "Eval (1k pts)",
        
        "symb_anafis_diff_only": "SymbAnaFis (Diff Only)",
        "symb_anafis_diff+simplify": "SymbAnaFis (Diff+Simp)",
        "symb_anafis_raw": "SymbAnaFis (Raw)",
        "symb_anafis_simplified": "SymbAnaFis (Simp)"
    }

    # Configuration per group: 
    # (Preferred Columns, Primary Comparison Pair, Swap Rows/Cols)
    # Pair format: (Baseline, Candidate) where Speedup = Baseline / Candidate
    # Swap Rows/Cols: If True, Expressions become Columns and Variants become Rows (used for Large Exprs setup)
    
    group_config = {
        # Standard Main Benchmarks
        "1_parse": (["symb_anafis", "symbolica"], ("symbolica", "symb_anafis"), False),
        # ... (other standard ones remain same, implicitly handled by default logic if not specified)
        "2_diff": (["symb_anafis_light", "symbolica"], ("symbolica", "symb_anafis_light"), False),
        "5_compile": (["simplified", "symbolica"], ("symbolica", "simplified"), False),
        "6_eval_1000pts": (["compiled_simplified", "symbolica"], ("symbolica", "compiled_simplified"), False),
        
        # Parallel Benchmarks 
        "eval_methods_1000pts": (["compiled_loop", "eval_batch", "tree_walk_evaluate"], ("compiled_loop", "eval_batch"), False),
        
        # Large Expressions 
        # For Large Exprs, we used `swap_rows=True`. 
        # Row Keys will be: 1_parse, 2_diff, 3_compile...
        # Col Keys will be: symb_anafis, symbolica, symb_anafis_diff_only, etc.
        # We need to list ALL columns we want to show here.
        "large_expr_100": (
            ["symb_anafis", "symbolica", "symb_anafis_diff_only", "symb_anafis_simplified"], 
            None, # Comparison is tricky because baselines change per row (Parse vs Diff etc). We let it be "-" or handle simpler.
            True
        ),
        "large_expr_300": (
            ["symb_anafis", "symbolica", "symb_anafis_diff_only", "symb_anafis_simplified"], 
            None,
            True
        ),
    }

    # Remove Debug Print


    # Add parallel and large groups to order
    groups_order.extend([
        "eval_methods_1000pts",
        "eval_scaling",
        "multi_expr_batch",
        "eval_api_comparison",
        "large_expr_100",
        "large_expr_300"
    ])
    
    print(f"DEBUG: Data Keys: {list(data.keys())}")
    
    # Update readable names
    readable_groups.update({
        "eval_methods_1000pts": "Parallel: Evaluation Methods (1k pts)",
        "eval_scaling": "Parallel: Scaling (Points)",
        "multi_expr_batch": "Parallel: Multi-Expression Batch",
        "eval_api_comparison": "Parallel: API Comparison (10k pts)",
        "large_expr_100": "Large Expressions (100 terms)",
        "large_expr_300": "Large Expressions (300 terms)"
    })
    
    for group in groups_order:
        if group not in data:
            continue
            
        print(f"## {readable_groups.get(group, group)}")
        print("")
        
        # Get configuration for this group
        columns, comparison, swap_rows = group_config.get(group, (None, None, False))
        
        # Prepare Data View
        # If swap_rows is True:
        #   We want Rows = Variants (e.g. 1_parse)
        #   We want Cols = Expressions (e.g. symb_anafis)
        #   Current Data Structure: data[group][expr][variant]
        #   We need to invert: view[row_key][col_key] = time
        
        view_data = defaultdict(dict)
        
        if swap_rows:
            # Pivot data
            # Iterate original structure
            for expr in data[group]: # Original Expr (now Col) e.g. symb_anafis
                for var in data[group][expr]: # Original Variant (now Row) e.g. 1_parse
                     view_data[var][expr] = data[group][expr][var]
        else:
            # Standard View
            view_data = data[group]

        # Determine Row Keys (Expressions or Operations)
        row_keys = sorted(view_data.keys())
        
        # Determine Columns (Variants or Libraries)
        available_cols = set()
        for r in row_keys:
            for c in view_data[r]:
                available_cols.add(c)
        
        if not columns:
            columns = sorted(list(available_cols))
        else:
            extras = [c for c in sorted(list(available_cols)) if c not in columns]
            columns.extend(extras)
            
        # Build Table Header
        header_cols = [variant_map.get(c, c) for c in columns]
        
        # If swapped, the first column is "Operation", else "Expression"
        first_col_name = "Operation" if swap_rows else "Expression"
        
        header = f"| {first_col_name} | " + " | ".join(header_cols) + " | Speedup |"
        separator = "| :--- | " + " | ".join([":---:" for _ in columns]) + " | :---: |"
        
        print(header)
        print(separator)
        
        for row_key in row_keys:
            # For swapped rows (large expr), row_key might be "1_parse", clean it up
            display_row = row_key
            if swap_rows:
                # remove sorting prefix like "1_"
                display_row = re.sub(r'^\d+_', '', row_key).capitalize()
                # fix specific names
                if "eval" in row_key: display_row = "Evaluate"
                if "compile" in row_key: display_row = "Compile"
                if "diff" in row_key: display_row = "Diff"
                if "parse" in row_key: display_row = "Parse"

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
            print(row)
        
        print("")
        print("---")
        print("")

if __name__ == "__main__":
    main()
