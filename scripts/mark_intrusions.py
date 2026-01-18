import argparse
import os
import sys
import pandas as pd


def parse_indices(s: str):
    try:
        return [int(x.strip()) for x in s.split(',') if x.strip()]
    except Exception:
        raise argparse.ArgumentTypeError("indices must be a comma-separated list of integers (0-based)")


def main():
    parser = argparse.ArgumentParser(description="Mark selected rows in a predictions CSV as intrusions (attacks).")
    parser.add_argument("--input", required=True, help="Path to predictions CSV to modify")
    parser.add_argument("--output", default=None, help="Optional output path; if omitted, overwrite the input file (with a .bak backup)")
    parser.add_argument("--count", type=int, default=10, help="Number of rows to mark as attacks when indices not provided")
    parser.add_argument("--indices", type=parse_indices, default=None, help="Comma-separated 0-based row indices to mark (e.g., 0,5,12)")
    parser.add_argument("--confidence", type=float, default=0.95, help="Confidence to set for marked rows (0-1)")

    args = parser.parse_args()

    in_path = os.path.abspath(args.input)
    if not os.path.exists(in_path):
        print(f"Error: input file not found: {in_path}")
        sys.exit(1)

    df = pd.read_csv(in_path)
    required_cols = {"Prediction", "Is_Attack", "Confidence"}
    if not required_cols.issubset(set(df.columns)):
        print(f"Error: CSV missing required columns: {required_cols}. Found: {list(df.columns)}")
        sys.exit(1)

    # Determine rows to mark
    to_mark = []
    if args.indices:
        to_mark = [i for i in args.indices if 0 <= i < len(df)]
        if not to_mark:
            print("No valid indices within file length; nothing to do.")
            sys.exit(0)
    else:
        # Prefer normal rows to flip
        normal_idxs = df.index[df["Is_Attack"] == False].tolist()
        if not normal_idxs:
            # Fallback: take first N rows
            normal_idxs = list(range(len(df)))
        to_mark = normal_idxs[:max(0, args.count)]

    # Apply changes
    df.loc[to_mark, "Prediction"] = "Attack"
    df.loc[to_mark, "Is_Attack"] = True
    if "Confidence" in df.columns:
        df.loc[to_mark, "Confidence"] = float(args.confidence)

    # Save
    if args.output:
        out_path = os.path.abspath(args.output)
        df.to_csv(out_path, index=False)
        print(f"Wrote modified predictions to: {out_path}")
    else:
        # Backup
        bak_path = in_path + ".bak"
        try:
            os.replace(in_path, bak_path)
        except Exception:
            # If replace fails, try copy
            import shutil
            shutil.copy2(in_path, bak_path)
        df.to_csv(in_path, index=False)
        print(f"Overwrote input file; backup saved at: {bak_path}")

    # Summary
    total = len(df)
    attacks = int((df["Is_Attack"] == True).sum())
    normals = total - attacks
    print("Summary after edit:")
    print(f"  Total rows: {total}")
    print(f"  Attacks: {attacks}")
    print(f"  Normals: {normals}")
    print(f"  Marked indices: {to_mark}")


if __name__ == "__main__":
    main()
