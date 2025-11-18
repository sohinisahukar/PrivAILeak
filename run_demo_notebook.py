#!/usr/bin/env python3
"""
Quick script to run the demo notebook code with progress indicators
"""

import json
import sys
from pathlib import Path
import time

print("="*70)
print("PRIVAI-LEAK DEMO NOTEBOOK RUNNER")
print("="*70)
print()

# Load notebook
notebook_path = Path(__file__).parent / "Demo_Presentation.ipynb"
print(f"📖 Loading notebook: {notebook_path}")
with open(notebook_path, 'r') as f:
    nb = json.load(f)

# Extract code cells
code_cells = []
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if source.strip():
            code_cells.append((i, source))

print(f"✅ Found {len(code_cells)} code cells to execute\n")

# Create shared namespace
namespace = {
    '__name__': '__main__',
    '__builtins__': __builtins__,
    '__file__': __file__
}

# Execute cells with progress
for idx, (cell_num, code) in enumerate(code_cells, 1):
    print(f"\n{'='*70}")
    print(f"📝 CELL {idx}/{len(code_cells)} (Notebook cell {cell_num})")
    print('='*70)
    
    # Show first line of code
    first_line = code.split('\n')[0][:80]
    print(f"Code: {first_line}...")
    print('-'*70)
    
    start_time = time.time()
    
    try:
        exec(code, namespace)
        elapsed = time.time() - start_time
        print(f"✅ Completed in {elapsed:.2f}s")
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ Error after {elapsed:.2f}s: {type(e).__name__}: {e}")
        
        # For certain errors, continue anyway
        if 'results' in str(e).lower() or 'not defined' in str(e).lower():
            print("   (This might be expected if previous cells failed)")
            continue
        else:
            import traceback
            traceback.print_exc()
            print("\n⚠️  Stopping execution due to error")
            break

print("\n" + "="*70)
print("✅ DEMO NOTEBOOK EXECUTION COMPLETE")
print("="*70)
print("\n💡 Note: Model loading and text generation can take 30-60 seconds per model")
print("   This is normal - GPT-2 models are large (~500MB each)")

