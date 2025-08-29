#!/usr/bin/env python3
"""
Convert semantic_ids.json (integer keys) to semantic_ids_fixed.json (ASIN keys)
"""

import argparse
import json
import os
from typing import Dict, List, Any

def convert_ids(
    semantic_ids_path: str,
    datamaps_path: str,
    output_path: str
) -> Dict[str, Any]:
    """Converts a JSON file from integer keys to ASIN keys."""
    
    # Check for file existence before proceeding
    if not os.path.exists(semantic_ids_path):
        raise FileNotFoundError(f"Input file not found: {semantic_ids_path}")
    if not os.path.exists(datamaps_path):
        raise FileNotFoundError(f"Datamaps file not found: {datamaps_path}")

    # Load data
    with open(semantic_ids_path, 'r') as f:
        semantic_data = json.load(f)
    with open(datamaps_path, 'r') as f:
        datamaps = json.load(f)

    # Create mapping from ID to ASIN
    id2item = {str(item_id): asin for asin, item_id in datamaps['item2id'].items()}
    
    # Perform conversion and preserve metadata
    converted_semantic_ids = {
        id2item[item_id_str]: semantic_id
        for item_id_str, semantic_id in semantic_data['semantic_ids'].items()
        if item_id_str in id2item
    }

    # Prepare output data
    output_data = semantic_data.copy()
    output_data['semantic_ids'] = converted_semantic_ids

    # Save converted file
    with open(output_path, 'w') as f:
        json.dump(output_data, f, separators=(',', ':'))
    
    return output_data

def main() -> None:
    """Parses arguments and runs the conversion."""
    parser = argparse.ArgumentParser(description='Convert semantic IDs from integer to ASIN keys.')
    parser.add_argument('--input', default='beauty/semantic_ids.json',
                       help='Input semantic_ids.json file.')
    parser.add_argument('--datamaps', default='beauty/datamaps.json',
                       help='Datamaps file.')
    parser.add_argument('--output', default='beauty/semantic_ids_fixed.json',
                       help='Output file.')
    args = parser.parse_args()
    convert_ids(args.input, args.datamaps, args.output)

if __name__ == "__main__":
    main()