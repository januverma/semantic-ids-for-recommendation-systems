#!/usr/bin/env python3
"""
Analyze and visualize generated semantic IDs
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import os

def load_semantic_ids(semantic_ids_path: str):
    """Load semantic IDs from JSON file"""
    if not os.path.exists(semantic_ids_path):
        print(f"❌ Semantic IDs file not found: {semantic_ids_path}")
        print("   Please run generate_semantic_ids.py first")
        return None
    
    with open(semantic_ids_path, 'r') as f:
        data = json.load(f)
    
    # Convert string keys back to integers and lists back to tuples
    semantic_ids = {int(k): tuple(v) for k, v in data['semantic_ids'].items()}
    
    return {
        'semantic_ids': semantic_ids,
        'metadata': {k: v for k, v in data.items() if k != 'semantic_ids'}
    }

def analyze_semantic_id_distribution(semantic_ids):
    """Analyze the distribution of semantic IDs"""
    print("📊 SEMANTIC ID ANALYSIS")
    print("=" * 40)
    
    # Basic statistics
    total_items = len(semantic_ids)
    print(f"Total items: {total_items:,}")
    
    # Analyze each level
    num_levels = len(next(iter(semantic_ids.values())))
    print(f"Number of levels: {num_levels}")
    
    level_distributions = []
    
    for level in range(num_levels):
        level_values = [sid[level] for sid in semantic_ids.values()]
        level_counter = Counter(level_values)
        level_distributions.append(level_counter)
        
        print(f"\\nLevel {level + 1} distribution:")
        print(f"  Unique values: {len(level_counter)}")
        print(f"  Most common: {level_counter.most_common(5)}")
        print(f"  Least common: {level_counter.most_common()[-5:]}")
    
    return level_distributions

def analyze_collisions(semantic_ids):
    """Analyze collision patterns"""
    print("\\n⚠️  COLLISION ANALYSIS")
    print("=" * 40)
    
    # Group by first 3 levels (original semantic ID)
    original_ids = defaultdict(list)
    for item_id, semantic_id in semantic_ids.items():
        original_id = semantic_id[:-1]  # Remove collision suffix
        original_ids[original_id].append((item_id, semantic_id[-1]))
    
    # Find collisions
    collisions = {k: v for k, v in original_ids.items() if len(v) > 1}
    
    print(f"Original semantic IDs with collisions: {len(collisions)}")
    print(f"Total items affected by collisions: {sum(len(items) for items in collisions.values())}")
    
    if collisions:
        # Analyze collision sizes
        collision_sizes = [len(items) for items in collisions.values()]
        print(f"\\nCollision size distribution:")
        size_counter = Counter(collision_sizes)
        for size, count in sorted(size_counter.items()):
            print(f"  {size} items: {count} collisions")
        
        # Show largest collisions
        largest_collisions = sorted(collisions.items(), key=lambda x: len(x[1]), reverse=True)
        print(f"\\nLargest collisions (top 5):")
        for i, (semantic_id, items) in enumerate(largest_collisions[:5]):
            print(f"  {i+1}. ID {semantic_id}: {len(items)} items")
            for item_id, suffix in items[:3]:
                print(f"     Item {item_id} → {semantic_id + (suffix,)}")
            if len(items) > 3:
                print(f"     ... and {len(items) - 3} more")
    
    return collisions

def visualize_semantic_ids(semantic_ids, level_distributions, output_dir='plots'):
    """Create visualizations for semantic ID analysis"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    num_levels = len(level_distributions)
    
    # Plot distribution for each level
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Semantic ID Level Distributions', fontsize=16)
    
    for level in range(min(num_levels, 4)):
        row = level // 2
        col = level % 2
        ax = axes[row, col]
        
        level_values = list(level_distributions[level].values())
        
        # Histogram
        ax.hist(level_values, bins=50, alpha=0.7, edgecolor='black')
        ax.set_title(f'Level {level + 1} Distribution')
        ax.set_xlabel('Usage Count')
        ax.set_ylabel('Number of Codes')
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        mean_usage = np.mean(level_values)
        ax.axvline(mean_usage, color='red', linestyle='--', 
                  label=f'Mean: {mean_usage:.1f}')
        ax.legend()
    
    # Remove empty subplots
    if num_levels < 4:
        for level in range(num_levels, 4):
            row = level // 2
            col = level % 2
            fig.delaxes(axes[row, col])
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/semantic_id_distributions.png', dpi=300, bbox_inches='tight')
    print(f"📊 Saved distribution plot to {output_dir}/semantic_id_distributions.png")
    
    # Plot collision analysis
    plt.figure(figsize=(10, 6))
    
    # Group by collision suffix to show collision resolution
    suffix_counts = Counter(sid[-1] for sid in semantic_ids.values())
    
    plt.bar(suffix_counts.keys(), suffix_counts.values(), alpha=0.7)
    plt.xlabel('Collision Suffix')
    plt.ylabel('Number of Items')
    plt.title('Collision Resolution Distribution')
    plt.grid(True, alpha=0.3)
    
    # Highlight collision items
    collision_items = sum(count for suffix, count in suffix_counts.items() if suffix > 0)
    total_items = len(semantic_ids)
    plt.text(0.7, 0.9, f'Collisions: {collision_items}/{total_items} ({collision_items/total_items*100:.2f}%)',
             transform=plt.gca().transAxes, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/collision_analysis.png', dpi=300, bbox_inches='tight')
    print(f"📊 Saved collision plot to {output_dir}/collision_analysis.png")

def analyze_semantic_similarity(semantic_ids, sample_size=1000):
    """Analyze semantic similarity patterns"""
    print("\\n🔍 SEMANTIC SIMILARITY ANALYSIS")
    print("=" * 40)
    
    # Sample items for analysis
    sampled_items = dict(list(semantic_ids.items())[:sample_size])
    
    # Calculate pairwise distances
    def hamming_distance(id1, id2):
        return sum(a != b for a, b in zip(id1[:-1], id2[:-1]))  # Exclude collision suffix
    
    distances = []
    for i, (item1, id1) in enumerate(sampled_items.items()):
        for j, (item2, id2) in enumerate(sampled_items.items()):
            if i < j:  # Avoid duplicates
                dist = hamming_distance(id1, id2)
                distances.append(dist)
    
    distances = np.array(distances)
    
    print(f"Analyzed {len(distances):,} pairwise distances")
    print(f"Mean Hamming distance: {distances.mean():.2f}")
    print(f"Std Hamming distance: {distances.std():.2f}")
    print(f"Distance range: {distances.min()} - {distances.max()}")
    
    # Distance distribution
    dist_counts = Counter(distances)
    print(f"\\nDistance distribution:")
    for dist in sorted(dist_counts.keys()):
        count = dist_counts[dist]
        percentage = count / len(distances) * 100
        print(f"  Distance {dist}: {count:,} pairs ({percentage:.1f}%)")
    
    return distances


def main():
    """Main analysis function"""
    
    semantic_ids_path = 'beauty/semantic_ids.json'
    
    print("🔍 SEMANTIC ID ANALYZER")
    print("=" * 50)
    
    # Load semantic IDs
    data = load_semantic_ids(semantic_ids_path)
    if data is None:
        return
    
    semantic_ids = data['semantic_ids']
    metadata = data['metadata']
    
    print(f"✅ Loaded {len(semantic_ids):,} semantic IDs")
    print(f"📋 Metadata: {metadata}")
    
    # Perform analyses
    level_distributions = analyze_semantic_id_distribution(semantic_ids)
    collisions = analyze_collisions(semantic_ids)
    distances = analyze_semantic_similarity(semantic_ids)
    
    # Generate visualizations
    visualize_semantic_ids(semantic_ids, level_distributions)
    
    print("\\n🎉 Analysis completed!")
    print("\\n📁 Check the 'plots' directory for visualizations")

if __name__ == "__main__":
    main()