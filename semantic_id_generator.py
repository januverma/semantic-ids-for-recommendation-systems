"""
Semantic ID Generator using KMeans Residual Quantization

This module generates semantic IDs for items using their product metadata (title, brand, 
category, price, description) through KMeans residual quantization with collision handling.
"""

import json
import gzip
import pickle
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import KMeans
    import torch
    print("All required packages loaded successfully!")
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Please install: pip install sentence-transformers scikit-learn torch")


class SemanticIDGenerator:
    """
    Generate semantic IDs for items using KMeans residual quantization
    """
    
    def __init__(self, num_levels: int = 3, codebook_size: int = 256, 
                 embedding_model: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the semantic ID generator
        
        Args:
            num_levels: Number of quantization levels (will become 4 after collision handling)
            codebook_size: Size of each codebook (number of clusters)
            embedding_model: Sentence transformer model name
        """
        self.num_levels = num_levels
        self.codebook_size = codebook_size
        self.embedding_model_name = embedding_model
        
        # Initialize sentence transformer
        print(f"Loading sentence transformer model: {embedding_model}")
        self.encoder = SentenceTransformer(embedding_model)
        
        # Initialize codebooks (KMeans models)
        self.codebooks = []
        
        # Collision handling
        self.semantic_id_to_items = defaultdict(list)
        self.item_to_semantic_id = {}
        self.collision_resolved = False
        
    def extract_item_text(self, product: Dict[str, Any]) -> str:
        """
        Extract and combine text features from product metadata
        
        Args:
            product: Product metadata dictionary
            
        Returns:
            Combined text representation
        """
        text_parts = []
        
        # Title
        if 'title' in product and product['title']:
            text_parts.append(f"Title: {product['title']}")
        
        # Brand (if available)
        if 'brand' in product and product['brand']:
            text_parts.append(f"Brand: {product['brand']}")
        
        # Categories
        if 'categories' in product and product['categories']:
            categories = []
            for cat_path in product['categories']:
                if isinstance(cat_path, list):
                    categories.extend(cat_path)
                else:
                    categories.append(str(cat_path))
            if categories:
                text_parts.append(f"Category: {' '.join(categories[:3])}")  # Limit to first 3
        
        # Price (from sales rank as proxy)
        if 'salesRank' in product and product['salesRank']:
            rank_info = []
            for category, rank in product['salesRank'].items():
                rank_info.append(f"{category} rank {rank}")
            if rank_info:
                text_parts.append(f"Sales: {' '.join(rank_info[:2])}")  # Limit to first 2
        
        # Description
        if 'description' in product and product['description']:
            # Truncate long descriptions
            desc = str(product['description'])[:200]
            text_parts.append(f"Description: {desc}")
        
        # Combine all parts
        combined_text = " | ".join(text_parts)
        
        # Fallback if no text found
        if not combined_text.strip():
            asin = product.get('asin', 'unknown')
            combined_text = f"Product: {asin}"
            
        return combined_text
    
    def load_product_data(self, metadata_path: str, datamaps_path: str) -> Tuple[Dict[str, str], Dict[int, str]]:
        """
        Load product metadata and create text representations
        
        Args:
            metadata_path: Path to meta.json.gz file
            datamaps_path: Path to datamaps.json file
            
        Returns:
            Tuple of (asin_to_text, item_id_to_text) dictionaries
        """
        print("Loading product metadata...")
        
        # Load datamaps for item ID mapping
        with open(datamaps_path, 'r') as f:
            datamaps = json.load(f)
        
        asin_to_text = {}
        item_count = 0
        
        # Load and process metadata
        with gzip.open(metadata_path, 'rt', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                    
                try:
                    # Try standard JSON parsing first
                    try:
                        product = json.loads(line)
                    except json.JSONDecodeError:
                        # If that fails, try using eval() as fallback for Python dict format
                        # This handles single quotes in the data
                        product = eval(line)
                    
                    if 'asin' in product:
                        text = self.extract_item_text(product)
                        asin_to_text[product['asin']] = text
                        item_count += 1
                        
                        if item_count % 1000 == 0:
                            print(f"Processed {item_count} products...")
                            
                except (json.JSONDecodeError, SyntaxError, ValueError) as e:
                    print(f"Warning: Could not parse line {line_num + 1}: {str(e)}")
                    if line_num < 10:  # Only show first 10 problematic lines to avoid spam
                        print(f"Line content (first 100 chars): {line[:100]}...")
                    continue
                except Exception as e:
                    print(f"Warning: Error processing line {line_num + 1}: {str(e)}")
                    continue
        
        print(f"Loaded text for {len(asin_to_text)} products")
        
        # Map to item IDs
        item_id_to_text = {}
        asin_to_id = datamaps.get('item2id', {})
        
        for asin, text in asin_to_text.items():
            if asin in asin_to_id:
                item_id = asin_to_id[asin]
                item_id_to_text[item_id] = text
        
        print(f"Mapped {len(item_id_to_text)} products to item IDs")
        
        return asin_to_text, item_id_to_text
    
    def generate_embeddings(self, item_texts: Dict[int, str]) -> Dict[int, np.ndarray]:
        """
        Generate sentence embeddings for all items
        
        Args:
            item_texts: Dictionary mapping item IDs to text descriptions
            
        Returns:
            Dictionary mapping item IDs to embeddings
        """
        print("Generating sentence embeddings...")
        
        item_ids = list(item_texts.keys())
        texts = list(item_texts.values())
        
        # Generate embeddings in batches
        batch_size = 32
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = self.encoder.encode(batch_texts, convert_to_numpy=True)
            embeddings.extend(batch_embeddings)
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"Generated embeddings for {min(i + batch_size, len(texts))} / {len(texts)} items")
        
        # Create item ID to embedding mapping
        item_embeddings = {item_id: emb for item_id, emb in zip(item_ids, embeddings)}
        
        print(f"Generated embeddings with shape {embeddings[0].shape} for {len(item_embeddings)} items")
        
        return item_embeddings
    
    def train_residual_quantization(self, embeddings: Dict[int, np.ndarray]) -> None:
        """
        Train KMeans residual quantization codebooks
        
        Args:
            embeddings: Dictionary mapping item IDs to embeddings
        """
        print(f"Training residual quantization with {self.num_levels} levels...")
        
        # Convert to numpy array
        item_ids = list(embeddings.keys())
        X = np.array([embeddings[item_id] for item_id in item_ids])
        
        print(f"Training on {X.shape[0]} embeddings of dimension {X.shape[1]}")
        
        # Train residual quantization
        residuals = X.copy()
        
        for level in range(self.num_levels):
            print(f"Training level {level + 1}/{self.num_levels}...")
            
            # Train KMeans on residuals
            kmeans = KMeans(n_clusters=self.codebook_size, random_state=42, n_init=10)
            kmeans.fit(residuals)
            
            # Store codebook
            self.codebooks.append(kmeans)
            
            # Update residuals by subtracting quantized vectors
            quantized = kmeans.cluster_centers_[kmeans.labels_]
            residuals = residuals - quantized
            
            print(f"Level {level + 1} quantization error: {np.mean(np.linalg.norm(residuals, axis=1)):.4f}")
        
        print("Residual quantization training completed!")
    
    def quantize_embedding(self, embedding: np.ndarray) -> List[int]:
        """
        Quantize a single embedding using trained codebooks
        
        Args:
            embedding: Input embedding vector
            
        Returns:
            List of quantization codes
        """
        codes = []
        residual = embedding.copy()
        
        for level, codebook in enumerate(self.codebooks):
            # Find closest cluster
            distances = np.linalg.norm(codebook.cluster_centers_ - residual, axis=1)
            code = np.argmin(distances)
            codes.append(int(code))
            
            # Update residual
            residual = residual - codebook.cluster_centers_[code]
        
        return codes
    
    def generate_semantic_ids(self, embeddings: Dict[int, np.ndarray]) -> Dict[int, Tuple[int, ...]]:
        """
        Generate semantic IDs for all items
        
        Args:
            embeddings: Dictionary mapping item IDs to embeddings
            
        Returns:
            Dictionary mapping item IDs to semantic ID tuples (before collision handling)
        """
        print("Generating semantic IDs...")
        
        semantic_ids = {}
        
        for item_id, embedding in embeddings.items():
            codes = self.quantize_embedding(embedding)
            semantic_id = tuple(codes)
            semantic_ids[item_id] = semantic_id
            
            # Track for collision detection
            self.semantic_id_to_items[semantic_id].append(item_id)
        
        print(f"Generated semantic IDs for {len(semantic_ids)} items")
        return semantic_ids
    
    def resolve_collisions(self, semantic_ids: Dict[int, Tuple[int, ...]]) -> Dict[int, Tuple[int, ...]]:
        """
        Resolve collisions by appending suffix tokens
        
        Args:
            semantic_ids: Original semantic IDs before collision resolution
            
        Returns:
            Updated semantic IDs with collision resolution (4-level)
        """
        print("Resolving semantic ID collisions...")
        
        collisions_found = 0
        resolved_ids = {}
        
        for semantic_id, items in self.semantic_id_to_items.items():
            if len(items) > 1:
                # Collision detected
                collisions_found += 1
                
                # Assign suffix tokens to differentiate
                for idx, item_id in enumerate(items):
                    new_semantic_id = semantic_id + (idx,)  # Append suffix
                    resolved_ids[item_id] = new_semantic_id
                    
            else:
                # No collision, append 0 as suffix for consistency
                item_id = items[0]
                resolved_ids[item_id] = semantic_id + (0,)
        
        print(f"Found and resolved {collisions_found} collisions")
        print(f"Semantic IDs are now {self.num_levels + 1}-level (with collision suffix)")
        
        # Update final mapping
        self.item_to_semantic_id = resolved_ids
        self.collision_resolved = True
        
        return resolved_ids
    
    def save_semantic_ids(self, semantic_ids: Dict[int, Tuple[int, ...]], output_path: str) -> None:
        """
        Save semantic IDs to file
        
        Args:
            semantic_ids: Dictionary mapping item IDs to semantic ID tuples
            output_path: Path to save the semantic IDs
        """
        print(f"Saving semantic IDs to {output_path}...")
        
        # Convert tuples to lists for JSON serialization
        serializable_ids = {str(item_id): list(semantic_id) 
                          for item_id, semantic_id in semantic_ids.items()}
        
        with open(output_path, 'w') as f:
            json.dump({
                'semantic_ids': serializable_ids,
                'num_levels': self.num_levels + 1,  # +1 for collision suffix
                'codebook_size': self.codebook_size,
                'total_items': len(semantic_ids),
                'embedding_model': self.embedding_model_name
            }, f, indent=2)
        
        print(f"Saved semantic IDs for {len(semantic_ids)} items")
    
    def save_codebooks(self, codebooks_path: str) -> None:
        """
        Save trained codebooks for future use
        
        Args:
            codebooks_path: Path to save the codebooks
        """
        print(f"Saving codebooks to {codebooks_path}...")
        
        with open(codebooks_path, 'wb') as f:
            pickle.dump({
                'codebooks': self.codebooks,
                'num_levels': self.num_levels,
                'codebook_size': self.codebook_size,
                'embedding_model': self.embedding_model_name
            }, f)
        
        print("Codebooks saved successfully!")
    
    def load_codebooks(self, codebooks_path: str) -> None:
        """
        Load pre-trained codebooks
        
        Args:
            codebooks_path: Path to load the codebooks from
        """
        print(f"Loading codebooks from {codebooks_path}...")
        
        with open(codebooks_path, 'rb') as f:
            data = pickle.load(f)
        
        self.codebooks = data['codebooks']
        self.num_levels = data['num_levels']
        self.codebook_size = data['codebook_size']
        
        print(f"Loaded {len(self.codebooks)} codebooks")
    
    def fit_and_generate(self, metadata_path: str, datamaps_path: str, 
                        output_path: str, codebooks_path: str) -> Dict[int, Tuple[int, ...]]:
        """
        Complete pipeline: load data, train quantization, generate semantic IDs
        
        Args:
            metadata_path: Path to product metadata file
            datamaps_path: Path to datamaps file  
            output_path: Path to save semantic IDs
            codebooks_path: Path to save codebooks
            
        Returns:
            Dictionary mapping item IDs to final semantic IDs
        """
        print("=" * 60)
        print("SEMANTIC ID GENERATION PIPELINE")
        print("=" * 60)
        
        # Step 1: Load and process product data
        asin_to_text, item_to_text = self.load_product_data(metadata_path, datamaps_path)
        
        # Step 2: Generate embeddings
        embeddings_dict = self.generate_embeddings(item_to_text)
        
        # Step 3: Train residual quantization
        self.train_residual_quantization(embeddings_dict)
        
        # Step 4: Generate semantic IDs
        semantic_ids = self.generate_semantic_ids(embeddings_dict)
        
        # Step 5: Resolve collisions
        final_semantic_ids = self.resolve_collisions(semantic_ids)
        
        # Step 6: Save results
        self.save_semantic_ids(final_semantic_ids, output_path)
        self.save_codebooks(codebooks_path)
        
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Generated semantic IDs for {len(final_semantic_ids)} items")
        print(f"Semantic ID format: {self.num_levels + 1}-level tuples")
        print(f"Codebook size per level: {self.codebook_size}")
        
        return final_semantic_ids


def main():
    """Example usage of the Semantic ID Generator"""
    
    # Configuration
    NUM_LEVELS = 3
    CODEBOOK_SIZE = 256
    
    # File paths
    METADATA_PATH = 'beauty/meta.json.gz'
    DATAMAPS_PATH = 'beauty/datamaps.json'
    OUTPUT_PATH = 'beauty/semantic_ids.json'
    CODEBOOKS_PATH = 'beauty/semantic_codebooks.pkl'
    
    # Initialize generator
    generator = SemanticIDGenerator(
        num_levels=NUM_LEVELS,
        codebook_size=CODEBOOK_SIZE,
        embedding_model='all-MiniLM-L6-v2'  # Lightweight model
    )
    
    try:
        # Run complete pipeline
        semantic_ids = generator.fit_and_generate(
            metadata_path=METADATA_PATH,
            datamaps_path=DATAMAPS_PATH,
            output_path=OUTPUT_PATH,
            codebooks_path=CODEBOOKS_PATH
        )
        
        # Show some examples
        print("\nExample Semantic IDs:")
        for i, (item_id, semantic_id) in enumerate(list(semantic_ids.items())[:5]):
            print(f"  Item {item_id}: {semantic_id}")
        
        print(f"\nSemantic IDs saved to: {OUTPUT_PATH}")
        print(f"Codebooks saved to: {CODEBOOKS_PATH}")
        
    except Exception as e:
        print(f"Error in pipeline: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()