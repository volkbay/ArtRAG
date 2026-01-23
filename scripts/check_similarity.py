from sentence_transformers import SentenceTransformer, util
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
import sys
import argparse
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description='Find similar entities in a knowledge graph based on semantic similarity.')
    parser.add_argument('--threshold', type=float, default=0.95,
                        help='Similarity threshold for considering entities as synonyms (default: 0.95)')
    parser.add_argument('--graph-path', type=str, 
                        default="./built_graph/All_gpt_4o_mini_prompt_tuning_style_event_clean/graph_chunk_entity_relation.graphml",
                        help='Path to the GraphML file')
    parser.add_argument('--output', type=str, default="ackg_synonym_candidates.csv",
                        help='Output CSV file path (default: ackg_synonym_candidates.csv)')
    return parser.parse_args()

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        stream=sys.stdout
    )

def load_and_process_graph(graph_path):
    try:
        G = nx.read_graphml(graph_path)
        logging.info(f"Successfully loaded graph with {len(G.nodes)} nodes and {len(G.edges)} edges")
        
        # Extract node info
        entities = []
        for node_id, data in G.nodes(data=True):
            try:
                label = node_id.strip('"')
                description = data.get("description", "").replace("&lt;SEP&gt;", " ").strip('"')
                entity_type = data.get("entity_type", "").strip('"')
                combined_text = f"{label}. {entity_type}. {description}"
                entities.append({"label": label, "text": combined_text, "entity_type": entity_type})
            except Exception as e:
                logging.warning(f"Failed to process node {node_id}: {str(e)}")
                continue
        
        logging.info(f"Processed {len(entities)} entities")
        return entities
    except Exception as e:
        logging.error(f"Failed to load graph: {str(e)}")
        raise

def compute_similarities(entities, threshold):
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        texts = [e["text"] for e in entities]
        embeddings = model.encode(texts, convert_to_tensor=True)
        logging.info("Successfully encoded all entities")
        
        # Compute pairwise cosine similarities
        cosine_scores = util.cos_sim(embeddings, embeddings).cpu().numpy()
        
        n = len(entities)
        similar_pairs = []
        
        logging.info(f"Starting similarity computation with threshold {threshold}")
        for i in tqdm(range(n), desc="Computing similarities"):
            for j in range(i + 1, n):
                if entities[i]["entity_type"] != entities[j]["entity_type"]:
                    continue 
                sim = cosine_scores[i][j]
                if sim > threshold:
                    similar_pairs.append({
                        "entity_1": entities[i]["label"],
                        "entity_2": entities[j]["label"],
                        "similarity": round(sim, 4)
                    })
                    logging.debug(f"Found similar pair: {entities[i]['label']} - {entities[j]['label']} ({sim:.4f})")
        
        logging.info(f"Found {len(similar_pairs)} similar pairs")
        return similar_pairs
    except Exception as e:
        logging.error(f"Failed during similarity computation: {str(e)}")
        raise

def compute_statistics(entities, similar_pairs):
    # Group entities by type
    type_to_entities = defaultdict(set)
    for e in entities:
        if e["entity_type"]:
            type_to_entities[e["entity_type"]].add(e["label"])

    # Track how many entities appear in synonym pairs (per type)
    type_to_synonyms = defaultdict(set)
    for pair in similar_pairs:
        try:
            # Get entity types from the original entities list
            entity1_type = next(e["entity_type"] for e in entities if e["label"] == pair["entity_1"])
            entity2_type = next(e["entity_type"] for e in entities if e["label"] == pair["entity_2"])
            
            # Since we only compare entities of the same type, we can use either type
            typ = entity1_type
            type_to_synonyms[typ].add(pair["entity_1"])
            type_to_synonyms[typ].add(pair["entity_2"])
        except StopIteration:
            logging.warning(f"Could not find entity type for pair: {pair['entity_1']} - {pair['entity_2']}")
            continue

    # Compute and print stats
    print("\n=== Synonymy Coverage by Entity Type ===")
    for entity_type in type_to_entities:
        total = len(type_to_entities[entity_type])
        covered = len(type_to_synonyms.get(entity_type, []))
        ratio = 100.0 * covered / total if total > 0 else 0
        print(f"{entity_type}: {covered}/{total} entities (~{ratio:.2f}%) in synonym pairs")

def save_results(similar_pairs, output_path):
    df_similar = pd.DataFrame(similar_pairs)
    df_similar = df_similar.sort_values(by="similarity", ascending=False)
    
    # Show top results
    print("\nTop 10 most similar pairs:")
    print(df_similar.head(10))
    
    # Save to CSV
    try:
        df_similar.to_csv(output_path, index=False)
        logging.info(f"Successfully saved results to {output_path}")
    except Exception as e:
        logging.error(f"Failed to save CSV: {str(e)}")

def main():
    try:
        # Parse arguments
        args = parse_args()
        
        # Setup logging
        setup_logging()
        
        # Load and process graph
        entities = load_and_process_graph(args.graph_path)
        if not entities:
            logging.error("No entities found in the graph")
            return
        
        # Compute similarities
        similar_pairs = compute_similarities(entities, args.threshold)
        if not similar_pairs:
            logging.warning(f"No similar pairs found with threshold {args.threshold}")
            return
        
        # Compute and display statistics
        compute_statistics(entities, similar_pairs)
        
        # Save results
        save_results(similar_pairs, args.output)
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()

