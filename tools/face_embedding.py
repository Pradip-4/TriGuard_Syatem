import os
import pickle
import numpy as np
import importlib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preload_face_runtime() -> None:
    """
    Dynamically patches the TensorFlow/Keras environment to mock missing layers 
    and legacy properties (like KerasHistory.layer) required by DeepFace.
    """
    # 1. Patch missing LocallyConnected2D layer
    try:
        layers_mod = importlib.import_module("tensorflow.keras.layers")
        if not hasattr(layers_mod, "LocallyConnected2D"):
            base_layer = getattr(layers_mod, "Layer")

            class LocallyConnected2D(base_layer):
                def __init__(self, *args, **kwargs):
                    super().__init__(**kwargs)

                def call(self, inputs):
                    return inputs

            setattr(layers_mod, "LocallyConnected2D", LocallyConnected2D)

        input_layer_cls = getattr(layers_mod, "InputLayer", None)
        if input_layer_cls is not None and not hasattr(input_layer_cls, "is_placeholder"):
            setattr(input_layer_cls, "is_placeholder", property(lambda self: False))
            
    except Exception as e:
        logger.warning(f"Could not apply structural Keras layers patch: {e}")

    # 2. Patch KerasHistory and Node operations for Keras 3 compatibility
    try:
        history_mod = importlib.import_module("keras.src.ops.node")
        
        keras_history = getattr(history_mod, "KerasHistory", None)
        if keras_history is not None and not hasattr(keras_history, "layer"):
            setattr(keras_history, "layer", property(lambda self: self.operation))

        node_cls = getattr(history_mod, "Node", None)
        if node_cls is not None and not hasattr(node_cls, "keras_inputs"):
            setattr(
                node_cls,
                "keras_inputs",
                property(lambda self: getattr(self, "input_tensors", None)),
            )
        if node_cls is not None and not hasattr(node_cls, "layer"):
            setattr(node_cls, "layer", property(lambda self: getattr(self, "operation", None)))
            
        logger.info("Successfully applied KerasHistory legacy API patches.")
    except Exception:
        pass


# CRITICAL: Run the patch BEFORE importing DeepFace
preload_face_runtime()
from deepface import DeepFace


def build_database(dataset_path: str = "image_data", output_path: str = "embeddings.pkl"):
    """
    Scans the dataset folder, generates ArcFace embeddings, 
    and saves ALL individual embeddings per identity to a pickle file for KNN matching.
    """
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset folder '{dataset_path}' not found.")
        return

    db = {}
    
    # Iterate through each person's folder
    for person_name in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person_name)
        
        if not os.path.isdir(person_dir):
            continue
            
        logger.info(f"Processing identity: {person_name}")
        person_embeddings = []
        
        # Process each image in the person's folder
        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)
            
            try:
                # Extract embedding using ArcFace
                result = DeepFace.represent(
                    img_path=image_path, 
                    model_name="ArcFace", 
                    enforce_detection=True
                )
                
                embedding = result[0]["embedding"]
                person_embeddings.append(embedding)
                logger.info(f"  -> Successfully processed {image_name}")
                
            except Exception as e:
                logger.warning(f"  -> Failed to process {image_name}: {e}")
                
        # Preserve every individual embedding array in a list for K-NN matching
        if person_embeddings:
            db[person_name] = [np.asarray(emb, dtype=np.float32) for emb in person_embeddings]
            logger.info(f"  -> Saved all {len(person_embeddings)} raw embeddings for {person_name}.")
        else:
            logger.warning(f"No valid faces found for {person_name}. Skipping.")

    if not db:
        logger.error("No embeddings were generated. Database is empty.")
        return

    # Save the database
    with open(output_path, "wb") as f:
        pickle.dump(db, f)
        
    total_embeddings = sum(len(embs) for embs in db.values())
    logger.info(
        f"Database successfully saved to '{output_path}' with {total_embeddings} total "
        f"embeddings across {len(db)} individual identities."
    )

if __name__ == "__main__":
    build_database()