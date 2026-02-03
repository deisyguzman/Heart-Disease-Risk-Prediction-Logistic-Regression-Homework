
"""
SageMaker Inference Script for Heart Disease Prediction
"""

import json
import numpy as np
import os

def model_fn(model_dir):
    """
    Load the trained model from the model directory.

    Args:
        model_dir: Directory where model artifacts are stored

    Returns:
        Dictionary containing model weights, bias, and metadata
    """
    weights = np.load(os.path.join(model_dir, 'weights.npy'))
    bias = np.load(os.path.join(model_dir, 'bias.npy'))[0]

    with open(os.path.join(model_dir, 'metadata.json'), 'r') as f:
        metadata = json.load(f)

    return {
        'weights': weights,
        'bias': bias,
        'metadata': metadata
    }


def input_fn(request_body, content_type='application/json'):
    """
    Deserialize and prepare the input data.

    Expected JSON format:
    {
        "Age": 54,
        "Cholesterol": 239,
        "BP": 135,
        "Max HR": 160,
        "ST depression": 1.2,
        "Number of vessels fluro": 1
    }

    Args:
        request_body: The input data (JSON string)
        content_type: Content type of the request

    Returns:
        Numpy array of features
    """
    if content_type == 'application/json':
        input_data = json.loads(request_body)

        # Extract features in correct order
        features = [
            input_data.get('Age', 0),
            input_data.get('Cholesterol', 0),
            input_data.get('BP', 0),
            input_data.get('Max HR', 0),
            input_data.get('ST depression', 0),
            input_data.get('Number of vessels fluro', 0)
        ]

        return np.array(features, dtype=np.float64)
    else:
        raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_data, model):
    """
    Make prediction on preprocessed input.

    Args:
        input_data: Preprocessed input features (numpy array)
        model: Loaded model dictionary

    Returns:
        Dictionary with prediction results
    """
    w = model['weights']
    b = model['bias']
    metadata = model['metadata']

    # Standardize features using training statistics
    means = np.array(metadata['feature_means'])
    stds = np.array(metadata['feature_stds'])
    X_scaled = (input_data - means) / stds

    # Compute prediction
    z = np.dot(X_scaled, w) + b
    prob = 1 / (1 + np.exp(-z))  # Sigmoid
    prediction = 1 if prob >= 0.5 else 0

    return {
        'prediction': int(prediction),
        'probability': float(prob),
        'risk_level': 'HIGH RISK' if prediction == 1 else 'LOW RISK'
    }


def output_fn(prediction, accept='application/json'):
    """
    Serialize the prediction output.

    Args:
        prediction: Prediction results
        accept: Requested response content type

    Returns:
        JSON string with prediction results
    """
    if accept == 'application/json':
        return json.dumps(prediction), accept
    else:
        raise ValueError(f"Unsupported accept type: {accept}")
