import os
import json
import torch
from .setup import default_tokenizer, device
from sklearn.metrics import classification_report, accuracy_score


# -------------------- Save and Load model -------------------- #


def save_model(model, model_config, save_dir, tokenizer=default_tokenizer):
    """
    Save the model, tokenizer, and associated metadata

    Args:
        model    : The model in question
        tokenizer: The BERT tokenizer used with the model
        save_dir : Directory path where the model should be saved
    """
    # create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # save model state
    torch.save(model.state_dict(), f"{save_dir}/hbert_model.pt")

    # save tokenizer
    tokenizer.save_pretrained(save_dir)
    with open(f"{save_dir}/model_config.json", 'w') as f:
        json.dump(model_config, f)

    print(f"Model successfully saved to {save_dir}")
    return save_dir


def load_model(Model, save_dir, tokenizer=default_tokenizer):
    """
    Load a saved model, tokenizer and its metadata.

    Args:
        save_dir: Directory path containing the saved model

    Returns:
        A tuple containing (model, tokenizer)
    """
    if not os.path.exists(save_dir):
        raise FileNotFoundError(f"Model directory {save_dir} not found")

    # load model configuration
    with open(f"{save_dir}/model_config.json", 'r') as f:
        model_config = json.load(f)

    # initialize model with saved configuration
    model = Model(**model_config)

    # load model weights
    model.load_state_dict(torch.load(f"{save_dir}/hbert_model.pt", map_location=device))
    model.to(device)
    model.eval()
    print(f"Model successfully loaded from {save_dir}")
    return model, tokenizer
