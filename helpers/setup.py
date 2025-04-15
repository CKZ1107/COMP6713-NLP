import ssl
import torch
import nltk
import warnings
import numpy as np
from transformers import BertTokenizer


# ------------------ NLTK setup ------------------ #

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


def download_nltk_resources():
    """
    Download NTLK resources.
    """
    resources = ['punkt', 'wordnet', 'averaged_perceptron_tagger', 'sentiwordnet', 'omw-1.4']
    for resource in resources:
        try:
            print(f"Downloading NLTK resource: {resource}")
            nltk.download(resource, quiet=False)
        except Exception as e:
            print(f"Error downloading {resource}: {e}")
    print("NLTK resource download complete")


# ------------------ Full environment setup ------------------ #

def setup():
    warnings.filterwarnings('ignore')
    download_nltk_resources()

    # check for GPU availability
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("MPS (Apple Silicon) device detected!")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA device detected!")
    else:
        device = torch.device("cpu")
        print("Using CPU for training. This will be slower.")

    # set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    return device


label_list = [-1, 1]
num_labels = len(label_list)
date = 'trading_day'
label = 'label'
post = 'post'
stock = 'stock'
dataset_path = 'dataset.csv'
model_path = '.models'
default_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
device = setup()
