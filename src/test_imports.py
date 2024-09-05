import transformers
import torch
import torchaudio
import tokenizers
import datasets


if __name__ == "__main__":
    print(transformers.__version__)
    print(torch.__version__)
    print(torchaudio.__version__)
    print(tokenizers.__version__)
    print(datasets.__version__)
    print(torch.cuda.is_available())
