import whisperx
from datasets import load_dataset

device = "cuda"
batch_size = 16
compute_type = "float32"


sova_dataset = load_dataset("bond005/sova_rudevices", split="test")
sample = sova_dataset[0]["audio"]["array"]
sample = sample.astype("float32")


model = whisperx.load_model("large-v3", device, compute_type=compute_type)
result = model.transcribe(sample, batch_size=batch_size)
print(result["segments"])


model_a, metadata = whisperx.load_align_model(
    language_code=result["language"], device=device
)
result = whisperx.align(
    result["segments"], model_a, metadata, sample, device, return_char_alignments=False
)

print(result["segments"])
