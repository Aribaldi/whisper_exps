import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from datasets import load_dataset

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=False, use_safetensors=False
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

sova_dataset = load_dataset("bond005/sova_rudevices", split="test")
sample = sova_dataset[0]["audio"]["array"]

inputs = processor(
    sample,
    sampling_rate=16000,
    return_tensors="pt",
    return_attention_mask=True,
)
inputs = inputs.to(device, dtype=torch_dtype)

gen_kwargs = {
    "max_new_tokens": 256,
    "num_beams": 1,
    "condition_on_prev_tokens": False,
    "compression_ratio_threshold": 1.35,
    "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    "logprob_threshold": -1.0,
    "no_speech_threshold": 0.6,
    "return_timestamps": True,
}

pred_ids = model.generate(**inputs, **gen_kwargs)
pred_text = processor.batch_decode(
    pred_ids, skip_special_tokens=True, decode_with_timestamps=False
)


print(pred_text)
