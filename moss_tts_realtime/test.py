import torch
import torchaudio
from typing import Any
from mossttsrealtime.modeling_mossttsrealtime import MossTTSRealtime
from inferencer import MossTTSRealtimeInference
from transformers import AutoTokenizer
from transformers import AutoModel

# ========== H200 极致优化 ==========
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_bf16_fp16_reduced_precision_reduction = True

model_path = "/data/models/MOSS-TTS-Realtime"
codec_path = "/data/models/MOSS-Audio-Tokenizer"
attn_implementation = "flash_attention_3"
device = "cuda"
CODEC_SAMPLE_RATE = 24000

model = MossTTSRealtime.from_pretrained(
    model_path, attn_implementation=attn_implementation, torch_dtype=torch.bfloat16).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)
codec = AutoModel.from_pretrained(codec_path, trust_remote_code=True).eval()
codec = codec.to(device)

inferencer = MossTTSRealtimeInference(model, tokenizer, max_length=5000, codec=codec,
                                      codec_sample_rate=CODEC_SAMPLE_RATE, codec_encode_kwargs={"chunk_duration": 8})

# 加载 Realtime 模型（1.7B）
# model = MossTTSRealtime.from_pretrained(
#     "/data/models/MOSS-TTS-Realtime",
#     torch_dtype=torch.bfloat16,
#     device_map="cuda",
#     attn_implementation="flash_attention_2"  # 关键
# )
# 流式生成（首字延迟最低）
# from moss_tts.streamer import AudioIteratorStreamer
# streamer = AudioIteratorStreamer(model, chunk_ms=30)  # 30ms 音频块

# 推理
# text = "你好，这是实时语音。"
# model.generate(text, streamer=streamer, language="zh")

text = ["你好，这是实时语音。"]
reference_audio_path = ["./audio/prompt_audio.mp3"]

result = inferencer.generate(
    text=text,
    reference_audio_path=reference_audio_path,
    temperature=0.8,
    top_p=0.6,
    top_k=30,
    repetition_penalty=1.1,
    repetition_window=50,
    device=device,
)

for i, generated_tokens, in enumerate[Any](result):
    output = torch.tensor(generated_tokens).to(device)
    decode_result = codec.decode(output.permute(1, 0), chunk_duration=8)
    wav = decode_result["audio"][0].cpu().detach()

    if wav.ndim == 1:
        wav = wav.unsqueeze(0)

    torchaudio.save(f'{i}.wav', wav, CODEC_SAMPLE_RATE)

# if __name__ == "__main__":
#     model_path = "OpenMOSS-Team/MOSS-TTS-Realtime"
#     codec_path = "OpenMOSS-Team/MOSS-Audio-Tokenizer"
#     main(model_path, codec_path) 