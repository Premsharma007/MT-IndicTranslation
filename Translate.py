import os
import io
import re
import tempfile
from typing import List, Tuple

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor
import gradio as gr
from tqdm import tqdm

# -----------------------------
# Config
# -----------------------------
MODEL_NAME = "ai4bharat/indictrans2-indic-indic-1B"  # change to dist-320M for CPU
_device = "cuda" if torch.cuda.is_available() else "cpu"

# Language codes supported by Indic→Indic model
LANG_CODE_MAP = {
    "Assamese": "asm_Beng",
    "Bengali": "ben_Beng",
    "Bodo": "brx_Deva",
    "Dogri": "doi_Deva",
    "English": "eng_Latn",
    "Gujarati": "guj_Gujr",
    "Hindi": "hin_Deva",
    "Kannada": "kan_Knda",
    "Kashmiri (Devanagari)": "kas_Deva",
    "Kashmiri (Perso-Arabic)": "kas_Arab",
    "Konkani": "gom_Deva",
    "Maithili": "mai_Deva",
    "Malayalam": "mal_Mlym",
    "Manipuri (Bengali script)": "mni_Beng",
    "Manipuri (Meitei script)": "mni_Mtei",
    "Marathi": "mar_Deva",
    "Nepali": "npi_Deva",
    "Odia": "ory_Orya",
    "Punjabi": "pan_Guru",
    "Sanskrit": "san_Deva",
    "Santali (Devanagari)": "sat_Deva",
    "Santali (Ol Chiki)": "sat_Olck",
    "Sindhi (Devanagari)": "snd_Deva",
    "Sindhi (Perso-Arabic)": "snd_Arab",
    "Tamil": "tam_Taml",
    "Telugu": "tel_Telu",
    "Urdu": "urd_Arab"
}

GEN_KW = dict(
    use_cache=False,  # FIX: avoid KeyError bug
    min_length=0,
    max_length=256,
    num_beams=5,
    num_return_sequences=1,
)

# -----------------------------
# Lazy model loading
# -----------------------------
_tokenizer = None
_model = None
_iproc = None

def _load_once():
    global _tokenizer, _model, _iproc
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        _model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=torch.float16 if _device == "cuda" else torch.float32,
        ).to(_device)
        _iproc = IndicProcessor(inference=True)
        _model.config.use_cache = False

# -----------------------------
# File reading
# -----------------------------
def _read_txt(file_obj) -> str:
    """Handles Gradio v4 file object or path string."""
    if hasattr(file_obj, "name"):  
        file_path = file_obj.name
    elif isinstance(file_obj, str):
        file_path = file_obj
    else:
        raise ValueError(f"Unsupported file object type: {type(file_obj)}")

    for enc in ("utf-8", "utf-16", "utf-8-sig", "latin-1"):
        try:
            with open(file_path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

# -----------------------------
# Chunking
# -----------------------------
_SPLIT_RE = re.compile(r"(?<=[\.\?\!।])\s+|\n+")

def _split_into_chunks(text: str, max_chars: int = 600) -> List[str]:
    parts = [p.strip() for p in _SPLIT_RE.split(text) if p.strip()]
    chunks, buf, cur_len = [], [], 0
    for p in parts:
        if cur_len + len(p) + 1 > max_chars and buf:
            chunks.append(" ".join(buf))
            buf, cur_len = [p], len(p)
        else:
            buf.append(p)
            cur_len += len(p) + 1
    if buf:
        chunks.append(" ".join(buf))
    if not chunks:
        text = text.strip()
        return [text[i:i+max_chars] for i in range(0, len(text), max_chars)]
    return chunks

# -----------------------------
# Translation
# -----------------------------
def _translate_lines(lines: List[str], src_code: str, tgt_code: str) -> List[str]:
    _load_once()
    batch = _iproc.preprocess_batch(lines, src_lang=src_code, tgt_lang=tgt_code)
    inputs = _tokenizer(
        batch,
        truncation=True,
        padding="longest",
        return_tensors="pt",
        return_attention_mask=True,
    ).to(_device)
    with torch.no_grad():
        generated = _model.generate(**inputs, **GEN_KW)
    out = _tokenizer.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    out = _iproc.postprocess_batch(out, lang=tgt_code)
    return out

# -----------------------------
# Main pipeline
# -----------------------------
def pipeline(file, src_lang_label, tgt_lang_label, progress=gr.Progress()) -> Tuple[str, str]:
    if file is None:
        return "Please upload a text file.", None
    src_code = LANG_CODE_MAP[src_lang_label]
    tgt_code = LANG_CODE_MAP[tgt_lang_label]

    raw_text = _read_txt(file)
    if not raw_text.strip():
        return "Empty or unreadable file.", None

    chunks = _split_into_chunks(raw_text, max_chars=600)
    translated_chunks = []
    BATCH = 8

    for i in progress.tqdm(range(0, len(chunks), BATCH), desc="Translating"):
        batch = chunks[i:i+BATCH]
        translated_chunks.extend(_translate_lines(batch, src_code, tgt_code))

    full_translation = "\n".join(translated_chunks)

    tmpdir = tempfile.mkdtemp(prefix="it2_")
    out_path = os.path.join(tmpdir, "translation.txt")
    with io.open(out_path, "w", encoding="utf-8") as f:
        f.write(full_translation)

    preview = (full_translation[:1200] + " …") if len(full_translation) > 1200 else full_translation
    return preview, out_path

# -----------------------------
# Gradio UI
# -----------------------------
title = "Spider『X』(T2TT) MT Indic_Trans-2 1B Model - 1.0"
description = (
    "Upload a .txt file, choose source and target Indian languages, "
    "and download the translation."
)

with gr.Blocks() as demo:
    gr.Markdown(f"# {title}\n{description}")

    with gr.Row():
        file_in = gr.File(label="Upload .txt file", file_types=[".txt"])
    with gr.Row():
        src_dd = gr.Dropdown(choices=list(LANG_CODE_MAP.keys()), value="Tamil", label="Source Language")
        tgt_dd = gr.Dropdown(choices=list(LANG_CODE_MAP.keys()), value="Hindi", label="Target Language")

    translate_btn = gr.Button("Translate", variant="primary")
    output_text = gr.Textbox(label="Translated preview", lines=12)
    file_out = gr.File(label="Download translated .txt")

    translate_btn.click(fn=pipeline, inputs=[file_in, src_dd, tgt_dd], outputs=[output_text, file_out])

if __name__ == "__main__":
    demo.launch()


