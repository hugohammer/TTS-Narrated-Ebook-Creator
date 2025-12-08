#!/usr/bin/env python3
"""
epub_processor_xtts.py

The "Perfect Sync" Pipeline for EPUB 3 Audiobooks using XTTS-v2.
- Generates compliant XHTML (with fixed titles & namespaces).
- Generates compliant SMIL (with fixed relative paths & gapless timing).
- Generates Audio (using robust TTS handling).
"""

import os
import sys
import shutil
import zipfile
import subprocess 
import unicodedata
import re
import json
import torch
import numpy as np
import soundfile as sf
import argparse
import nltk
from tqdm import tqdm
from bs4 import BeautifulSoup
from TTS.api import TTS

# === TUNING CONSTANTS ===
CHUNK_CHAR_LIMIT = 200
# ========================

def check_cuda():
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / (1024 ** 3)
        # print(f"Found CUDA device with {vram_gb:.2f} GB VRAM.")
        return "cuda"
    return "cpu"

def apply_pytorch_safe_globals_patch():
    """Fixes PyTorch 2.6+ security restriction for XTTS."""
    import torch
    if hasattr(torch.serialization, 'add_safe_globals'):
        try:
            from TTS.tts.configs.xtts_config import XttsConfig
            from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
            from TTS.config.shared_configs import BaseDatasetConfig
            torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, XttsArgs, BaseDatasetConfig])
            # print("✅ Applied PyTorch 2.6+ safe globals patch.")
        except Exception as e:
            print(f"⚠️ Warning: Could not apply PyTorch patch: {e}")

# -------------------------------------------------------------------------
# 1. EPUB INFRASTRUCTURE
# -------------------------------------------------------------------------
def unzip_epub(epub_path, extract_to):
    if os.path.exists(extract_to) and os.path.exists(os.path.join(extract_to, "META-INF")):
        print(f"Work directory exists. Resuming from: {extract_to}")
        return

    if os.path.exists(extract_to):
        shutil.rmtree(extract_to)
    with zipfile.ZipFile(epub_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Unzipped to: {extract_to}")

def find_opf(work_dir):
    container = os.path.join(work_dir, "META-INF", "container.xml")
    with open(container, 'r') as f:
        soup = BeautifulSoup(f, 'xml')
    rootfile = soup.find('rootfile')
    full_path = os.path.join(work_dir, rootfile['full-path'])
    return full_path, os.path.dirname(full_path)

def detect_styles_dir(manifest_soup, oebps_dir):
    css_item = manifest_soup.find('item', attrs={'media-type': 'text/css'})
    if css_item:
        href = css_item['href']
        style_folder = os.path.dirname(href)
        return os.path.join(oebps_dir, style_folder), style_folder
    else:
        return os.path.join(oebps_dir, "Styles"), "Styles"

def write_nuclear_css(styles_dir, css_filename):
    css_path = os.path.join(styles_dir, css_filename)
    nuclear_css = """
    /* Force High Priority Highlighting */
    span.-epub-media-overlay-active, .-epub-media-overlay-active,
    span.epub-media-overlay-active, .epub-media-overlay-active {
        background-color: #ffff00 !important;
        color: black !important;
        border-radius: 2px;
    }
    @media (prefers-color-scheme: dark) {
        span.-epub-media-overlay-active, .-epub-media-overlay-active,
        span.epub-media-overlay-active, .epub-media-overlay-active {
            background-color: #7b1fa2 !important;
            color: white !important;
        }
    }
    """
    with open(css_path, "w") as f:
        f.write(nuclear_css)

# -------------------------------------------------------------------------
# 2. TEXT PROCESSING
# -------------------------------------------------------------------------
def clean_text_for_tts(text):
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'\.\s\.\s\.', '...', text) 
    text = text.replace('—', ', ').replace('–', '-')
    
    # Smart quote handling
    text = re.sub(r'(?<=[a-zA-Z])[\u2018\u2019\u0027](?=[a-zA-Z])', '___APO___', text)
    text = re.sub(r'["“”‘’\']', '', text)
    text = text.replace('___APO___', "'")
    
    text = re.sub(r'[!]{2,}', '!', text)
    text = re.sub(r'[?]{2,}', '?', text)
    allowed = re.compile(r"[^a-zA-Z0-9\s.,?!;:'-]")
    text = allowed.sub("", text)
    return re.sub(r'\s+', ' ', text).strip()

def process_xhtml_inplace(filepath, global_id_start, css_rel_path):
    with open(filepath, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'xml')

    for tag in soup.find_all("a"):
        classes = tag.get("class", [])
        role = tag.get("role", "")
        epub_type = tag.get("epub:type", "")
        if "noteref" in classes or role == "doc-noteref" or "noteref" in epub_type:
            tag.decompose()

    head = soup.find('head')
    if head:
        css_name = os.path.basename(css_rel_path)
        exists = False
        for link in head.find_all('link'):
            if css_name in link.get('href', ''):
                exists = True
                break
        if not exists:
            new_link = soup.new_tag("link", rel="stylesheet", href=css_rel_path, type="text/css")
            head.append(new_link)
            
    if soup.title and not soup.title.string:
        soup.title.string = ""

    segments = []
    current_id = global_id_start
    block_tags = ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'blockquote', 'div']
    
    for tag in soup.find_all(block_tags):
        if tag.find(block_tags): continue
            
        original_text = tag.get_text()
        clean = clean_text_for_tts(original_text)
        if not clean: continue

        tag.clear()
        sentences = nltk.sent_tokenize(clean)
        for sent in sentences:
            seg_id = f"f{current_id:06d}"
            current_id += 1
            span = soup.new_tag("span", id=seg_id)
            span.string = sent + " "
            tag.append(span)
            segments.append({"id": seg_id, "text": sent})

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(str(soup.prettify()))
    return segments, current_id

# -------------------------------------------------------------------------
# 3. AUDIO & SMIL GENERATION
# -------------------------------------------------------------------------
def get_audio_duration(filepath):
    try:
        cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", filepath]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return float(result.stdout.strip())
    except: return 0.0

def safe_tts(model, text, speaker, lang):
    chunk = text.replace("\n", " ").strip()
    if not chunk: return []
    if len(chunk) > 200:
        mid = len(chunk)//2
        split = chunk.rfind(" ", 0, mid)
        if split == -1: split = mid
        return safe_tts(model, chunk[:split], speaker, lang) + safe_tts(model, chunk[split:], speaker, lang)
    try:
        # XTTS API
        wav = model.tts(text=chunk, speaker_wav=speaker, language=lang)
        return [np.array(wav)]
    except Exception as e:
        print(f"\n[TTS ERROR] Failed on text: '{chunk[:20]}...'\nReason: {e}")
        return [np.zeros(24000)]

def format_time(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{int(h)}:{int(m):02d}:{s:06.3f}"

def create_smil_content(basename, segments, audio_filename, rel_text_path):
    """Generates compliant SMIL 3.0 XML with gapless timing."""
    xml = '<?xml version="1.0" encoding="utf-8"?>\n'
    xml += '<smil xmlns="http://www.w3.org/ns/SMIL" xmlns:epub="http://www.idpf.org/2007/ops" version="3.0">\n'
    xml += '  <body>\n'
    xml += f'    <seq id="{basename}_seq" epub:textref="{rel_text_path}">\n'

    audio_ref = f"../Audio/{audio_filename}" 

    for i, seg in enumerate(segments):
        clip_start = seg['start']
        if i < len(segments) - 1:
            clip_end = segments[i+1]['start'] # Gapless
        else:
            clip_end = seg['end']

        t_start_str = format_time(clip_start)
        t_end_str = format_time(clip_end)
        t_id = seg['id']
        
        xml += f'      <par id="par_{t_id}">\n'
        xml += f'        <text src="{rel_text_path}#{t_id}"/>\n'
        xml += f'        <audio src="{audio_ref}" clipBegin="{t_start_str}" clipEnd="{t_end_str}"/>\n'
        xml += '      </par>\n'

    xml += '    </seq>\n'
    xml += '  </body>\n'
    xml += '</smil>'
    return xml

# --- FIX: Added generate_audio_flag argument ---
def generate_media(basename, segments, model, speaker, lang, audio_dir, smil_dir, rel_text_path, generate_audio_flag):
    mp3_path = os.path.join(audio_dir, f"{basename}.mp3")
    smil_path = os.path.join(smil_dir, f"{basename}.smil")
    
    if os.path.exists(mp3_path) and os.path.exists(smil_path):
        print(f"  [SKIP] Audio exists: {basename}.mp3")
        return get_audio_duration(mp3_path)

    full_audio = []
    sync_data = []
    curr_time = 0.0
    sr = 24000
    
    if generate_audio_flag:
        for seg in tqdm(segments, desc="TTS", leave=False):
            wavs = safe_tts(model, seg['text'], speaker, lang)
            wav_data = np.concatenate(wavs) if wavs else np.zeros(int(0.5*sr))
            
            dur = len(wav_data) / sr
            start = curr_time
            end = curr_time + dur
            
            sync_data.append({"id": seg['id'], "start": start, "end": end})
            full_audio.append(wav_data)
            full_audio.append(np.zeros(int(0.15 * sr)))
            curr_time = end + 0.15
            
        if full_audio:
            wav_path = os.path.join(audio_dir, f"{basename}.wav")
            final_wav = np.concatenate(full_audio)
            sf.write(wav_path, final_wav, sr)
            
            subprocess.run(["ffmpeg", "-i", wav_path, "-b:a", "128k", "-y", mp3_path, "-loglevel", "error"], check=True)
            os.remove(wav_path)
            total_duration = curr_time
        else:
            total_duration = 0
    else:
        total_duration = 10.0
        # Mock sync data for SMIL generation if audio is skipped
        for i, seg in enumerate(segments):
            sync_data.append({"id": seg['id'], "start": i*5.0, "end": (i+1)*5.0})

    # Save SMIL
    smil_content = create_smil_content(basename, sync_data, f"{basename}.mp3", rel_text_path)
    with open(smil_path, "w", encoding="utf-8") as f:
        f.write(smil_content)

    return total_duration

# -------------------------------------------------------------------------
# 4. OPF UPDATER
# -------------------------------------------------------------------------
def update_opf(opf_path, manifest_additions, spine_mapping, total_durations):
    with open(opf_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'xml')

    for meta in soup.metadata.find_all("meta", property="media:duration"): meta.decompose()
    for meta in soup.metadata.find_all("meta", property="media:active-class"): meta.decompose()

    meta_dur = soup.new_tag("meta", property="media:duration")
    total_sec = sum(total_durations.values())
    m, s = divmod(total_sec, 60)
    h, m = divmod(m, 60)
    meta_dur.string = f"{int(h)}:{int(m):02d}:{s:06.3f}"
    soup.metadata.append(meta_dur)
    
    meta_active = soup.new_tag("meta", property="media:active-class")
    meta_active.string = "-epub-media-overlay-active"
    soup.metadata.append(meta_active)

    for smil_id, dur_sec in total_durations.items():
        meta = soup.new_tag("meta", property="media:duration")
        meta['refines'] = f"#{smil_id}"
        meta.string = format_time(dur_sec)
        soup.metadata.append(meta)

    manifest = soup.manifest
    for item in manifest_additions:
        existing = manifest.find("item", id=item['id'])
        if existing: existing.decompose()
        
    for item in manifest_additions:
        new_tag = soup.new_tag("item", **item)
        manifest.append(new_tag)

    for text_id, smil_id in spine_mapping.items():
        item = manifest.find("item", id=text_id)
        if item:
            item['media-overlay'] = smil_id

    xml_str = str(soup.prettify())
    xml_str = re.sub(r'(<meta property="media:active-class"[^>]*>)\s+([^\s<]+)\s+(</meta>)', r'\1\2\3', xml_str)
    xml_str = re.sub(r'(<opf:meta property="media:active-class"[^>]*>)\s+([^\s<]+)\s+(</opf:meta>)', r'\1\2\3', xml_str)

    with open(opf_path, 'w', encoding='utf-8') as f:
        f.write(xml_str)

# -------------------------------------------------------------------------
# 5. PACKAGING
# -------------------------------------------------------------------------
def zip_dir(folder, output_path):
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(os.path.join(folder, "mimetype"), "mimetype", compress_type=zipfile.ZIP_STORED)
        for root, dirs, files in os.walk(folder):
            for file in files:
                if file == "mimetype": continue
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, folder)
                zipf.write(abs_path, rel_path)

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generative Audiobook Creator (XTTS-v2): Convert EPUB to EPUB 3 with Media Overlays."
    )
    
    parser.add_argument("input_epub", help="Path to the input .epub file")
    parser.add_argument("--voice", "-v", required=True, help="Path to the reference voice sample (.wav)")
    parser.add_argument("--output", "-o", help="Path for the output .epub file")
    parser.add_argument("--language", "-l", default="en", help="Language code (default: en)")
    parser.add_argument("--skip-audio", action="store_true", help="Skip TTS generation (for debugging layout)")
    parser.add_argument("--gpu", action="store_true", help="Force usage of GPU if available")

    return parser.parse_args()

# -------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------
def main():
    args = parse_arguments()
    
    # 1. Setup Inputs
    INPUT_EPUB = args.input_epub
    SPEAKER_WAV = args.voice
    BOOK_LANGUAGE = args.language
    GENERATE_AUDIO = not args.skip_audio

    # 2. Setup Output Path (ROBUST LOGIC)
    # Get the input filename without extension (e.g., "mybook")
    input_basename = os.path.splitext(os.path.basename(INPUT_EPUB))[0]
    default_filename = f"{input_basename}_Audio_Overlay.epub"

    if args.output:
        # Check if the argument provided is an existing directory or looks like one
        if os.path.isdir(args.output) or args.output.endswith(os.sep):
            # It is a folder -> Place the file inside it with the default name
            # Ensure directory exists
            os.makedirs(args.output, exist_ok=True)
            OUTPUT_EPUB = os.path.join(args.output, default_filename)
        else:
            # It is a specific file path -> Use it exactly as provided
            OUTPUT_EPUB = args.output
            # Ensure parent directory exists
            parent_dir = os.path.dirname(os.path.abspath(OUTPUT_EPUB))
            if parent_dir: os.makedirs(parent_dir, exist_ok=True)
    else:
        # Default: Place next to input file
        input_dir = os.path.dirname(os.path.abspath(INPUT_EPUB))
        OUTPUT_EPUB = os.path.join(input_dir, default_filename)

    # 3. Setup Work Directory
    # We place it next to the INPUT file, hidden, unique to the book name
    input_dir = os.path.dirname(os.path.abspath(INPUT_EPUB))
    WORK_DIR = os.path.join(input_dir, f".{input_basename}_workdir")

    print(f"=== STARTING PIPELINE ===")
    print(f"Input:    {INPUT_EPUB}")
    print(f"Work Dir: {WORK_DIR}")
    print(f"Output:   {OUTPUT_EPUB}")
    print(f"Voice:    {SPEAKER_WAV}")

    # Validate Inputs
    if not os.path.exists(INPUT_EPUB):
        print(f"Error: Input file not found: {INPUT_EPUB}")
        sys.exit(1)
    if not os.path.exists(SPEAKER_WAV):
        print(f"Error: Voice sample not found: {SPEAKER_WAV}")
        sys.exit(1)

    device = "cuda" if args.gpu and torch.cuda.is_available() else check_cuda()
    apply_pytorch_safe_globals_patch()
    try: nltk.data.find("tokenizers/punkt")
    except: nltk.download("punkt")

    unzip_epub(INPUT_EPUB, WORK_DIR)
    opf_path, oebps_dir = find_opf(WORK_DIR)
    
    with open(opf_path, 'r') as f: soup = BeautifulSoup(f, 'xml')
    styles_dir, styles_rel_prefix = detect_styles_dir(soup.manifest, oebps_dir)
    
    audio_dir = os.path.join(oebps_dir, "Audio")
    smil_dir = os.path.join(oebps_dir, "Smil")
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(smil_dir, exist_ok=True)
    os.makedirs(styles_dir, exist_ok=True)

    css_name = "sync_highlight.css"
    write_nuclear_css(styles_dir, css_name)

    print("\n--- Processing Content ---")
    manifest = soup.manifest
    spine = soup.spine
    
    if GENERATE_AUDIO:
        print("Loading XTTS...")
        try:
            tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Critical Model Error: {e}")
            sys.exit(1)
    else: 
        tts = None
        print("[INFO] Skipping Audio Generation (--skip-audio used)")

    manifest_additions = [] 
    manifest_additions.append({"id": "sync_css", "href": f"{styles_rel_prefix}/{css_name}", "media-type": "text/css"})
    
    spine_mapping = {} 
    total_durations = {} 
    global_id = 1
    processed_files = set()

    for itemref in spine.find_all("itemref"):
        item_id = itemref['idref']
        item = manifest.find("item", id=item_id)
        if not item or item['media-type'] != "application/xhtml+xml":
            continue
            
        filename = item['href'] 
        full_path = os.path.join(oebps_dir, filename)
        basename = os.path.splitext(os.path.basename(filename))[0]
        
        if full_path in processed_files or item.get('properties') == 'nav':
            continue
        processed_files.add(full_path)

        rel_text_path = f"../{filename}"
        rel_css = os.path.relpath(os.path.join(styles_dir, css_name), os.path.dirname(full_path))

        print(f"Processing: {basename}")
        
        segments, global_id = process_xhtml_inplace(full_path, global_id, rel_css)
        
        if not segments: continue 
        
        # --- FIX: Pass GENERATE_AUDIO to function ---
        duration = generate_media(basename, segments, tts, SPEAKER_WAV, BOOK_LANGUAGE, audio_dir, smil_dir, rel_text_path, GENERATE_AUDIO)
        
        smil_id = f"smil_{basename}"
        audio_id = f"audio_{basename}"
        
        manifest_additions.append({"id": smil_id, "href": f"Smil/{basename}.smil", "media-type": "application/smil+xml"})
        manifest_additions.append({"id": audio_id, "href": f"Audio/{basename}.mp3", "media-type": "audio/mpeg"})
        
        spine_mapping[item_id] = smil_id
        total_durations[smil_id] = duration

    print("\n--- Updating Metadata ---")
    update_opf(opf_path, manifest_additions, spine_mapping, total_durations)

    print("\n--- Packaging EPUB ---")
    zip_dir(WORK_DIR, OUTPUT_EPUB)
    
    print(f"\nSUCCESS! Saved to: {OUTPUT_EPUB}")

if __name__ == "__main__":
    main()
