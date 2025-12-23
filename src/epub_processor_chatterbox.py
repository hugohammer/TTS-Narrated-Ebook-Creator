#!/usr/bin/env python3
"""
epub_processor_chatterbox.py

The "Perfect Sync" Pipeline using Chatterbox TTS.
- Engine: Chatterbox (Faster, Flow Matching).
- Features: Smart Splitting, Ellipsis Prosody, Audio Crossfading.
- Structure: Preserves layout/CSS/Images visually while cleaning text for Audio.
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

# --- CHATTERBOX IMPORT ---
try:
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
except ImportError:
    print("Error: Chatterbox not found. Please run: pip install chatterbox-tts")
    sys.exit(1)

# --- PYDUB IMPORT (REQUIRED FOR CROSSFADE) ---
try:
    from pydub import AudioSegment
except ImportError:
    print("Please install pydub: pip install pydub")
    sys.exit(1)

# === TUNING CONSTANTS ===
CHUNK_CHAR_LIMIT = 400 
CROSSFADE_MS = 50
# ========================

def check_cuda():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

# -------------------------------------------------------------------------
# 1. HELPER FUNCTIONS (SMART LOGIC ALWAYS ON)
# -------------------------------------------------------------------------

def smart_split_text(text, max_chars=CHUNK_CHAR_LIMIT):
    """
    Splits text while preserving prosody.
    Prioritizes splitting at:
    1. Strong punctuation (.?!)
    2. Semi-strong punctuation (:;)
    3. Weak punctuation (,)
    """
    text = text.strip()
    if len(text) <= max_chars:
        return [text]

    # Define split priorities (Regex Pattern)
    split_patterns = [
        r'([.?!])\s',      # 1. End of sentence
        r'([:;])\s',       # 2. Semi-colons
        r'([,])\s',        # 3. Commas
        r'(\s)'            # 4. Any space (fallback)
    ]

    best_split_point = -1
    search_start = max_chars // 4
    search_end = max_chars

    chunk = text[:search_end]
    
    for pattern in split_patterns:
        matches = list(re.finditer(pattern, chunk))
        if matches:
            valid_matches = [m for m in matches if m.start() > search_start]
            if valid_matches:
                last_match = valid_matches[-1]
                best_split_point = last_match.end()
                break 
    
    if best_split_point == -1:
        best_split_point = max_chars

    part1 = text[:best_split_point].strip()
    part2 = text[best_split_point:].strip()
    
    return [part1] + smart_split_text(part2, max_chars)

def apply_ellipsis_to_chunks(chunks):
    """Adds '...' to cut boundaries to improve TTS prosody."""
    modified_chunks = []
    for i, chunk in enumerate(chunks):
        new_chunk = chunk
        if i < len(chunks) - 1:
            if not new_chunk.strip()[-1] in ".?!":
                 new_chunk = new_chunk + "..."
        if i > 0:
            new_chunk = "..." + new_chunk
        modified_chunks.append(new_chunk)
    return modified_chunks

def crossfade_audio_segments(wav_list, fade_ms=CROSSFADE_MS):
    """Stitches a list of numpy audio arrays using pydub crossfade."""
    if not wav_list:
        return np.array([])

    def numpy_to_audiosegment(data, sr=24000):
        # normalize to 16-bit PCM
        audio_int16 = (data * 32767).astype(np.int16)
        return AudioSegment(
            audio_int16.tobytes(), 
            frame_rate=sr,
            sample_width=2, 
            channels=1
        )

    combined = numpy_to_audiosegment(wav_list[0])

    for next_wav in wav_list[1:]:
        next_seg = numpy_to_audiosegment(next_wav)
        combined = combined.append(next_seg, crossfade=fade_ms)

    samples = np.array(combined.get_array_of_samples())
    return samples.astype(np.float32) / 32768.0

# -------------------------------------------------------------------------
# 2. EPUB INFRASTRUCTURE
# -------------------------------------------------------------------------
def unzip_epub(epub_path, extract_to):
    if os.path.exists(extract_to) and os.path.exists(os.path.join(extract_to, "META-INF")):
        return
    if os.path.exists(extract_to):
        shutil.rmtree(extract_to)
    with zipfile.ZipFile(epub_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

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
# 3. TEXT PROCESSING (DECOUPLED VISUAL/AUDIO)
# -------------------------------------------------------------------------
def clean_text_for_tts(text):
    """
    Cleans text for TTS only. Visual text remains untouched.
    """
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'\.\s\.\s\.', '...', text)
    text = text.replace('—', ', ').replace('–', '-')
    text = re.sub(r'(?<=[a-zA-Z])[\u2018\u2019\u0027](?=[a-zA-Z])', '___APO___', text)
    text = re.sub(r'["“”‘’\']', '', text)
    text = text.replace('___APO___', "'")
    text = re.sub(r'[!]{2,}', '!', text)
    text = re.sub(r'[?]{2,}', '?', text)
    allowed = re.compile(r"[^a-zA-Z0-9\s.,?!;:'-]")
    text = allowed.sub("", text)
    return re.sub(r'\s+', ' ', text).strip()


# -------------------------------------------------------------------------
# 3. TEXT PROCESSING (HTML-PRESERVING DOM TRAVERSAL)
# -------------------------------------------------------------------------
def clean_text_for_tts(text):
    """
    Cleans text for TTS only. Visual text remains untouched.
    """
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'\.\s\.\s\.', '...', text)
    text = text.replace('—', ', ').replace('–', '-')
    text = re.sub(r'(?<=[a-zA-Z])[\u2018\u2019\u0027](?=[a-zA-Z])', '___APO___', text)
    text = re.sub(r'["“”‘’\']', '', text)
    text = text.replace('___APO___', "'")
    text = re.sub(r'[!]{2,}', '!', text)
    text = re.sub(r'[?]{2,}', '?', text)
    allowed = re.compile(r"[^a-zA-Z0-9\s.,?!;:'-]")
    text = allowed.sub("", text)
    return re.sub(r'\s+', ' ', text).strip()

def process_xhtml_inplace(filepath, global_id_start, css_rel_path):
    filename = os.path.basename(filepath)
    print(f"Processing: {filename}")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'xml')

        # Cleanup & CSS
        for tag in soup.find_all("a"):
            if "noteref" in tag.get("class", []) or tag.get("role") == "doc-noteref": tag.decompose()
        
        head = soup.find('head')
        if head:
            css_name = os.path.basename(css_rel_path)
            if not any(css_name in l.get('href', '') for l in head.find_all('link')):
                head.append(soup.new_tag("link", rel="stylesheet", href=css_rel_path, type="text/css"))

        segments = []
        current_id = global_id_start
        block_tags = ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'blockquote', 'div']
        VOID_TAGS = {'br', 'img', 'hr', 'area', 'base', 'col', 'embed', 'input', 'link', 'meta', 'param', 'source', 'track', 'wbr'}

        for tag in soup.find_all(block_tags):
            if tag.find(block_tags): continue 
            
            full_text_raw = tag.get_text()
            if not full_text_raw.strip(): continue

            # Clean for NLTK
            full_text_clean = re.sub(r'\s+', ' ', full_text_raw).strip()
            sentences_clean = nltk.sent_tokenize(full_text_clean)
            if not sentences_clean: continue
            
            # 1. Fuzzy Boundary Calc
            split_indices = []
            cursor_raw = 0
            for i, sent_clean in enumerate(sentences_clean):
                safe_sent = re.escape(sent_clean)
                pattern_str = safe_sent.replace(r'\ ', r'\s+')
                pattern = re.compile(pattern_str)
                match = pattern.search(full_text_raw, cursor_raw)
                
                if match:
                    end_raw = match.end()
                    while end_raw < len(full_text_raw) and full_text_raw[end_raw].isspace():
                        end_raw += 1
                    cursor_raw = end_raw
                else:
                    cursor_raw = len(full_text_raw) # Fallback
                split_indices.append(cursor_raw)

            # 2. Reconstruction
            new_html_content = ""
            current_sent_idx = 0
            current_char_count = 0
            
            seg_id = f"f{current_id:06d}"
            segments.append({"id": seg_id, "text": clean_text_for_tts(sentences_clean[0])})
            current_id += 1
            
            new_html_content += f'<span id="{seg_id}">'
            
            def traverse(node, open_tags):
                nonlocal new_html_content, current_char_count, current_sent_idx, current_id
                
                if isinstance(node, str):
                    text = str(node)
                    while len(text) > 0:
                        if current_sent_idx >= len(split_indices):
                            new_html_content += text
                            current_char_count += len(text)
                            break

                        boundary = split_indices[current_sent_idx]
                        remaining_len = boundary - current_char_count
                        
                        # --- THE FIX IS HERE ---
                        # If we have reached (or passed) the boundary exactly at the end of this node,
                        # we must TRIGGER THE SPLIT logic to register the next sentence.
                        
                        if remaining_len <= 0:
                            # Exact match or drift
                            
                            # Close current
                            for t_name, _ in reversed(open_tags): new_html_content += f"</{t_name}>"
                            new_html_content += "</span>"
                            
                            # Increment Index
                            current_sent_idx += 1
                            
                            # Start Next (if exists)
                            if current_sent_idx < len(sentences_clean):
                                seg_id = f"f{current_id:06d}"
                                current_id += 1
                                segments.append({"id": seg_id, "text": clean_text_for_tts(sentences_clean[current_sent_idx])})
                                
                                new_html_content += f'<span id="{seg_id}">'
                                for t_name, t_attrs in open_tags:
                                    attr_str = " ".join([f'{k}="{v}"' for k,v in t_attrs.items()])
                                    new_html_content += f"<{t_name} {attr_str}>" if attr_str else f"<{t_name}>"
                            
                            # Note: We do NOT consume text here because remaining_len was 0.
                            # We just perform the state switch and loop again to process the text 
                            # (which now belongs to the new sentence) or exit if text is empty.
                            continue

                        # Normal Processing
                        if len(text) <= remaining_len:
                            new_html_content += text
                            current_char_count += len(text)
                            break
                        else:
                            # Split in middle of node
                            chunk = text[:remaining_len]
                            new_html_content += chunk
                            current_char_count += len(chunk)
                            
                            for t_name, _ in reversed(open_tags): new_html_content += f"</{t_name}>"
                            new_html_content += "</span>"
                            
                            current_sent_idx += 1
                            if current_sent_idx < len(sentences_clean):
                                seg_id = f"f{current_id:06d}"
                                current_id += 1
                                segments.append({"id": seg_id, "text": clean_text_for_tts(sentences_clean[current_sent_idx])})
                                
                                new_html_content += f'<span id="{seg_id}">'
                                for t_name, t_attrs in open_tags:
                                    attr_str = " ".join([f'{k}="{v}"' for k,v in t_attrs.items()])
                                    new_html_content += f"<{t_name} {attr_str}>" if attr_str else f"<{t_name}>"
                            
                            text = text[remaining_len:]

                elif node.name:
                    attrs = {k: " ".join(v) if isinstance(v, list) else v for k, v in node.attrs.items()}
                    attr_str = " ".join([f'{k}="{v}"' for k,v in attrs.items()])
                    tag_open = f"<{node.name} {attr_str}>" if attr_str else f"<{node.name}>"

                    if node.name in VOID_TAGS:
                        new_html_content += tag_open.replace(">", " />")
                    else:
                        open_tags.append((node.name, attrs))
                        new_html_content += tag_open
                        for child in node.contents:
                            traverse(child, open_tags)
                        new_html_content += f"</{node.name}>"
                        open_tags.pop()

            for child in tag.contents:
                traverse(child, [])
                
            new_html_content += "</span>"
            
            wrapped_content = f"<body>{new_html_content}</body>"
            new_soup = BeautifulSoup(wrapped_content, 'xml')
            tag.clear()
            if new_soup.body:
                for child in list(new_soup.body.contents): tag.append(child)

    except Exception as e:
        print(f"  [ERROR] Failed to process {filename}: {e}")
        return [], global_id_start

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(str(soup.prettify()))
    return segments, current_id



# -------------------------------------------------------------------------
# 4. AUDIO & SMIL GENERATION
# -------------------------------------------------------------------------
def get_audio_duration(filepath):
    try:
        cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", filepath]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return float(result.stdout.strip())
    except: return 0.0

def generate_tts_chunk_chatterbox(model, text, speaker, lang):
    """
    Basic wrapper for Chatterbox generation of a single chunk.
    """
    clean_chunk = text.replace("\n", " ").strip()
    if not clean_chunk: return np.zeros(24000)

    try:
        wav = model.generate(
            text=clean_chunk, 
            language_id=lang, 
            audio_prompt_path=speaker 
        )
        if hasattr(wav, "cpu"):
            wav = wav.cpu().numpy()
        return wav.flatten()
        
    except Exception as e:
        print(f"\n[Chatterbox Error] Failed on '{clean_chunk[:20]}...': {e}")
        return np.zeros(24000)

def format_time(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{int(h)}:{int(m):02d}:{s:06.3f}"

def create_smil_content(basename, segments, audio_filename, rel_text_path):
    xml = '<?xml version="1.0" encoding="utf-8"?>\n'
    xml += '<smil xmlns="http://www.w3.org/ns/SMIL" xmlns:epub="http://www.idpf.org/2007/ops" version="3.0">\n'
    xml += '  <body>\n'
    xml += f'    <seq id="{basename}_seq" epub:textref="{rel_text_path}">\n'

    audio_ref = f"../Audio/{audio_filename}" 

    for i, seg in enumerate(segments):
        clip_start = seg['start']
        if i < len(segments) - 1:
            clip_end = segments[i+1]['start']
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
            # seg['text'] is clean text
            text = seg['text'].replace("\n", " ").strip()
            if not text: continue
            
            # --- 1. SMART SPLIT ---
            chunks = smart_split_text(text)
            
            # --- 2. ELLIPSIS PROSODY ---
            chunks = apply_ellipsis_to_chunks(chunks)

            # --- 3. GENERATION ---
            chunk_wavs = []
            for chunk in chunks:
                w = generate_tts_chunk_chatterbox(model, chunk, speaker, lang)
                chunk_wavs.append(w)
            
            # --- 4. CROSSFADE ---
            if len(chunk_wavs) > 1:
                final_sentence_wav = crossfade_audio_segments(chunk_wavs, fade_ms=50)
            else:
                final_sentence_wav = np.concatenate(chunk_wavs) if chunk_wavs else np.zeros(0)

            dur = len(final_sentence_wav) / sr
            start = curr_time
            end = curr_time + dur
            
            sync_data.append({"id": seg['id'], "start": start, "end": end})
            full_audio.append(final_sentence_wav)
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
        for i, seg in enumerate(segments):
            sync_data.append({"id": seg['id'], "start": i*5.0, "end": (i+1)*5.0})

    smil_content = create_smil_content(basename, sync_data, f"{basename}.mp3", rel_text_path)
    with open(smil_path, "w", encoding="utf-8") as f:
        f.write(smil_content)

    return total_duration

# -------------------------------------------------------------------------
# 5. OPF & PACKAGING
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
        description="Generative Audiobook Creator (Chatterbox): Convert EPUB to EPUB 3 with Media Overlays."
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
    
    INPUT_EPUB = args.input_epub
    SPEAKER_WAV = args.voice
    BOOK_LANGUAGE = args.language
    GENERATE_AUDIO = not args.skip_audio

    input_basename = os.path.splitext(os.path.basename(INPUT_EPUB))[0]
    default_filename = f"{input_basename}_Narrated_Chatterbox.epub"

    if args.output:
        if os.path.isdir(args.output) or args.output.endswith(os.sep):
            os.makedirs(args.output, exist_ok=True)
            OUTPUT_EPUB = os.path.join(args.output, default_filename)
        else:
            OUTPUT_EPUB = args.output
            parent_dir = os.path.dirname(os.path.abspath(OUTPUT_EPUB))
            if parent_dir: os.makedirs(parent_dir, exist_ok=True)
    else:
        input_dir = os.path.dirname(os.path.abspath(INPUT_EPUB))
        OUTPUT_EPUB = os.path.join(input_dir, default_filename)

    input_dir = os.path.dirname(os.path.abspath(INPUT_EPUB))
    book_basename = os.path.splitext(os.path.basename(INPUT_EPUB))[0]
    WORK_DIR = os.path.join(input_dir, f".{book_basename}_workdir")

    print(f"=== STARTING CHATTERBOX PIPELINE ===")
    print(f"Input:  {INPUT_EPUB}")
    print(f"Output: {OUTPUT_EPUB}")
    print(f"Features: Smart Split + Ellipsis + Crossfade ENABLED")

    if not os.path.exists(INPUT_EPUB):
        print(f"Error: Input file not found: {INPUT_EPUB}")
        sys.exit(1)
    if not os.path.exists(SPEAKER_WAV):
        print(f"Error: Voice sample not found: {SPEAKER_WAV}")
        sys.exit(1)

    device = "cuda" if args.gpu and torch.cuda.is_available() else check_cuda()
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
        print(f"Loading Chatterbox on {device}...")
        tts = ChatterboxMultilingualTTS.from_pretrained(device=device)
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