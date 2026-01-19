"""PaddleOCR engine plugin for OCRmyPDF."""

from __future__ import annotations

import logging
from io import BytesIO
import base64
from pathlib import Path

import requests
from ocrmypdf import hookimpl
from ocrmypdf.pluginspec import OcrEngine, OrientationConfidence
from PIL import Image

log = logging.getLogger(__name__)


@hookimpl
def add_options(parser):
    """Add PaddleOCR-Remote-specific options to the argument parser."""
    paddle = parser.add_argument_group(
        "PaddleOCR",
        "Options for PaddleOCR engine"
    )
    paddle.add_argument(
        '--paddle-remote',
        metavar='URL', type=str,
        help='PaddleOCR remote server',
    )


@hookimpl
def check_options(options):
    """Validate PaddleOCR options."""
    pass


class PaddleOCRRemote:
    
    def __init__(self, lang: str, base_url: str) -> None:
        self.lang = lang
        self.base_url = base_url.rstrip('/') + '/'
        
    def predict(self, filepath: str):
        url = self.base_url + 'ocr'
        headers = {
            "Content-Type": "application/json"
        }
        
        try:
            buf = BytesIO()
            im = Image.open(filepath).convert('1')
            scale = float(im.width)
            im.thumbnail((3000, 3000))
            scale /= im.width
            im.save(buf, format='jpeg')
            
            req = {
                "file": base64.b64encode(buf.getvalue()).decode('ascii'),
                "fileType": 1,
                "returnWordBox": True,
                "visualize": False,
                "useDocUnwarping": False
            }
        
            response = requests.post(url, headers=headers, json=req, timeout=30)
            response.raise_for_status()
            result = response.json()
            pruned = result['result'].get('ocrResults', [{}])[0]
            
            if pruned:
                return pruned
            else:
                return {}
        except Exception as e:
            raise Exception(f'PaddleOCR Remote failure: {e}', e)


class PaddleOCREngine(OcrEngine):
    """Implements OCR with PaddleOCR."""

    # Mapping from Tesseract/OCRmyPDF language codes to PaddleOCR codes
    LANGUAGE_MAP = {
        'eng': 'en',
        'chi_sim': 'ch',
        'chi_tra': 'chinese_cht',
        'fra': 'fr',
        'deu': 'german',
        'jpn': 'japan',
        'kor': 'korean',
        'spa': 'spanish',
        'rus': 'ru',
        'ara': 'ar',
        'hin': 'hi',
        'por': 'pt',
        'ita': 'it',
        'tur': 'tr',
        'vie': 'vi',
        'tha': 'th',
    }

    @staticmethod
    def version():
        """Return PaddleOCR version."""
        return "3.3.x"

    @staticmethod
    def creator_tag(options):
        """Return the creator tag to identify this software."""
        return f"PaddleOCR {PaddleOCREngine.version()}"

    def __str__(self):
        """Return name of OCR engine and version."""
        return f"PaddleOCR {PaddleOCREngine.version()}"

    @staticmethod
    def languages(options):
        """Return the set of all languages supported by PaddleOCR."""
        # PaddleOCR supports many languages - return a comprehensive list
        return {
            'en', 'ch', 'chinese_cht', 'ta', 'te', 'ka', 'latin', 'ar', 'cy', 'da',
            'de', 'es', 'et', 'fr', 'ga', 'hi', 'it', 'ja', 'ko', 'la', 'nl', 'no',
            'oc', 'pt', 'ro', 'ru', 'sr', 'sv', 'tr', 'uk', 'vi',
            # Also include common Tesseract codes for compatibility
            'eng', 'chi_sim', 'chi_tra', 'deu', 'fra', 'spa', 'rus', 'jpn', 'kor'
        }

    @staticmethod
    def _get_paddle_lang(options):
        """Convert OCRmyPDF language to PaddleOCR language."""
        if not options.languages:
            return 'en'

        # Use first language
        lang = options.languages[0].lower()
        return PaddleOCREngine.LANGUAGE_MAP.get(lang, lang)

    @staticmethod
    def _get_paddle_ocr(options):
        """Create and configure PaddleOCR instance."""
        paddle_lang = PaddleOCREngine._get_paddle_lang(options)
        log.debug(f"Initializing PaddleOCR with language: {paddle_lang}")
        return PaddleOCRRemote(lang=paddle_lang, base_url=options.paddle_remote)

    @staticmethod
    def get_orientation(input_file: Path, options) -> OrientationConfidence:
        """Get page orientation."""
        # PaddleOCR handles orientation internally if use_angle_cls=True
        # Since we enable angle classification by default, we return neutral values
        return OrientationConfidence(angle=0, confidence=0.0)

    @staticmethod
    def get_deskew(input_file: Path, options) -> float:
        """Get deskew angle."""
        # PaddleOCR doesn't provide deskew information
        return 0.0

    @staticmethod
    def generate_hocr(input_file: Path, output_hocr: Path, output_text: Path, options):
        """Generate hOCR output for an image."""
        log.debug(f"Running PaddleOCR on {input_file}")

        # Initialize PaddleOCR
        paddle_ocr = PaddleOCREngine._get_paddle_ocr(options)

        # Get image dimensions and DPI info
        with Image.open(input_file) as img:
            width, height = img.size
            dpi = img.info.get('dpi', (300, 300))
            log.debug(f"Input image: {width}x{height}, DPI: {dpi}")

        # Run OCR - use predict() instead of deprecated ocr()
        result = paddle_ocr.predict(str(input_file))
        ocr_result = result['prunedResult']
        
        # Calculate scaling factors from preprocessed image
        scale_x = 1.0
        scale_y = 1.0
       
        log.debug(f"scaling factors: x={scale_x:.4f}, y={scale_y:.4f}")

        # Get language for hOCR
        lang = PaddleOCREngine._get_paddle_lang(options)
        # Map back to Tesseract-style language codes for compatibility
        lang_map_reverse = {v: k for k, v in PaddleOCREngine.LANGUAGE_MAP.items()}
        hocr_lang = lang_map_reverse.get(lang, 'eng')

        # Convert PaddleOCR 3.x output to hOCR
        hocr_lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN"',
            '    "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">',
            '<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en">',
            '<head>',
            '<title></title>',
            '<meta http-equiv="content-type" content="text/html; charset=utf-8" />',
            '<meta name="ocr-system" content="PaddleOCR via ocrmypdf-paddleocr" />',
            '<meta name="ocr-capabilities" content="ocr_page ocr_carea ocr_par ocr_line ocrx_word" />',
            '</head>',
            '<body>',
            f'<div class="ocr_page" id="page_1" title="bbox 0 0 {width} {height}">',
        ]

        # Collect all text for output_text
        all_text = []

        if ocr_result:
            
            # OCRResult is a dict-like object with keys: rec_texts, rec_scores, rec_polys
            texts = ocr_result.get('rec_texts', [])
            scores = ocr_result.get('rec_scores', [])
            polys = ocr_result.get('rec_boxes', [])

            log.debug(f"PaddleOCR found {len(texts)} text regions")

            word_id = 1
            carea_id = 1
            par_id = 1

            for line_id, (text, score, (x_min, y_min, x_max, y_max)) in enumerate(zip(texts, scores, polys), 1):
                if not text:
                    continue

                all_text.append(text)

                conf_pct = int(score * 100)

                # Create a carea and par for each line (simple structure)
                hocr_lines.append(
                    f'<div class="ocr_carea" id="carea_{carea_id}" title="bbox {x_min} {y_min} {x_max} {y_max}">'
                )
                hocr_lines.append(
                    f'<p class="ocr_par" id="par_{par_id}" lang="{hocr_lang}" title="bbox {x_min} {y_min} {x_max} {y_max}">'
                )

                # Start the line span with baseline info
                hocr_lines.append(
                    f'<span class="ocr_line" id="line_{line_id}" '
                    f'title="bbox {x_min} {y_min} {x_max} {y_max}; baseline 0 0; x_wconf {conf_pct}">'
                )

                # Split text into words and estimate bounding boxes
                # PaddleOCR doesn't provide word-level bboxes, so we estimate them
                words = text.split()
                if words:
                    line_width = x_max - x_min
                    # Calculate width available for words (excluding spaces)
                    total_chars = sum(len(w) for w in words)
                    num_spaces = len(words) - 1
                    # Allocate space for inter-word spaces
                    total_space_width = line_width - total_chars * (line_width / (total_chars + num_spaces))
                    space_width = int(total_space_width / num_spaces) if num_spaces > 0 else 0
                    # Width available for actual word characters
                    word_area_width = line_width - (space_width * num_spaces)

                    current_x = x_min
                    for i, word in enumerate(words):
                        # Estimate word width based on character proportion
                        if total_chars > 0:
                            word_width = int(word_area_width * len(word) / total_chars)
                        else:
                            word_width = line_width // len(words)

                        # For the last word, extend to line end to avoid rounding errors
                        if i == len(words) - 1:
                            word_x_max = x_max
                        else:
                            word_x_max = current_x + word_width

                        # Escape HTML entities in word
                        word_escaped = (word.replace('&', '&amp;')
                                           .replace('<', '&lt;')
                                           .replace('>', '&gt;'))

                        hocr_lines.append(
                            f'<span class="ocrx_word" id="word_{word_id}" '
                            f'title="bbox {current_x} {y_min} {word_x_max} {y_max}; '
                            f'x_wconf {conf_pct}">{word_escaped}</span>'
                        )

                        # Add space after word (except for last word)
                        if i < len(words) - 1:
                            hocr_lines.append(' ')

                        current_x = word_x_max + space_width
                        word_id += 1

                # Close the line span
                hocr_lines.append('</span>')
                # Close par and carea for this line
                hocr_lines.append('</p>')
                hocr_lines.append('</div>')

                carea_id += 1
                par_id += 1

        hocr_lines.extend([
            '</div>',  # ocr_page
            '</body>',
            '</html>',
        ])

        # Write hOCR output
        output_hocr.write_text('\n'.join(hocr_lines), encoding='utf-8')

        # Write text output
        text_content = '\n'.join(all_text)
        output_text.write_text(text_content, encoding='utf-8')

        log.debug(f"Generated hOCR with {len(all_text)} text regions")

    @staticmethod
    def generate_pdf(input_file: Path, output_pdf: Path, output_text: Path, options):
        """Generate a text-only PDF from an image.

        PaddleOCR doesn't have native PDF generation, so we use hOCR as intermediate
        and convert it to PDF using OCRmyPDF's HocrTransform.
        """
        log.debug(f"Generating PDF from {input_file}")

        # Create a temporary hOCR file
        output_hocr = output_pdf.with_suffix('.hocr')

        # Generate hOCR
        PaddleOCREngine.generate_hocr(input_file, output_hocr, output_text, options)

        # Convert hOCR to PDF using OCRmyPDF's hocrtransform
        from ocrmypdf.hocrtransform import HocrTransform
        # Get DPI from image
        from PIL import Image
        with Image.open(input_file) as img:
            dpi = img.info.get('dpi', (300, 300))[0]  # Default to 300 DPI if not set

        hocr_transform = HocrTransform(
            hocr_filename=output_hocr,
            dpi=dpi
        )
        hocr_transform.to_pdf(
            out_filename=output_pdf,
            image_filename=input_file,
            invisible_text=True  # Text should be invisible since it's an overlay
        )


@hookimpl
def get_ocr_engine():
    """Register PaddleOCR as an OCR engine."""
    return PaddleOCREngine()
