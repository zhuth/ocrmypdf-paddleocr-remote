# OCRmyPDF-PaddleOCR-Remote

A PaddleOCR plugin for OCRmyPDF, enabling the use of a remote PaddleOCR served by Paddlex as an alternative OCR engine to Tesseract.

## Prerequisite

You should run a PaddleOCR-enabled PaddleX service. Consider starting with the following Dockerfile:

```Dockerfile
FROM ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/paddlex:paddlex3.3.4-paddlepaddle3.2.0-gpu-cuda11.8-cudnn8.9-trt8.6
RUN python -m pip install "paddleocr[all]==3.3.2"
RUN paddlex --install serving
```

Buid it, and then start with `docker run -d --gpus=all -p<YOUR_DESIRED_PORT>:8080 <BUILD_TAG_FOR_ABOVE_DOCKERFILE>` to utilize GPU and expose it.

## Installation

### Using pip

```bash
# Install from source
pip install .

# Or in development mode
pip install -e .
```

### Dependencies

- Python >= 3.8
- OCRmyPDF >= 14.0.0
- Pillow >= 9.0.0

## Usage

### Command Line

Use PaddleOCR as the OCR engine with the `--plugin` flag, suppose you are exposing paddlex-paddleocr on `localhost:8080`:

```bash
ocrmypdf --plugin ocrmypdf_paddleocr_remote --paddle-remote http://localhost:8080 input.pdf output.pdf
```

### Python API

```python
import ocrmypdf

ocrmypdf.ocr(
    'input.pdf',
    'output.pdf',
    plugins=['ocrmypdf_paddleocr'],
    paddle_remote='http://localhost:8080'
)
```

## Command Line Options

The plugin adds the following PaddleOCR-specific options:

- `--paddle-remote`: Remote Paddle OCR Endpoint

## Supported Languages

No `language` option required actually. PaddleOCR can handle that automatically.

## How It Works

The plugin implements the OCRmyPDF `OcrEngine` interface, by invoking a remote PaddleOCR API to processes each page image and returns bounding boxes with recognized text and confidence scores. The plugin converts this to hOCR (HTML-based OCR) format, which OCRmyPDF uses to create a searchable PDF.

## Bounding Box Accuracy

Thanks to the original `ocrmypdf-paddleocr` plugin, this plugin also includes optimized bounding box calculation for accurate text selection in the output PDF:

### Improved Word-Level Boxes

PaddleOCR provides line-level text detection only. The plugin estimates word-level bounding boxes by:
- Properly allocating horizontal space proportionally to character count
- Accounting for inter-word spacing to eliminate gaps at line ends
- Ensuring the last word extends to the full line width

**Result**: Word bounding boxes now accurately cover the entire line with zero gap at the end.

### Polygon-Based Vertical Bounds

Instead of using simple min/max coordinates, the plugin uses PaddleOCR's 4-point polygon geometry:
- For horizontal text, points 0-1 define the top edge and points 2-3 define the bottom edge
- Averaging these edge points provides tighter vertical bounds
- Falls back to min/max for non-standard polygon shapes

**Result**: Line heights are reduced by 2-3 pixels (3-6%), providing tighter text selection without clipping.

These improvements make text selection in the output PDF more precise and visually aligned with the actual text in the document. For technical details, see [CLAUDE.md](CLAUDE.md).

## License

MPL-2.0 - Same as OCRmyPDF

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Credits

- [OCRmyPDF](https://github.com/ocrmypdf/OCRmyPDF) - PDF OCR tool
- [OCRmyPDF-PaddleOCR](https://github.com/clefru/ocrmypdf-paddleocr) - The original PaddleOCR plugin
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) - Multilingual OCR toolkit
- [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) - Deep learning framework

## See Also

- [OCRmyPDF Documentation](https://ocrmypdf.readthedocs.io/)
- [OCRmyPDF Plugin Development](https://ocrmypdf.readthedocs.io/en/latest/plugins.html)
- [PaddleOCR Documentation](https://github.com/PaddlePaddle/PaddleOCR/blob/main/README_en.md)
