# OCRmyPDF-PaddleOCR-Remote

A PaddleOCR plugin for OCRmyPDF, enabling the use of a remote PaddleOCR served by paddlex as an alternative OCR engine to Tesseract.

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

### With Language Selection

```bash
# English
ocrmypdf --plugin ocrmypdf_paddleocr_remote --paddle-remote http://localhost:8080 -l eng input.pdf output.pdf

# Chinese Simplified
ocrmypdf --plugin ocrmypdf_paddleocr_remote --paddle-remote http://localhost:8080 -l chi_sim input.pdf output.pdf

# Multiple languages (uses first language)
ocrmypdf --plugin ocrmypdf_paddleocr_remote --paddle-remote http://localhost:8080 -l eng+fra input.pdf output.pdf
```

### Python API

```python
import ocrmypdf

ocrmypdf.ocr(
    'input.pdf',
    'output.pdf',
    plugins=['ocrmypdf_paddleocr'],
    language='eng',
    paddle_remote='http://localhost:8080'
)
```

## Command Line Options

The plugin adds the following PaddleOCR-specific options:

- `--paddle-remote`: Remote Paddle OCR Endpoint

## Supported Languages

PaddleOCR supports many languages. The plugin maps common Tesseract language codes to PaddleOCR codes:

| Tesseract Code | PaddleOCR Code | Language |
|---------------|----------------|----------|
| eng | en | English |
| chi_sim | ch | Chinese Simplified |
| chi_tra | chinese_cht | Chinese Traditional |
| fra | fr | French |
| deu | german | German |
| spa | spanish | Spanish |
| rus | ru | Russian |
| jpn | japan | Japanese |
| kor | korean | Korean |
| ara | ar | Arabic |
| hin | hi | Hindi |
| por | pt | Portuguese |
| ita | it | Italian |
| tur | tr | Turkish |
| vie | vi | Vietnamese |
| tha | th | Thai |

And many more! See PaddleOCR documentation for the complete list.

## Development

### Building from Source

```bash
# Install in development mode
pip install -e .

# Build distribution
python -m build
```

## How It Works

The plugin implements the OCRmyPDF `OcrEngine` interface, which requires:

1. **Language support**: Maps OCRmyPDF/Tesseract language codes to PaddleOCR codes
2. **Text detection**: Uses PaddleOCR to detect text regions in images
3. **Text recognition**: Recognizes text within detected regions
4. **hOCR generation**: Converts PaddleOCR output to hOCR format for OCRmyPDF to overlay on PDFs

PaddleOCR processes each page image and returns bounding boxes with recognized text and confidence scores. The plugin converts this to hOCR (HTML-based OCR) format, which OCRmyPDF uses to create a searchable PDF.

## Bounding Box Accuracy

This plugin includes optimized bounding box calculation for accurate text selection in the output PDF:

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

## Troubleshooting

### Poor OCR Quality

Try these options:

1. Increase image quality: `--oversample 300`
2. Preprocess images: `--clean` or `--deskew`
3. Disable angle classification if it's causing issues: `--paddle-no-angle-cls`

### GPU Not Being Used

Verify PaddlePaddle GPU installation:

```python
import paddle
print(paddle.device.is_compiled_with_cuda())  # Should return True
print(paddle.device.get_device())  # Should show GPU
```

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
