from pathlib import Path

import pypdfium2 as pdfium


root = Path(__file__).resolve().parents[2]
source = root / "C题.pdf"
out_dir = root / "tmp" / "pdfs" / "c_problem"
out_dir.mkdir(parents=True, exist_ok=True)
pdf = pdfium.PdfDocument(source)
for index in range(len(pdf)):
    page = pdf[index]
    bitmap = page.render(scale=2.0)
    image = bitmap.to_pil()
    image.save(out_dir / f"page-{index + 1}.png")
print(out_dir)
