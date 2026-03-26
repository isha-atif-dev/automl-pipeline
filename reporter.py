from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import io

def generate_pdf(results_df, best_name: str, task: str) -> bytes:
    """Generate a PDF report and return as bytes."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle(
        "title", parent=styles["Title"], fontSize=20, spaceAfter=10
    )
    story.append(Paragraph("AutoML Pipeline Report", title_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph(f"Task Type: {task.capitalize()}", styles["Normal"]))
    story.append(Paragraph(f"Best Model: {best_name}", styles["Normal"]))
    story.append(Spacer(1, 20))
    story.append(Paragraph("Model Comparison", styles["Heading2"]))
    story.append(Spacer(1, 10))

    data = [list(results_df.columns)] + results_df.values.tolist()
    table = Table(data)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2d3748")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.HexColor("#f7fafc"), colors.HexColor("#edf2f7")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    story.append(table)

    doc.build(story)
    buffer.seek(0)
    return buffer.read()