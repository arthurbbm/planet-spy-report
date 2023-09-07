from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Image, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from model import Field
from datetime import date


def generate_pdf(fields: list, output_file: str):
    """
    Generate a PDF with images sorted by rows and columns.

    Parameters:
    - image_pairs: list of tuples. Each tuple contains the field id, two image paths and the avg excess green.
    - output_file: path for the output PDF.
    """
    assert all((isinstance(field, Field) for field in fields))

    doc = SimpleDocTemplate(output_file, pagesize=landscape(letter))
    story = []

    styles = getSampleStyleSheet()
    header_text = f"Report of the experimental fields on {date.today().strftime('%B %d, %Y')}"
    header_paragraph = Paragraph(header_text, styles['Heading1'])
    story.append(header_paragraph)

    # data = [["Field ID", "Region", "Agent", "Latitude", "Longitude",
    #          "Avg Excess\nGreen", "Field Image", "Excess Green"]]
    data = [["Field ID", "Region", "Agent", "Latitude", "Longitude",
             "Field Image"]]

    # Add images to the table data
    for field in fields:
        field_image = Image(field.image_path, 2 * inch, 2 * inch, kind='bound')  # Adjust size as needed
        # excess_green_img = Image(field.excess_green_path, 2 * inch, 2 * inch, kind='bound')  # Adjust size as needed
        # data.append([field.field_id, field.region, field.agent, field.latitude, field.longitude,
        #              field.avg_excess_green, field_image, excess_green_img])
        data.append([field.field_id, field.region, field.agent, field.latitude, field.longitude,
                     field_image])

    # Create table with images
    # table = Table(data, colWidths=[0.6 * inch] * 3 + [inch] * 2 + [0.9 * inch] + [3 * inch] * 2, rowHeights=[0.5 * inch] + [2.5 * inch] * len(fields))
    table = Table(data, colWidths=[0.6 * inch] * 3 + [inch] * 2 + [3 * inch], rowHeights=[0.5 * inch] + [2.5 * inch] * len(fields))

    style = TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LINEABOVE', (0, 0), (-1, -1), 0.5, "black"),
        ('LINEABOVE', (0, 0), (-1, 0), 1, "black"),
        ('LINEBELOW', (0, 0), (-1, 0), 1, "black"),
        ('LINEBELOW', (0, -1), (-1, -1), 1, "black")
    ])
    table.setStyle(style)

    story.append(table)
    doc.build(story)
