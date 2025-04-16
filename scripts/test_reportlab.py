from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

def generate_conversation_pdf(output_filename, conversation, images):
    c = canvas.Canvas(output_filename, pagesize=letter)
    width, height = letter

    # Starting position for text
    y_position = height - 50

    for entry in conversation:
        speaker, text = entry
        c.setFont("Helvetica-Bold", 12 if speaker == "Chatbot" else 11)  # Different font for each speaker
        c.drawString(50, y_position, f"{speaker}: {text}")

        # Move text position down
        y_position -= 30

        # Insert corresponding image if available
        if speaker in images:
            img_path = images[speaker]
            img = ImageReader(img_path)
            c.drawImage(img, 50, y_position - 100, width=200, height=100, preserveAspectRatio=True)
            y_position -= 120  # Move down to avoid overlapping

        # Page break if necessary
        if y_position < 50:
            c.showPage()
            y_position = height - 50

    c.save()

# Example conversation and images
conversation = [
    ("Chatbot", "Hello! How can I assist you today?"),
    ("Codebase", "I need to generate a PDF with a conversation."),
    ("Chatbot", "You can use the reportlab library for that."),
]*20

images = {
    "Chatbot": "/home/aidan/predicators/mock_task_images/drawer_clean/container, blue cup in container.jpg",
    "Codebase": "/home/aidan/predicators/mock_task_images/drawer_clean/container, blue cup in container.jpg",
}

generate_conversation_pdf("conversation.pdf", conversation, images)