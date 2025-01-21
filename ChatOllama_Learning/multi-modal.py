import base64
from io import BytesIO
from IPython.display import HTML, display
from PIL import Image

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

def convert_to_base64(pil_image):
    """
    Converts a PIL Image to a base64-encoded string.
    """
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def plt_img_base64(img_base64):
    """
    Displays an image from its base64-encoded string in Jupyter Notebook.
    """
    # Create an HTML img tag with the base64 string as the source
    image_html = f'<img src="data:image/jpeg;base64,{img_base64}" />'
    # Display the image by rendering the HTML
    display(HTML(image_html))

# Provide the file path to your image
file_path = "images/IMG_2473.jpeg"
pil_image = Image.open(file_path)

# Convert the image to a base64-encoded string
image_b64 = convert_to_base64(pil_image)

# Display the image in the notebook
plt_img_base64(image_b64)



llm = ChatOllama(model="llava:latest")

def prompt_func(data):
    text = data["text"]
    image = data["image"]

    image_part = {
        "type": "image_url",
        "image_url": f"data:image/jpeg;base64,{image}",
    }

    content_parts = []

    text_part = {"type": "text", "text": text}

    content_parts.append(image_part)
    content_parts.append(text_part)

    return [HumanMessage(content=content_parts)]

chain = prompt_func | llm | StrOutputParser()

query_chain = chain.invoke(
    {"text": "Whats in the picture", "image": image_b64}
)

print(query_chain)