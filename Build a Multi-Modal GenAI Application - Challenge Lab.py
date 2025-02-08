import argparse

import vertexai
from vertexai.preview.vision_models import ImageGenerationModel
from vertexai.generative_models import GenerativeModel, Part
import logging

# --------  Important: Variable declaration  --------
project_id = "qwiklabs-gcp-00-cafce5c8da19"
location = "europe-west4"

# Initialize GCP logging
logging.basicConfig(level=logging.INFO)

# Initialize Vertex AI
vertexai.init(project=project_id, location=location)
# Load the model
model = GenerativeModel("gemini-1.0-pro-vision")
chat = model.start_chat()


def generate_bouquet_image(
    project_id: str, location: str, output_file: str, prompt: str
) -> vertexai.preview.vision_models.ImageGenerationResponse:
    """Generate an image using a text prompt.
    Args:
      project_id: Google Cloud project ID, used to initialize Vertex AI.
      location: Google Cloud region, used to initialize Vertex AI.
      output_file: Local path to the output image file.
      prompt: The text prompt describing what you want to see."""

    model = ImageGenerationModel.from_pretrained("imagegeneration@002")

    images = model.generate_images(
        prompt=prompt,
        # Optional parameters
        number_of_images=1,
        seed=1,
        add_watermark=False,
    )

    images[0].save(location=output_file)

    return images


def analyze_bouquet_image(image_uri: str):
    # Load the model
    multimodal_model = GenerativeModel("gemini-1.0-pro-vision")
    logging.info(f'Sending Image: {image_uri}')
    # Query the model
    response_chunks = multimodal_model.generate_content(
        [
            # Add an example image
            Part.from_uri(image_uri, mime_type="image/jpeg"),
            # Add an example query
            "generate birthday wishes based on the image passed",
        ],
        stream=True
    )

    text_response = []
    for chunk in response_chunks:
        text_response.append(chunk.text)

    full_response = "".join(text_response)
    logging.info(f'Received response: {full_response}')

    return full_response


generate_bouquet_image(
    project_id=project_id,
    location=location,
    output_file='image.jpeg',
    prompt='Create an image containing a bouquet of 2 sunflowers and 3 roses',
)

response_chunks = analyze_bouquet_image(
    image_uri='gs://generativeai-downloads/images/scones.jpg')
print(response_chunks)
