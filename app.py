import spaces
import gradio as gr
import math
import torch
import yaml
from torchvision import transforms
from PIL import Image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


with open("plant_model.yaml", "r") as f:
    config = yaml.safe_load(f)

classes = config["classes"]
input_config = config["model"]["input"]


model = torch.jit.load(
    config["model"]["file"],
    map_location="cpu"
)
model.eval()


transform = transforms.Compose([
    transforms.Resize((input_config["height"], input_config["width"])),
    transforms.ToTensor(),
    transforms.Normalize(
        config["model"]['preprocessing']['normalize']['mean'],
        config["model"]['preprocessing']['normalize']['std']
    )
])

@spaces.GPU
def predict(files):
    results = []
    gallery = []

    if not files:
        return gallery, results

    model.to(device)

    for file in files:
        image = Image.open(file).convert("RGB")
        img = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img)
            prob = torch.softmax(outputs, dim=1)[0]

        confidence, pred = torch.max(prob, 0)

        predicted_class = classes[pred.item()]
        confidence_text = f"{confidence.item() * 100:.2f}%"

        gallery.append(
            (image.copy(), f"{predicted_class} — {confidence_text}")
        )

        results.append([
            file.split("/")[-1],
            predicted_class,
            confidence_text
        ])
        
    total_images = len(gallery)
    dynamic_rows = max(1, math.ceil(total_images / 2))

    return gr.update(value=gallery, rows=dynamic_rows), results

# Gradio 
with gr.Blocks(title="Plant Disease Classifier") as demo:
    gr.Markdown("""
    # Plant Disease Classification System

    Upload images to classify them into various plant disease categories.
    """)

    # 1. Place the upload area and the output gallery side by side using a Row
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.File(
                label="Upload Multiple Plant Images",
                file_count="multiple",
                file_types=["image"],
                type="filepath",
                height=320
            )
            predict_button = gr.Button("🔍 Detect Disease", variant="primary")

        with gr.Column(scale=1):
            # 2. 5 columns per row, compact height
            gallery_output = gr.Gallery(
                label="Uploaded Images with Predictions",
                columns=2,
                object_fit="contain",
                preview=True
            )

    # Prediction table placed full-width underneath
    output = gr.Dataframe(
        headers=["Image", "Predicted Class", "Confidence"],
        label="Prediction Results",
        interactive=False
    )

    predict_button.click(
        fn=predict,
        inputs=image_input,
        outputs=[gallery_output, output]
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())