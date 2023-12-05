import numpy as np

from deepsparse import BasePipeline
from deepsparse.clip import (
    CLIPTextInput,
    CLIPVisualInput,
    CLIPZeroShotInput
)

possible_classes = ["ice cream", "an elephant", "a dog", "a building", "a church"]
images = ["basilica.jpg", "buddy.jpeg", "thailand.jpg", "basilica.jpg", "buddy.jpeg", "thailand.jpg", "thailand.jpg", "thailand.jpg", "thailand.jpg"]

model_path_text = "models-rs/ViT-B-16-plus-240-laion400m_e32/textual_laion_chrisnoquant.onnx"
model_path_visual = "models-rs/ViT-B-16-plus-240-laion400m_e32/visual_laion_chrisnoquant.onnx"
model_path_visual = "models-rs/ViT-B-16-plus-240-laion400m_e32/visual_laion_chris_cpu.onnx"
model_path_text = "models-rs/ViT-B-16-plus-240-laion400m_e32/textual_laion_chris_cpu.onnx"

kwargs = {
    "visual_model_path": model_path_visual,
    "text_model_path": model_path_text,
}
pipeline = BasePipeline.create(task="clip_zeroshot", **kwargs)

pipeline_input = CLIPZeroShotInput(
    image=CLIPVisualInput(images=images),
    text=CLIPTextInput(text=possible_classes),
)

output = pipeline(pipeline_input).text_scores
print("output is", output)
for i in range(len(output)):
    prediction = possible_classes[np.argmax(output[i])]
    print(f"Image {images[i]} is a picture of {prediction}")
