"""Interactive demo: paste 19th-century English newspaper text, see place
names (LOC/STREET/BUILDING) highlighted. Designed to run as a Hugging Face
Space using Gradio.

Usage locally:
    pip install gradio
    python app.py

To deploy on Hugging Face Spaces: create a new Space (SDK: Gradio), push
this file (renamed to app.py, which HF Spaces looks for by default),
requirements.txt, and your fine-tuned model folder (or push the model to
the Hub separately and load it by repo id below instead of a local path).
"""
import gradio as gr
from transformers import pipeline

# Once you've fine-tuned and either kept the model locally or pushed it to
# the Hub, point this at either a local folder ("results/model") or your
# Hub repo id ("your-username/historical-placename-ner").
MODEL_PATH = "results/model"

ner_pipeline = pipeline(
    "token-classification", model=MODEL_PATH, tokenizer=MODEL_PATH, aggregation_strategy="simple"
)

EXAMPLE_TEXT = (
    "The steamship left Liverpool for New York on Tuesday, calling at "
    "Queenstown before crossing the Atlantic. The Grand Hotel on King "
    "Street was reported to be fully booked for the season."
)


def predict(text: str):
    if not text.strip():
        return []
    entities = ner_pipeline(text)
    highlights = []
    cursor = 0
    for ent in entities:
        if ent["start"] > cursor:
            highlights.append((text[cursor:ent["start"]], None))
        highlights.append((text[ent["start"]:ent["end"]], ent["entity_group"]))
        cursor = ent["end"]
    if cursor < len(text):
        highlights.append((text[cursor:], None))
    return highlights


demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=5, value=EXAMPLE_TEXT, label="19th-century newspaper text"),
    outputs=gr.HighlightedText(label="Detected place names"),
    title="Historical Place-Name NER (19th-century English newspapers)",
    description=(
        "Fine-tuned on the topres19th dataset (Ardanuy et al., 2022 / "
        "HIPE-2022 shared task) — real 19th-century English newspaper text "
        "annotated for LOC (general places), STREET, and BUILDING entities. "
        "Full code: [link to GitHub repo]"
    ),
    examples=[
        [EXAMPLE_TEXT],
        ["A fire broke out near London Bridge yesterday evening, spreading "
         "quickly along Fish Street Hill before firemen from the nearby "
         "station could bring it under control."],
    ],
)

if __name__ == "__main__":
    demo.launch()
