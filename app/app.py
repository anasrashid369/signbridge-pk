"""Gradio demo for the SignBridge PK models (Deliverable 3).

Run it from the repository root:

    python app/app.py

The server binds to ``server.host`` and ``server.port`` in ``configs/app.yaml``
and never creates a public link, because hosting is out of scope.
"""

from __future__ import annotations

import gradio as gr

from signbridge.app_config import AppSettings, load_app_settings
from signbridge.app_service import PredictionResult, predict_image

MODELS = ("cnn", "landmark")


def honesty_markdown(settings: AppSettings) -> str:
    """Return the honesty notes that sit under the results."""
    lines = ["### About these numbers"]
    lines.extend(f"- {message}" for message in settings.messages.values())
    return "\n".join(lines)


def build_app(settings: AppSettings | None = None) -> gr.Blocks:
    """Build the Blocks app; kept separate so tests can build it without launching."""
    settings = settings or load_app_settings()
    default_model = settings.default_model if settings.default_model in MODELS else MODELS[0]

    with gr.Blocks(title="SignBridge PK") as demo:
        gr.Markdown(
            "# SignBridge PK\n"
            "ASL fingerspelling demo for the letters A-Z. "
            "Upload a photo or take a webcam picture, choose a model and press **Predict**."
        )
        with gr.Row():
            with gr.Column():
                image = gr.Image(
                    type="numpy",
                    sources=["upload", "webcam"],
                    label="Hand image",
                    height=320,
                )
                model = gr.Radio(choices=list(MODELS), value=default_model, label="Model")
                top_k = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=settings.top_k,
                    step=1,
                    label="Predictions to show",
                )
                button = gr.Button("Predict", variant="primary")
            with gr.Column():
                status = gr.Markdown("Upload an image, then press **Predict**.")
                table = gr.Dataframe(
                    headers=["Rank", "Letter", "Probability"],
                    datatype=["number", "str", "number"],
                    interactive=False,
                    label="Top predictions",
                )
                overlay = gr.Image(
                    type="numpy",
                    label="Landmark overlay (landmark model only)",
                    height=320,
                )

        gr.Markdown(honesty_markdown(settings))

        def run(
            uploaded: object, model_name: str, count: float
        ) -> tuple[str, list[list[object]], object]:
            result: PredictionResult = predict_image(uploaded, model_name, count, settings)
            return result.status, result.rows, result.overlay

        button.click(fn=run, inputs=[image, model, top_k], outputs=[status, table, overlay])

    return demo


def main() -> None:
    """Load the settings and serve the app on the configured host and port."""
    settings = load_app_settings()
    build_app(settings).launch(
        server_name=settings.host,
        server_port=settings.port,
        share=settings.share,
        inbrowser=True,
        max_file_size=settings.max_file_bytes,
    )


if __name__ == "__main__":
    main()
