import os
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

import gradio as gr
import pandas as pd
import uvicorn
from fastapi import FastAPI, UploadFile, File

from core_pipeline import Img2BuildPipeline

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "output"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the heavy generation pipeline once when the service starts."""
    global pipeline
    print(">>> [Server] Loading Img2Build Pipeline models...")
    try:
        pipeline = Img2BuildPipeline(device="cuda")
        print(">>> [Server] Pipeline loaded and ready.")
    except Exception as exc:
        print(f">>> [Server] Error loading pipeline: {exc}")

    yield

    print(">>> [Server] Shutting down...")
    pipeline = None


app = FastAPI(
    title="Img2Build API",
    description="Anonymous review artifact for the Img2Build/Diff-LEGO pipeline.",
    version="1.3",
    lifespan=lifespan,
)


def run_inference(image_path, enable_repair, resolution_val):
    if not image_path:
        return None, None, None, 0, None, "Please upload an image first."

    if pipeline is None:
        return None, None, None, 0, None, "Error: Pipeline not initialized."

    try:
        mpd_path, assembly_mesh_path, brick_count, brick_df, log_text = pipeline.run(
            image_path,
            use_repair=enable_repair,
            resolution=int(resolution_val),
        )

        preview_raw = None
        if mpd_path:
            task_dir = Path(mpd_path).parent
            raw_glb = task_dir / "raw_mesh.glb"
            if raw_glb.exists():
                preview_raw = str(raw_glb)

        preview_assembly = None
        if assembly_mesh_path and os.path.exists(assembly_mesh_path):
            preview_assembly = str(assembly_mesh_path)

        return preview_raw, preview_assembly, mpd_path, brick_count, brick_df, log_text
    except Exception as exc:
        import traceback

        err_msg = f"Critical Error: {exc}\n{traceback.format_exc()}"
        return None, None, None, 0, None, err_msg


custom_css = """
.container { max-width: 1200px; margin: auto; }
h1 { text-align: center; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; color: #333; }
.desc { text-align: center; color: #666; font-size: 1.1em; margin-bottom: 20px; }
#logs { font-family: 'Courier New', monospace; font-size: 0.9em; }
.force-vertical-center { display: flex; align-items: center; height: 100%; }
.dataframe-wrap { max-height: 400px; overflow-y: auto; }
"""

with gr.Blocks(css=custom_css, title="Img2Build Demo") as demo:
    with gr.Column(elem_classes="container"):
        gr.Markdown("# Img2Build: Physically Buildable LEGO-style Structure from a Single Image")
        gr.Markdown(
            "<div class='desc'>End-to-end image to manufacturable assembly via 3D diffusion, "
            "risk-aware repair, and brick mapping.</div>"
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(type="filepath", label="Input Single-View Image", height=350)

                with gr.Row():
                    resolution_drop = gr.Dropdown(
                        choices=[8, 12, 16],
                        value=16,
                        label="Voxel Resolution",
                        info="The public demo caps resolution at 16 for smaller builds.",
                    )
                    enable_repair_box = gr.Checkbox(
                        label="Enable PPO Repair",
                        value=True,
                        elem_classes="force-vertical-center",
                    )

                run_btn = gr.Button("Generate Assembly", variant="primary", size="lg")

            with gr.Column(scale=1):
                with gr.Tabs():
                    with gr.Tab("1. Geometry"):
                        output_raw_3d = gr.Model3D(
                            label="Raw Mesh",
                            clear_color=[0.95, 0.95, 0.95, 1.0],
                            height=400,
                        )
                    with gr.Tab("2. Assembly Preview"):
                        output_assembly_3d = gr.Model3D(
                            label="Assembly",
                            clear_color=[1.0, 1.0, 1.0, 1.0],
                            interactive=True,
                            height=400,
                        )
                    with gr.Tab("3. Brick List (BOM)"):
                        with gr.Column(elem_classes="dataframe-wrap"):
                            output_bom = gr.Dataframe(
                                headers=["Part Name", "Color", "Color ID", "Quantity"],
                                datatype=["str", "str", "number", "number"],
                                label="Bill of Materials",
                                interactive=False,
                            )

                with gr.Row():
                    output_file = gr.File(label="Download .mpd", scale=2)
                    output_count = gr.Number(label="Total Bricks", value=0, precision=0, scale=1)

                with gr.Accordion("System Logs", open=False):
                    output_log = gr.Textbox(label="Logs", lines=10, elem_id="logs", interactive=False)

    run_btn.click(
        fn=run_inference,
        inputs=[input_image, enable_repair_box, resolution_drop],
        outputs=[output_raw_3d, output_assembly_3d, output_file, output_count, output_bom, output_log],
    )


app = gr.mount_gradio_app(app, demo, path="/")


@app.post("/api/generate")
async def api_generate(file: UploadFile = File(...), resolution: int = 16, repair: bool = True):
    if pipeline is None:
        return {"error": "Pipeline loading..."}

    task_id = Path(file.filename).stem
    local_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(local_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    mpd_path, assembly_mesh, count, df, logs = pipeline.run(
        local_path,
        task_id=task_id,
        use_repair=repair,
        resolution=resolution,
    )

    bom_data = df.to_dict(orient="records") if isinstance(df, pd.DataFrame) else []
    return {
        "status": "success" if mpd_path else "failed",
        "mpd_file": mpd_path,
        "assembly_mesh": assembly_mesh,
        "brick_count": count,
        "bom": bom_data,
        "logs": logs,
    }


if __name__ == "__main__":
    print("Starting Img2Build Server...")
    print("Please open http://127.0.0.1:8000 in your browser.")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
