"""
FastAPI server for the Medical LLM Report Generator.

The application serves both:
    1. The web frontend
    2. The report-generation API

Both are available through the same host and port.

Example:
    http://127.0.0.1:8000/

API endpoint:
    POST /generate-report
"""


from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    BackgroundTasks,
)

from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from pathlib import Path
import shutil
import tempfile

from report_generation import run_pipeline


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Medical LLM Report Generator",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------

app.mount(
    "/static",
    StaticFiles(directory="frontend"),
    name="static",
)


# ---------------------------------------------------------------------------
# Root endpoint
# ---------------------------------------------------------------------------

@app.get("/")
def root():

    return FileResponse(
        "frontend/index.html"
    )


# ---------------------------------------------------------------------------
# Report generation endpoint
# ---------------------------------------------------------------------------

@app.post("/generate-report")
async def generate_report(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):

    # -----------------------------------------------------------------------
    # Create isolated temporary directory
    # -----------------------------------------------------------------------

    request_dir = Path(
        tempfile.mkdtemp(
            prefix="mammogram_"
        )
    )

    image_path = request_dir / file.filename

    try:

        # -------------------------------------------------------------------
        # Save uploaded image
        # -------------------------------------------------------------------

        with image_path.open("wb") as buffer:
            shutil.copyfileobj(
                file.file,
                buffer,
            )

        # -------------------------------------------------------------------
        # Run report-generation pipeline
        # -------------------------------------------------------------------
        #
        # The pipeline receives:
        #
        #     1. Path to the uploaded image
        #     2. Directory in which to create the report
        #
        # It returns the path to the generated Markdown file.
        # -------------------------------------------------------------------

        report_path = run_pipeline(
            str(image_path),
            str(request_dir),
        )

        # -------------------------------------------------------------------
        # Schedule temporary-directory cleanup
        # -------------------------------------------------------------------
        #
        # The directory cannot be deleted immediately because FileResponse
        # still needs to read the generated Markdown file.
        #
        # BackgroundTasks executes the cleanup after the response is sent.
        # -------------------------------------------------------------------

        background_tasks.add_task(
            shutil.rmtree,
            str(request_dir),
            ignore_errors=True,
        )

        # -------------------------------------------------------------------
        # Return generated Markdown report
        # -------------------------------------------------------------------

        return FileResponse(
            path=report_path,
            media_type="text/markdown",
            filename="mammogram_report.md",
        )

    except Exception as e:

        # -------------------------------------------------------------------
        # Clean up immediately if report generation fails
        # -------------------------------------------------------------------

        shutil.rmtree(
            request_dir,
            ignore_errors=True,
        )

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )