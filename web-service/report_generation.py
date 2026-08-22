import os
import json
import datetime
from pathlib import Path
from PIL import Image
import io
import base64
from openai import OpenAI
import shutil
import ast, re
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Get API key from .env file ; save API key in .env file first
# ---------------------------------------------------------------------------

load_dotenv()

API_KEY = os.getenv("API_KEY")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key= API_KEY,
)

MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"

FEATURE_FIELDS = [
    "mass_shape",
    "microcalcification_pattern",
    "architectural_distortion",
    "breast_density_category",
    "malignant",
]

# --- Function schema ---
INSTRUCTION_PROMPT = f'''
You have access to functions. You MUST respond only in the function call format:
[func_name(params_name1=value1, params_name2=value2, ...)]

Do NOT include any other text. Do not confirm or explain anything. 
Respond ONLY with the function call.

Available functions:
[[
    {{
      "name": "mammogram_feature_extraction",
      "description": "Extract categorical features from a mammogram for research use. Return image-level labels including mass shape, microcalcification pattern, architectural distortion, breast density category, and a binary malignancy decision.",
      "parameters": {{
        "type": "object",
        "properties": {{
          "mass_shape": {{
            "type": "string",
            "enum": ["round", "oval", "lobulated", "irregular", "stellate"],
            "description": "Shape of the detected mass. Lobulated is usually benign, irregular or stellate are suspicious."
          }},
          "microcalcification_pattern": {{
            "type": "string",
            "enum": ["none", "scattered", "clustered", "linear", "segmental", "pleomorphic"],
            "description": "Pattern of microcalcifications. Clustered, linear, segmental, and pleomorphic are highly suspicious."
          }},
          "architectural_distortion": {{
            "type": "string",
            "enum": ["none", "mild", "moderate", "severe"],
            "description": "Degree of tissue distortion. Moderate or severe distortion increases suspicion."
          }},
          "breast_density_category": {{
            "type": "string",
            "enum": [
              "fatty",
              "scattered",
              "heterogeneously dense",
              "extremely dense"
            ],
            "description": "Shortened BI-RADS breast density categories with full text for heterogeneously dense."
          }},
          "malignant": {{
            "type": "string",
            "enum": ["yes", "no"],
            "description": "Binary label indicating whether the breast contains a malignant finding based on visible features in the image."
          }}
        }}
      }}
    }}
]]

These are the system instructions:
((
    You are an expert breast radiologist specializing in mammography interpretation
    Your task is to analyze each mammogram image and extract a set of categorical features
    Provide strictly structured output using the defined function schema, with no free-text narrative
    For each image, return only the following standardized fields: mass_shape, microcalcification_pattern, architectural_distortion, breast_density_category, and malignant
    Be strictly objective and evidence-based; report only features visible on the provided image and avoid hallucinating any findings
    If a feature is not visible or cannot be determined from the image, mark it with the appropriate categorical value (e.g., 'none')
    Do not speculate, infer, or incorporate information beyond what is directly observable in the mammogram
    Ensure consistency, reproducibility, and uniform use of BI-RADS terminology across all images analyzed
    Output must follow the function schema exactly, with all required fields present
))
'''

FEATURE_PROMPT = (
    "Analyze the attached mammogram image. "
    "Return a structured output ONLY, using the required categorical fields defined in the schema. "

    "DO NOT provide any free text, explanations, or narrative descriptions. "
    "Respond strictly in the format required by the function call and include ALL required fields. "

    "Required output fields: "
    "- mass_shape (round | oval | lobulated | irregular | stellate) "
    "- microcalcification_pattern (none | scattered | clustered | linear | segmental | pleomorphic) "
    "- architectural_distortion (none | mild | moderate | severe) "
    "- breast_density_category (fatty | scattered | heterogeneously dense | extremely dense) "
    "- malignant (yes | no) "

    "Report only features that are directly visible in the provided mammogram image. "
    "If a feature cannot be identified, assign the appropriate categorical value (e.g., none). "
    "Do not infer or speculate beyond the observable findings."
)

FEATURE_PROMPT = """
ROLE:
You are an expert breast radiologist analyzing a mammogram image for structured feature extraction.

TASK:
Analyze the attached mammogram image and return a structured output ONLY, using the required categorical fields defined in the schema.

OUTPUT REQUIREMENTS:
- Return ONLY the required categorical fields.
- Do NOT provide free text, explanations, or narrative descriptions.
- Respond strictly in the format required by the function call.
- Include ALL required fields.

REQUIRED OUTPUT FIELDS:
- mass_shape:
  round | oval | lobulated | irregular | stellate

- microcalcification_pattern:
  none | scattered | clustered | linear | segmental | pleomorphic

- architectural_distortion:
  none | mild | moderate | severe

- breast_density_category:
  fatty | scattered | heterogeneously dense | extremely dense

- malignant:
  yes | no

OBSERVATION RULES:
- Report only features that are directly visible in the provided mammogram image.
- If a feature cannot be identified, assign the appropriate categorical value, such as "none".
- Do NOT infer or speculate beyond the observable findings.
"""

REPORT_PROMPT = """
ROLE:
You are an expert breast radiologist dictating a complete, detailed mammography report.

INPUT:
You are given a structured set of imaging features extracted from a mammogram.
You MUST base your report ONLY on these features.
You are NOT allowed to introduce findings not directly supported by the input.

STRUCTURED FEATURES:
{features_json}

REPORT REQUIREMENTS:

- Examination:
  Specify mammography examination (screening mammography).

- Breast Composition:
  Translate breast_density_category into standard BI-RADS language.

- Findings:
  Describe mass morphology, calcification pattern, and architectural distortion.
  Do not invent locations, sizes, or laterality.

- Impression:
  Summarize overall concern level using professional radiology tone.

- BI-RADS Assessment:
  Assign a BI-RADS category consistent with the provided features.

- Recommendation:
  Provide an appropriate next step (routine screening, short-term follow-up, or diagnostic workup).

STYLE RULES:
- Clinical, neutral, professional tone
- No disclaimers
- No speculation
- No future prediction beyond standard radiology practice
- Bullet points where appropriate
"""

# ---------------------------------------------------------------------------
# Image encoding
# ---------------------------------------------------------------------------

def encode_image(image_path):
    img = Image.open(image_path)
    img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")

    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(image_path: str) -> dict:
    img = encode_image(image_path)

    messages = [
        {
            "role": "user",
            "content": INSTRUCTION_PROMPT,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": FEATURE_PROMPT,
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img}"
                    }
                }
            ]
        }
    ]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.2,
    )

    msg = response.choices[0].message.content
    match = re.search(r"\w+\((.*?)\)", msg, re.DOTALL)
    if not match:
        raise RuntimeError("Feature extraction failed")

    fake_call = f"dummy({match.group(1)})"
    tree = ast.parse(fake_call, mode="eval")

    features = {}
    for kw in tree.body.keywords:
        features[kw.arg] = ast.literal_eval(kw.value)

    for f in FEATURE_FIELDS:
        if f not in features:
            raise ValueError(f"Missing feature: {f}")

    return features


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report_from_features(features: dict) -> str:
    prompt = REPORT_PROMPT.format(
        features_json=json.dumps(features, indent=2)
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )

    return response.choices[0].message.content.strip()

# ---------------------------------------------------------------------------
# Markdown generation
# ---------------------------------------------------------------------------

def generate_md(report_text, metadata, output_dir: str):
    report_md = f"""
# Mammogram Report

**Image ID:** {metadata['image_id']}  
**Model:** {metadata['model']}  
**Date:** {metadata['timestamp']}  

---

{report_text}

---

## Notes
- For research use only
"""

    report_path = os.path.join(
        output_dir,
        f"{metadata['run_id']}_report.md",
    )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    return report_path


# ---------------------------------------------------------------------------
# Complete pipeline
# ---------------------------------------------------------------------------

def run_pipeline(image_path: str, output_dir: str):

    print("Starting report generation...")

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    metadata = {
        "run_id": run_id,
        "image_id": Path(image_path).stem,
        "model": MODEL,
        "timestamp": datetime.datetime.now().isoformat(),
    }

    features = extract_features(image_path)

    print("Extracted features:")
    print(json.dumps(features, indent=2))

    report = generate_report_from_features(features)

    md_path = generate_md(
        report,
        metadata,
        output_dir,
    )

    print(f"Report saved to: {md_path}")

    return md_path


# ---------------------------------------------------------------------------
# Entry point (not needed in web service)
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    OUTPUT_DIR = "web-service/generated_files"
    INPUT_DIR = "web-service/source_files"

    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.isdir(INPUT_DIR):
        raise FileNotFoundError(f"Input folder not found: {INPUT_DIR}")

    image_files = [
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    if len(image_files) != 1:
        raise ValueError(
            f"Expected exactly one image in '{INPUT_DIR}', "
            f"but found {len(image_files)}."
        )

    IMAGE_PATH = os.path.join(INPUT_DIR, image_files[0])

    if not IMAGE_PATH:
        raise ValueError(
            "Set IMAGE_PATH to the path of a mammogram image before running."
        )

    try:
        run_pipeline(IMAGE_PATH, OUTPUT_DIR)
    finally:
        # Remove the input folder contents after processing
        #shutil.rmtree(INPUT_DIR)

        os.makedirs(INPUT_DIR, exist_ok=True)