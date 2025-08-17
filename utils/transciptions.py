import os
import re
import json 
import vertexai
import math
import pandas as pd
from pydub import AudioSegment
from google.cloud import storage
from langchain_google_vertexai import ChatVertexAI
from vertexai.generative_models import GenerativeModel, Part
from langchain_core.messages import SystemMessage, HumanMessage


from pydub import AudioSegment
import os

def split_audio(file_path: str,
                chunk_length_ms: int = 5 * 60 * 1000,
                target_formats=None):
    """
    Split a WAV/MP3 file into chunks and export them in one or more formats (wav, mp3).
    Does NOT convert input. The caller must ensure the input is either .wav or .mp3.
    """

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    ext = os.path.splitext(file_path)[1].replace('.', '').lower()
    if ext not in ("wav", "mp3"):
        raise ValueError(f"split_audio() expects .wav or .mp3 as input, got: {ext}")

    if target_formats is None:
        target_formats = ["wav"]

    audio = AudioSegment.from_file(file_path)
    audio_length = len(audio)
    result = {}

    for fmt in target_formats:
        if fmt not in ("wav", "mp3"):
            raise ValueError("target_formats can contain only 'wav' or 'mp3'")

        chunks_dir = os.path.join(os.path.dirname(file_path), f"chunks_{fmt}")
        os.makedirs(chunks_dir, exist_ok=True)

        if audio_length <= chunk_length_ms:
            out_path = os.path.join(chunks_dir, f"{base_name}.{fmt}")
            audio.export(out_path, format=fmt)
            result[fmt] = [out_path]
            continue

        paths = []
        for i in range(0, audio_length, chunk_length_ms):
            chunk = audio[i:i + chunk_length_ms]
            name = f"{base_name}_part{i // chunk_length_ms + 1}.{fmt}"
            path = os.path.join(chunks_dir, name)
            chunk.export(path, format=fmt)
            paths.append(path)
        result[fmt] = paths

    return result



def upload_to_gcs(bucket_name: str, source_file_path: str, destination_blob_name: str) -> str:
    if not os.path.exists(source_file_path):
        raise FileNotFoundError(f"The file {source_file_path} was not found.")

    print(f"Uploading '{source_file_path}' to bucket '{bucket_name}'...")
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_path)
    gcs_uri = f"gs://{bucket_name}/{destination_blob_name}"
    print(f"File uploaded successfully: {gcs_uri}")
    return gcs_uri


def split_and_upload(bucket_name: str, source_file_path: str, gcs_folder: str, target_formats=None) -> list:
    """
    Converts source file to .wav if it's not .wav or .mp3.
    Then splits and uploads each chunk to GCS.
    """
    # Convert to WAV if needed
    base_name = os.path.splitext(os.path.basename(source_file_path))[0]
    ext = os.path.splitext(source_file_path)[1].replace('.', '').lower()

    if ext not in ("wav", "mp3"):
        # convert to wav
        original_audio = AudioSegment.from_file(source_file_path)
        converted_path = os.path.join(os.path.dirname(source_file_path), f"{base_name}.wav")
        original_audio.export(converted_path, format="wav")
        source_file_path = converted_path  # use the converted file going forward

    # Split the (now validated) file
    chunk_dict = split_audio(source_file_path, target_formats=target_formats)
    gcs_uris = []

    # Upload each chunk
    for _, chunk_files in chunk_dict.items():
        for chunk_file in chunk_files:
            chunk_name = os.path.basename(chunk_file)
            gcs_path = f"{gcs_folder}/{chunk_name}"
            uri = upload_to_gcs(bucket_name, chunk_file, gcs_path)
            gcs_uris.append(uri)

    return gcs_uris



# def process_audio_with_gemini(project_id, location, gcs_uri):
#     """
#     Processes an audio file using Gemini 2.5 Pro on Vertex AI with a fixed transcription + diarization prompt.
#     Uses ChatVertexAI with temperature=0 for deterministic, repeatable output.
#     """

#     # Stable, deterministic, JSON-only prompt
    # prompt_for_transcription = """
    #         You are an expert transcription and translation assistant. 
    #         Your task is to:
    #         1. Transcribe spoken Bengali into fluent and accurate English.
    #         2. Diarize the conversation by identifying distinct speakers.
    #         3. Assign each sentence to the correct speaker by analyzing the context, dialogue flow, and conversational logic — not just based on alternating turns.
    #         4. Maintain consistent speaker labels throughout the conversation, using 'Interviewer' and 'Candidate'.
    #         5. Ensure that each transcript line makes logical sense with the preceding and following lines (e.g., questions should be assigned to Interviewer, answers to Candidate).
    #         6. Keep sentences complete and avoid splitting in unnatural places.
    #         7. Do not loose any content from the audio.

    #         Think step-by-step before deciding the speaker for each line.

    #         After completing the transcription, go through this checklist before finalizing:
    #         - ✅ Have all Bengali lines been accurately translated into fluent English?
    #         - ✅ Are speaker labels ('Interviewer', 'Candidate') used consistently and correctly?
    #         - ✅ Does each line logically follow from the previous one in terms of who is speaking?
    #         - ✅ Are there no broken or incomplete sentences?
    #         - ✅ Is the JSON valid, with proper formatting and escaping of special characters?

    #         Output ONLY a valid JSON string in the following format:
    #         [
    #             {
    #                 "speaker": "Interviewer",
    #                 "transcript": "Where is your CV? Let me see the CV."
    #             },
    #             {
    #                 "speaker": "Interviewer",
    #                 "transcript": "Ishak Ali?"
    #             },
    #             {
    #                 "speaker": "Candidate",
    #                 "transcript": "Yes, sir."
    #             }
    #         ]
    #         """

#     print("Initializing ChatVertexAI with Gemini 2.5 Pro...")
#     llm = ChatVertexAI(
#         model="gemini-2.5-pro",
#         temperature=0.4,
#         project=project_id,
#         location=location
#     )

#     print("Sending request to Gemini model... (This may take a moment)")
#     try:
#         result = llm.invoke([
#             SystemMessage(content=prompt_for_transcription),
#             HumanMessage(content=f"Transcribe and diarize the audio at: {gcs_uri}")
#         ])
#         return result.content
#     except Exception as e:
#         return f"An error occurred: {e}"

def process_audio_with_gemini(project_id, location, gcs_uri):
    """
    Processes a Bengali audio file using Gemini 2.5 Pro for transcription + diarization + translation.
    Returns raw model output (expected JSON string).
    """
    print("Initializing Vertex AI...")
    vertexai.init(project=project_id, location=location)

    model = GenerativeModel("gemini-2.5-pro")

    prompt_for_transcription = """
    You are an expert transcription and translation assistant.
    Your task is to:
    1. Transcribe spoken Bengali into fluent and accurate English.
    2. Diarize and Identify speakers based on context and assign meaningful role labels (e.g., "Interviewer" and "Candidate", "Salesman" and "Shopkeeper").
    3. Assign sentences logically to the correct speaker.
    4. Keep sentences complete and do not lose content.
    Output ONLY a valid JSON array, nothing else.
    Format:
    [
        {"speaker": "Interviewer", "transcript": "Sample text"},
        {"speaker": "Candidate", "transcript": "Sample text"}
    ]
    """

    audio_file_part = Part.from_uri(uri=gcs_uri, mime_type="audio/mpeg")

    print("Sending request to Gemini model... (This may take a moment)")
    try:
        response = model.generate_content([prompt_for_transcription, audio_file_part])
        if not getattr(response, "text", None) or not response.text.strip():
            print("⚠ Gemini returned empty output")
            return None

        return response.text.strip()

    except Exception as e:
        print(f"❌ Error during Gemini request: {e}")
        return None


def generate_report(transcription,questionaire):
    """
    Generates a structured JSON interview report from a Bengali transcript using Gemini 1.5/2.5 Pro via ChatVertexAI (LangChain),
    based on a provided questionnaire and sample report format.

    The output will:
    - Contain one entry for each of the 31 main questions in the questionnaire.
    - Include extra questions (those asked but not in the questionnaire) in a separate section with quality scoring.
    - Contain interviewer and candidate feedback sections.
    - Be fully in English.
    - Be valid JSON loadable by Python's json library.
    """

    # Use an absolute path to the 'questions.txt' file
    script_dir = os.path.dirname(__file__)
    questions_file_path = os.path.join(script_dir, 'questionaires', questionaire)

    with open(questions_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    questions = data["questions"]
    prompt = data["report_prompt"]
    system_prompt = data["auditor_system_prompt"]
    messages = [
        SystemMessage(
            content=(
                system_prompt
            )
        ),
        HumanMessage(
            content=f"""
Questionnaire:
\"\"\"
{questions}
\"\"\"
Conversational Transcript:
\"\"\"
{transcription}
\"\"\"
{prompt}
\"\"\"
       
"""
        )
    ]

    chat = ChatVertexAI(
        model="gemini-2.5-pro",
        temperature=0,
        project=os.getenv("GCP_PROJECT"),
        location=os.getenv("GCP_LOCATION", "us-central1")
    )

    response = chat.invoke(messages)
    report = response.content.strip()
    return report


def clean_and_parse_json(raw_text):
    if not raw_text or not raw_text.strip():
        print("⚠ No text provided to parse")
        return None

    # Try fenced code block first
    json_match = re.search(r'```json\s*(.*?)\s*```', raw_text, re.DOTALL)
    if json_match:
        raw_json_str = json_match.group(1).strip()
    else:
        # Try extracting the first {...} or [...] JSON-like structure
        json_match = re.search(r'(\{.*\}|\[.*\])', raw_text, re.DOTALL)
        raw_json_str = json_match.group(1).strip() if json_match else raw_text.strip()

    # Remove trailing commas
    raw_json_str = re.sub(r',\s*\]', ']', raw_json_str)
    raw_json_str = re.sub(r',\s*\}', '}', raw_json_str)

    try:
        return json.loads(raw_json_str)
    except json.JSONDecodeError as e:
        print(f"❌ Failed to parse JSON. Error: {e}")
        print("--- Raw text received ---")
        print(raw_text)
        print("--- Extracted JSON string ---")
        print(raw_json_str)
        return None
    

def export_report_to_single_excel(report_data, output_file="report.xlsx"):
    """
    Exports a report (interview or sales_call) into a single-sheet Excel file.
    Automatically detects the report type based on the keys in the JSON.
    """

    # Parse JSON string if needed
    if isinstance(report_data, str):
        data = json.loads(report_data)
    else:
        data = report_data

    blocks = []

    # ---------------------------------------------------------------
    # Detect structure → Interview vs Sales Call
    # ---------------------------------------------------------------
    if "Interview_Questionair_Responses" in data:
        # → Interview report format (old)
        blocks.append(("Interview_Questionair_Responses", pd.DataFrame(data.get("Interview_Questionair_Responses", []))))
        blocks.append(("Extra Questions", pd.DataFrame(data.get("extra_questions", []))))
        blocks.append(("Interviewer Feedback", pd.DataFrame([data.get("interviewer_feedback", {})])))

        # Candidate feedback (flatten personality)
        candidate_feedback = data.get("candidate_feedback", {})
        personality = candidate_feedback.get("personality_assessment", {})
        flat_personality = {}
        for group, traits in personality.items():
            for trait, detail in traits.items():
                col_prefix = f"{group}_{trait}".replace(" ", "_")
                flat_personality[f"{col_prefix}_value"] = detail.get("value")
                flat_personality[f"{col_prefix}_reason"] = detail.get("reason")
        candidate_df = pd.DataFrame([{**candidate_feedback, **flat_personality}])
        candidate_df.drop(columns=["personality_assessment"], errors="ignore", inplace=True)
        blocks.append(("Candidate Feedback", candidate_df))

    else:
        # → Sales call report
        results = (
            data.get("Pre-Training Sales Call Recording") 
            if "Pre-Training Sales Call Recording" in data 
            else data
        )
        blocks.append(("Pre-Training Sales Call Recording", pd.DataFrame(results)))

        # Extra questions
        extra = data.get("extra_questions", [])
        if extra:
            blocks.append(("Extra Questions", pd.DataFrame(extra)))

    # ---------------------------------------------------------------
    # Write to Excel (single sheet, styled blocks)
    # ---------------------------------------------------------------
    with pd.ExcelWriter(output_file, engine="xlsxwriter") as writer:
        workbook = writer.book
        worksheet = workbook.add_worksheet("Report")
        writer.sheets["Report"] = worksheet

        section_fmt = workbook.add_format({
            "bold": True, "font_color": "white", "bg_color": "#4F81BD",
            "align": "center", "valign": "vcenter"
        })
        header_fmt = workbook.add_format({
            "bold": True, "bg_color": "#D9E1F2",
            "align": "center", "valign": "vcenter"
        })

        row_cursor = 0
        for title, df in blocks:
            worksheet.write(row_cursor, 0, title, section_fmt)
            row_cursor += 1

            if not df.empty:
                # header
                for col_num, col_name in enumerate(df.columns):
                    worksheet.write(row_cursor, col_num, col_name, header_fmt)

                # rows
                for r in range(len(df)):
                    for c in range(len(df.columns)):
                        worksheet.write(row_cursor + 1 + r, c, df.iat[r, c])

                row_cursor += len(df) + 1
            else:
                worksheet.write(row_cursor, 0, "(No data)")
                row_cursor += 1

            row_cursor += 2  # spacing

        for col_num in range(worksheet.dim_colmax + 1):
            worksheet.set_column(col_num, col_num, 22)

    print(f"✅ Report exported to {output_file}")


