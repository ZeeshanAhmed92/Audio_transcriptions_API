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


def split_audio(file_path: str, chunk_length_ms: int = 5 * 60 * 1000):
    """
    Splits the audio file into chunks of specified length.
    
    Args:
        file_path (str): Path to the input audio file.
        chunk_length_ms (int): Length of each chunk in milliseconds (default 10 min).

    Returns:
        List[str]: List of paths to the chunked audio files.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    audio = AudioSegment.from_file(file_path)
    audio_length = len(audio)
    
    # If audio is shorter than the chunk size, return as-is
    if audio_length <= chunk_length_ms:
        return [file_path]

    chunks = []
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    ext = os.path.splitext(file_path)[1]

    for i in range(0, len(audio), chunk_length_ms):
        chunk = audio[i:i + chunk_length_ms]
        chunk_filename = f"{base_name}_part{i//chunk_length_ms + 1}{ext}"
        chunk_path = os.path.join(os.path.dirname(file_path), chunk_filename)
        chunk.export(chunk_path, format=ext.replace('.', ''))
        chunks.append(chunk_path)

    return chunks

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

def split_and_upload(bucket_name: str, source_file_path: str, gcs_folder: str) -> list:
    """
    Splits the audio into 10-min chunks, uploads each to GCS, and returns their URIs.
    """
    chunk_files = split_audio(source_file_path)
    gcs_uris = []

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


def generate_report(transcription):
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
    questions_file_path = os.path.join(script_dir, 'questionaires', 'questions.txt')

    with open(questions_file_path, 'r', encoding='utf-8') as file:
        questions = file.read()

    messages = [
        SystemMessage(
            content=(
                """You are a professional Auditor. You will be given:
1. A conversational transcript of an interview.
2. A set of predefined questionnaire questions.

Your job:
Fill the questionnaire in light of the provided transcript, strictly following the given format and scoring rules.

STRICT OUTPUT RULES:
- Use only the questionnaire that best matches the subject matter of the provided transcript.
- Include all 34 main questions from the questionnaire in the report, in the same order, without omission.
- Also identify and document any extra questions that were asked but are not in the questionnaire.
- Output must be valid JSON loadable directly in Python without any modifications.
"""
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

INSTRUCTIONS FOR REPORT GENERATION:
1. **Output Format**  
   - JSON array of objects, where each object represents one question (see Example below).
   - Each object must have:  
     `id`, `Competency`, `Predefined_Question`, `Question_asked`, `Question_Quality`, `Response`, `Summary`,  
     `Response_Quality`, `Question_score`, `Response_score`.

2. **Scoring Rules**  
   - Question_score:  
     1 = fully aligned with predefined question,  
     0.5 = partially aligned,  
     0 = not aligned at all.  
   - Response_score:  
     1 = fully answered,  
     0.5 = partially answered,  
     0 = wrong or negative answer.  
   - For extra questions not in the questionnaire:  
     Add them in a **separate section** of the JSON called `"extra_questions"`, each with:  
       `"Question"`, `"Quality"` ("Helpful", "Neutral", "Unhelpful"),  
       `"Question_quality_score"` (0.5 for Helpful, 0 for Neutral, -0.5 for Unhelpful), and  
       `"Summary"` (summarize if response is long).

3. **Coverage Rules**  
   - Every one of the 31 predefined questions must appear in the JSON, even if not asked (mark `"Question_Quality": "Not asked"`, `"Response_Quality": "Not answered"`, and scores = 0).
   - Do not skip any question from the questionnaire.

4. **Feedback Sections**  
   After the main JSON array, add:  
   - `"interviewer_feedback"`: total fully/partially/not asked counts, number of helpful/unhelpful extra questions, strengths, weaknesses, recommendations, and average interviewer score.  
   - `"candidate_feedback"`: average candidate score, response quality summary, recommendations, and personality assessment (positivity, honesty, humility, desire to work, discipline, job understanding, suitability).
   - Each `"reason"` must be concise (1–2 sentences) and supported by the transcript — no assumptions or fabricated details.
5. **Language**  
   - All content must be in English.
   - Preserve meaning exactly; do not invent information not present in the transcript.

Example JSON structure:
```json
{{ [ 
"questionnaire_responses":[
    {{
        "id": 1,
        "Competency": "Past work experience",
        "Predefined_Question": "<question from questionnaire>",
        "Question_asked": "Exact question asked",
        "Question_Quality": "Fully asked",
        "Reason for QQ": "<Reason for giving the question that quality>"
        "Response": "Exact candidate response",
        "Summary": "Brief 2-line summary of response",
        "Response_Quality": "Partially answered",
        "Reason for RQ": "<Reason for giving the Response that quality>"
        "Question_score": 1,
        "Response_score": 0.5
    }},
  ...
],
"extra_questions": [
  {{
    "Question": "Extra question text",
    "Quality": "Helpful",
    "Question_quality_score": 0.5,
    "Summary": "Short summary of answer"
  }}
],
"interviewer_feedback": {{
   "total_fully_asked": 20,
   "total_partially_asked": 8,
   "total_not_asked": 3,
   "helpful_extra_questions": 2,
   "unhelpful_extra_questions": 1,
   "strengths": "...",
   "weaknesses": "...",
   "recommendations": "...",
   "average_score": 0.78
}},
"candidate_feedback":
    - "average_score": Average candidate score across all predefined questions.
    - "response_quality_summary": Brief assessment of overall response quality.
    - "recommendations": Suggestions for improvement.
    - "personality_assessment": {{
    "Right Person": {{
        "Positivity": {{{{ "value": True/False, "reason": "<evidence from transcript>" }}}},
        "Cheerfulness/Energy": {{{{ "value": True/False, "reason": "<evidence from transcript>" }}}},
        "Willingness to learn": {{{{ "value": True/False, "reason": "<evidence from transcript>" }}}},
        "Honesty": {{{{ "value": True/false, "reason": "<evidence from transcript>" }}}},
        "Determination/Perseverance": {{{{ "value": True/False, "reason": "<evidence from transcript>" }}}}
    }},
    "Right Seat": {{
        "Understands the work": {{{{ "value": True/False, "reason": "<evidence from transcript>" }}}},
        "Wants to do the work": {{{{ "value": True/False, "reason": "<evidence from transcript>" }}}},
        "Has the ability to do the work": {{{{ "value": True/False, "reason": "<evidence from transcript>" }}}}
    }}
}}
    ]
    }}
       
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
    

def export_report_to_single_excel(report_data, output_file="interview_report2.xlsx"):
    """
    Exports all sections of the interview report into a single Excel sheet,
    each section with bold colored title row, bold colored header row,
    and 2 empty rows between sections.
    """
    # Ensure dict
    if isinstance(report_data, str):
        data = json.loads(report_data)
    else:
        data = report_data

    # Prepare blocks of (section_title, DataFrame)
    blocks = [
        ("Questionnaire Responses", pd.DataFrame(data.get("questionnaire_responses", []))),
        ("Extra Questions", pd.DataFrame(data.get("extra_questions", []))),
        ("Interviewer Feedback", pd.DataFrame([data.get("interviewer_feedback", {})])),
    ]

    # Candidate Feedback — flatten nested Right Person / Right Seat
    candidate_feedback = data.get("candidate_feedback", {})
    personality = candidate_feedback.get("personality_assessment", {})

    flat_personality = {}
    for group, traits in personality.items():  # group = "Right Person" / "Right Seat"
        for trait, detail in traits.items():
            col_prefix = f"{group}_{trait}".replace(" ", "_")
            flat_personality[f"{col_prefix}_value"] = detail.get("value")
            flat_personality[f"{col_prefix}_reason"] = detail.get("reason")
    candidate_df = pd.DataFrame([{**candidate_feedback, **flat_personality}])
    candidate_df.drop(columns=["personality_assessment"], errors="ignore", inplace=True)
    blocks.append(("Candidate Feedback", candidate_df))

    # Write to Excel
    with pd.ExcelWriter(output_file, engine="xlsxwriter") as writer:
        workbook = writer.book
        worksheet = workbook.add_worksheet("Interview Report")
        writer.sheets["Interview Report"] = worksheet

        # Formats
        section_format = workbook.add_format({
            "bold": True,
            "font_color": "white",
            "bg_color": "#4F81BD",
            "align": "center",
            "valign": "vcenter"
        })
        header_format = workbook.add_format({
            "bold": True,
            "bg_color": "#D9E1F2",
            "align": "center",
            "valign": "vcenter"
        })

        row_cursor = 0
        for section_title, df in blocks:
            # Section Title Row
            worksheet.write(row_cursor, 0, section_title, section_format)
            row_cursor += 1

            if not df.empty:
                # Write table header
                for col_num, col_name in enumerate(df.columns):
                    worksheet.write(row_cursor, col_num, col_name, header_format)

                # Write table rows
                for r in range(len(df)):
                    for c in range(len(df.columns)):
                        worksheet.write(row_cursor + 1 + r, c, df.iat[r, c])

                row_cursor += len(df) + 1  # +1 for header
            else:
                worksheet.write(row_cursor, 0, "(No data)")
                row_cursor += 1

            # Add 2 empty rows before next section
            row_cursor += 2

        # Auto column width
        for col_num in range(worksheet.dim_colmax + 1):
            worksheet.set_column(col_num, col_num, 20)

    print(f"✅ Report exported to {output_file} (single sheet, styled sections)")

