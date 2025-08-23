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


def generate_report(transcription, questionaire):
    """
    Generates a structured JSON report from a transcript using Gemini 1.5/2.5 Pro via ChatVertexAI.
    Supports:
    - Pre-call/sales call reports (Pre_Call_Planning / While_in_the_Shop)
    - Interactive Training Session reports (audio_3.json)
    - Both dict-of-sections and flat list questionnaires
    - Handles both 'questions' and 'topics' keys
    """
    script_dir = os.path.dirname(__file__)
    questions_file_path = os.path.join(script_dir, 'questionaires', questionaire)

    with open(questions_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 🔹 Prefer `topics` but fallback to `questions`
    if "topics" in data:
        items = data["topics"]
    elif "questions" in data:
        items = data["questions"]
    else:
        raise ValueError("JSON must contain either `topics` or `questions` key")

    report_prompt = data["report_prompt"]
    system_prompt = data["auditor_system_prompt"]

    # Handle both structures (sections vs flat list)
    if isinstance(items, dict):
        questionnaire_text = json.dumps(items, ensure_ascii=False, indent=2)
    elif isinstance(items, list):
        questionnaire_text = json.dumps({"Topics": items}, ensure_ascii=False, indent=2)
    else:
        raise ValueError("`topics/questions` must be a dict (sections) or a list (flat)")

    # Add topic_context if present
    extra_context_text = ""
    if questionaire in ["audio_2_4_5.json", "audio_3.json", "audio_6.json", "audio_7.json"] and "topic_context" in data:
        topic_context_text = json.dumps(data["topic_context"], ensure_ascii=False, indent=2)
        extra_context_text = f"\n\nTopic Context:\n\"\"\"\n{topic_context_text}\n\"\"\""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=f"""
Questionnaire (Topics):
\"\"\"{questionnaire_text}\"\"\"
{extra_context_text}

Conversational Transcript:
\"\"\"{transcription}\"\"\"

{report_prompt}
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


def export_report_to_single_excel(report_data, output_file="interview_report.xlsx"):
    """
    Exports all sections of the report into a single Excel sheet.
    Handles INTERVIEW-type reports, SALES-CALL / Interactive Training reports,
    Interactive Training Session sections, Candidate Self Understanding,
    and Manager Compliance Assessment.
    Adds totals and percentages for Recruiter, Candidate, and Manager scores.
    All section headers use blue color.
    """

    # Ensure dict
    if isinstance(report_data, str):
        data = json.loads(report_data)
    else:
        data = report_data

    blocks = []

    # ------------------------------
    # SALES-CALL / INTERACTIVE TRAINING REPORT
    # ------------------------------
    if "Pre_Call_Planning" in data and "While_in_the_Shop" in data:
        blocks.append(("Pre_Call_Planning", pd.DataFrame(data.get("Pre_Call_Planning", []))))
        blocks.append(("While_in_the_Shop", pd.DataFrame(data.get("While_in_the_Shop", []))))

        # Add Extra Topics only if present
        extra_topics = data.get("Extra_Topics", [])
        if extra_topics:
            blocks.append(("Extra_Topics", pd.DataFrame(extra_topics)))

        totals_dict = {}

        # --- Pre-Call Planning Totals ---
        pre_scores = [t.get("Topic_Score", 0) or 0 for t in data["Pre_Call_Planning"]]
        pre_subtotal = sum(pre_scores)
        pre_max = 2 * len(data["Pre_Call_Planning"])
        pre_percentage = round((pre_subtotal / pre_max) * 100, 2) if pre_max > 0 else 0.0
        pre_percentage = min(pre_percentage, 100.0)  # cap at 100

        totals_dict["Pre_Call_Planning_Subtotal"] = pre_subtotal
        totals_dict["Pre_Call_Planning_Max_Possible_Score"] = pre_max
        totals_dict["Pre_Call_Planning_Percentage"] = pre_percentage

        # --- While in the Shop Totals ---
        shop_scores = [t.get("Topic_Score", 0) or 0 for t in data["While_in_the_Shop"]]
        shop_subtotal = sum(shop_scores)
        shop_max = 2 * len(data["While_in_the_Shop"])
        shop_percentage = round((shop_subtotal / shop_max) * 100, 2) if shop_max > 0 else 0.0
        shop_percentage = min(shop_percentage, 100.0)  # cap at 100

        totals_dict["While_in_the_Shop_Subtotal"] = shop_subtotal
        totals_dict["While_in_the_Shop_Max_Possible_Score"] = shop_max
        totals_dict["While_in_the_Shop_Percentage"] = shop_percentage

        # --- Extra Topics Totals ---
        if extra_topics:
            extra_scores = [t.get("Topic_Score", 0) or 0 for t in extra_topics]
            extra_subtotal = sum(extra_scores)
            extra_max = len(extra_topics) * 0.5
            extra_percentage = round((extra_subtotal / extra_max) * 100, 2) if extra_max > 0 else 0.0

        else:
            extra_subtotal, extra_max, extra_percentage = 0.0, 0.0, 0.0

        totals_dict["Extra_Topics_Subtotal"] = extra_subtotal
        totals_dict["Extra_Topics_Max_Possible_Score"] = extra_max
        totals_dict["Extra_Topics_Percentage"] = extra_percentage

        blocks.append(("Totals & Percentages", pd.DataFrame([totals_dict])))


    # ------------------------------
    # Interactive Training Session
    # ------------------------------
    if "Interactive Training Session Conducted by Recruiter" in data:
        section_topics = data["Interactive Training Session Conducted by Recruiter"]
        blocks.append(("Interactive Training Session Conducted by Recruiter", pd.DataFrame(section_topics)))

        num_topics = len(section_topics)
        recruiter_scores = [t.get("Recruiter_Score", 0) or 0 for t in section_topics]
        candidate_scores = [t.get("Candidate_Score", 0) or 0 for t in section_topics]

        totals_dict = {
            "Recruiter_Total_Earned": sum(recruiter_scores),
            "Recruiter_Max": 2 * num_topics,
            "Recruiter_Percentage": round((sum(recruiter_scores) / (2 * num_topics)) * 100, 2) if num_topics > 0 else 0,
            "Candidate_Total_Earned": sum(candidate_scores),
            "Candidate_Max": 2 * num_topics,
            "Candidate_Percentage": round((sum(candidate_scores) / (2 * num_topics)) * 100, 2) if num_topics > 0 else 0
        }

        # Handle Extra Topics if present
        if "Extra_Topics" in data:
            extra_topics = data["Extra_Topics"]
            blocks.append(("Extra_Topics", pd.DataFrame(extra_topics)))

            num_extra = len(extra_topics)
            recruiter_extra_scores = [t.get("Recruiter_Score", 0) or 0 for t in extra_topics]
            candidate_extra_scores = [t.get("Candidate_Score", 0) or 0 for t in extra_topics]

            totals_dict.update({
                "Recruiter_Extra_Subtotal": sum(recruiter_extra_scores),
                "Recruiter_Extra_Max": 1 * num_extra,
                "Recruiter_Extra_Percentage": round((sum(recruiter_extra_scores) / (1 * num_extra)) * 100, 2) if num_extra > 0 else 0,
                "Candidate_Extra_Subtotal": sum(candidate_extra_scores),
                "Candidate_Extra_Max": 0.5 * num_extra,
                "Candidate_Extra_Percentage": round((sum(candidate_extra_scores) / (0.5 * num_extra)) * 100, 2) if num_extra > 0 else 0
            })

        blocks.append(("Interactive Training Totals", pd.DataFrame([totals_dict])))


    # ---------------------------------
    # CANDIDATE'S UNDERSTANDING REPORT
    # ---------------------------------
    if "Candidate_Self_Understanding" in data:
        section_topics = data["Candidate_Self_Understanding"]
        blocks.append(("Candidate_Self_Understanding", pd.DataFrame(section_topics)))

        if extra_topics:
            blocks.append(("Extra_Topics", pd.DataFrame(extra_topics)))

        num_topics = len(section_topics)
        scores = [t.get("Topic_Score", 0) or 0 for t in section_topics]
        subtotal = sum(scores)
        max_possible = 2 * num_topics
        percentage = round((subtotal / max_possible) * 100, 2) if max_possible > 0 else 0.0

        extra_topics = data.get("Extra_Topics", [])
        extra_scores = [t.get("Topic_Score", 0) or 0 for t in extra_topics]
        extra_total = sum(extra_scores)
        extra_max = len(extra_topics) * 0.5 if extra_topics else 0
        extra_pct = round((extra_total / extra_max) * 100, 2) if extra_max > 0 else 0.0

        totals_dict = {
            "Subtotal_Score": subtotal,
            "Max_Possible_Score": max_possible,
            "Candidate_Understanding_Percentage": percentage,
            "Extra_Topics_Impact_Score": extra_total,
            "Extra_Topics_Max_Possible_Score": extra_max,
            "Extra_Topics_Percentage": extra_pct
        }
        blocks.append(("Candidate Understanding Totals", pd.DataFrame([totals_dict])))

    # ------------------------------
    # MANAGER COMPLIANCE REPORT
    # ------------------------------
    if "Manager_Compliance_Assessment" in data:
        section_topics = data["Manager_Compliance_Assessment"]
        blocks.append(("Manager_Compliance_Assessment", pd.DataFrame(section_topics)))

        if extra_topics:
            blocks.append(("Extra_Topics", pd.DataFrame(extra_topics)))

        num_topics = len(section_topics)
        scores = [t.get("Topic_Score", 0) or 0 for t in section_topics]
        subtotal = sum(scores)
        max_possible = num_topics * 1
        percentage = round((subtotal / max_possible) * 100, 2) if max_possible > 0 else 0.0

        extra_topics = data.get("Extra_Topics", [])
        extra_scores = [t.get("Topic_Score", 0) or 0 for t in extra_topics]
        extra_total = sum(extra_scores)
        extra_max = len(extra_topics) * 0.5 if extra_topics else 0
        extra_pct = round((extra_total / extra_max) * 100, 2) if extra_max > 0 else 0.0

        totals_dict = {
            "Subtotal_Score": subtotal,
            "Max_Possible_Score": max_possible,
            "Manager_Compliance_Percentage": percentage,
            "Extra_Topics_Impact_Score": extra_total,
            "Extra_Topics_Max_Possible_Score": extra_max,
            "Extra_Topics_Percentage": extra_pct
        }
        blocks.append(("Manager Compliance Totals", pd.DataFrame([totals_dict])))

    # ------------------------------
    # INTERVIEW REPORT
    # ------------------------------
    elif "Interview_Questionair_Responses" in data:
        blocks.append(("Interview_Questionair_Responses", pd.DataFrame(data.get("Interview_Questionair_Responses", []))))
        blocks.append(("Extra Questions", pd.DataFrame(data.get("extra_questions", []))))
        blocks.append(("Interviewer Feedback", pd.DataFrame([data.get("interviewer_feedback", {})])))

        candidate_feedback = data.get("candidate_feedback", {})
        personality = candidate_feedback.get("personality_assessment", {})
        flat_personality = {}
        for category, traits in personality.items():
            for trait, detail in traits.items():
                flat_personality[f"{category}_{trait}_value"] = detail.get("value")
                flat_personality[f"{category}_{trait}_reason"] = detail.get("reason")
        candidate_df = pd.DataFrame([{**candidate_feedback, **flat_personality}])
        candidate_df.drop(columns=["personality_assessment"], errors="ignore", inplace=True)
        blocks.append(("Candidate Feedback", candidate_df))

        predefined = data.get("interview_coverage", {}).get("predefined_questions", {})
        extra = data.get("interview_coverage", {}).get("extra_questions", {})

        blocks.append(("Predefined Totals", pd.DataFrame([{
            "Total_predefined_question": predefined.get("Total_predefined_question"),
            "Questions_asked_by_recruiter_from_pre_defined": predefined.get("Questions_asked_by_recruiter_from_pre_defined"),
            "Recruiter_Percentage": predefined.get("Recruiter_Percentage"),
            "Answer_given_by_candidate_against_recruiter_asked_questions": predefined.get("Answer_given_by_candidate_against_recruiter_asked_questions"),
            "Candidate_Percentage": predefined.get("Candidate_Percentage"),
        }])))

        blocks.append(("Extra Questions Totals", pd.DataFrame([{
            "Total_extra_questions": extra.get("Total_extra_questions"),
            "Helpful_extra_questions": extra.get("Helpful_extra_questions"),
            "Neutral_extra_questions": extra.get("Neutral_extra_questions"),
            "Unhelpful_extra_questions": extra.get("Unhelpful_extra_questions"),
            "Candidate_answered_extra_questions": extra.get("Candidate_answered_extra_questions"),
        }])))

        blocks.append(("Extra Percentages", pd.DataFrame([{
            "Helpful_extra_percentage": extra.get("Recruiter_extra_percentages", {}).get("Helpful_extra_percentage"),
            "Neutral_extra_percentage": extra.get("Recruiter_extra_percentages", {}).get("Neutral_extra_percentage"),
            "Unhelpful_extra_percentage": extra.get("Recruiter_extra_percentages", {}).get("Unhelpful_extra_percentage"),
            "Overall_recruiter_extra_percentage": extra.get("Recruiter_extra_percentages", {}).get("Overall_recruiter_extra_percentage"),
            "Candidate_extra_percentage": extra.get("Candidate_extra_percentage"),
        }])))

    # ------------------------------
    # Write to Excel
    # ------------------------------
    with pd.ExcelWriter(output_file, engine="xlsxwriter") as writer:
        workbook = writer.book
        worksheet = workbook.add_worksheet("Report")
        writer.sheets["Report"] = worksheet

        section_format = workbook.add_format({
            "bold": True, "font_color": "white", "bg_color": "#4F81BD",
            "align": "center", "valign": "vcenter"
        })
        header_format = workbook.add_format({
            "bold": True, "bg_color": "#D9E1F2",
            "align": "center", "valign": "vcenter"
        })

        row_cursor = 0
        for section_title, df in blocks:
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame(df if isinstance(df, (list, dict)) else [])

            worksheet.write(row_cursor, 0, section_title, section_format)
            row_cursor += 1

            if not df.empty:
                for col_num, col_name in enumerate(df.columns):
                    worksheet.write(row_cursor, col_num, col_name, header_format)
                for r in range(len(df)):
                    for c in range(len(df.columns)):
                        worksheet.write(row_cursor + 1 + r, c, df.iat[r, c])
                for i, col in enumerate(df.columns):
                    max_len = max(df[col].astype(str).map(len).max(), len(col)) + 2
                    worksheet.set_column(i, i, max_len)
                row_cursor += len(df) + 1
            else:
                worksheet.write(row_cursor, 0, "(No data)")
                row_cursor += 1

            row_cursor += 1  # Space between blocks

    print(f"✅ Report exported to {output_file} (all headers in blue)")
