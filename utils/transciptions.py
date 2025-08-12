import os
import re
import json 
import vertexai
import pandas as pd
import streamlit as st
from google.cloud import storage
from langchain_google_vertexai import ChatVertexAI
from vertexai.generative_models import GenerativeModel, Part
from langchain_core.messages import SystemMessage, HumanMessage

def upload_to_gcs(bucket_name: str, source_file_path: str, destination_blob_name: str) -> str:
    """
    Uploads a file to the specified GCS bucket.

    Args:
        bucket_name (str): The name of the GCS bucket.
        source_file_path (str): The path to the local file to upload.
        destination_blob_name (str): The desired name of the file in the GCS bucket.

    Returns:
        str: The GCS URI of the uploaded file (e.g., "gs://bucket-name/file-name").
    """
    if not os.path.exists(source_file_path):
        raise FileNotFoundError(f"The file {source_file_path} was not found.")

    print(f"Uploading '{source_file_path}' to bucket '{bucket_name}'...")
    
    # Initialize the GCS client
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    # Upload the file
    blob.upload_from_filename(source_file_path)

    gcs_uri = f"gs://{bucket_name}/{destination_blob_name}"
    print(f"File uploaded successfully. GCS URI: {gcs_uri}")
    return gcs_uri


def process_audio_with_gemini(project_id, location, gcs_uri):
    """
    Processes an audio file using Gemini 1.5 Pro on Vertex AI with a fixed transcription + diarization prompt.

    Args:
        project_id (str): Your Google Cloud project ID.
        location (str): The region of your project.
        gcs_uri (str): The GCS URI of the audio file to process.

    Returns:
        str: The generated text response from the model.
    """
    print("Initializing Vertex AI...")
    vertexai.init(project=project_id, location=location)

    print("Loading the Gemini 2.5 Pro model...")
    model = GenerativeModel("gemini-2.5-pro")

    prompt_for_transcription = """Transcribe from Bengali to English. Diarize the audio into speakers 
        and assign proper labels to those speakers. 
        Make sure you assign the sentences to the proper speaker in a way that makes logical sense. 
        Provide output in JSON string format.
        Example:
        [
            {
                "speaker": "Interviewer",
                "transcript": "Where is your CV? Let me see the CV."
            },
            {
                "speaker": "Interviewer",
                "transcript": "Ishak Ali?"
            },
            {
                "speaker": "Candidate",
                "transcript": "Yes, sir."
            }
        ]
    """

    # Prepare the audio part from the GCS URI
    audio_file_part = Part.from_uri(
        uri=gcs_uri,
        mime_type="audio/mpeg"  # Change if needed (e.g., audio/wav)
    )

    # Request content includes both the prompt and audio
    request_content = [prompt_for_transcription, audio_file_part]

    print("Sending request to Gemini model... (This may take a moment)")
    try:
        response = model.generate_content(request_content)
        return response.text
    except Exception as e:
        return f"An error occurred: {e}"


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
- Include all 31 main questions from the questionnaire in the report, in the same order, without omission.
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
       - "average_score"`: Average candidate score across all predefined questions.  
       - "response_quality_summary"`: Brief assessment of overall response quality.  
       - "recommendations"`: Suggestions for improvement.  
       - "personality_assessment"`: An object containing the following keys, each with a boolean value and a short justification **based only on evidence from the transcript**:  
         {{
            "positivity": {{ "value": true, "reason": "Candidate remained optimistic even when discussing past challenges." }},
            "honesty": {{ "value": true, "reason": "Candidate openly admitted to lacking experience in one area." }},
            "humility": {{ "value": true, "reason": "Candidate credited team members for successes." }},
            "desire_to_work": {{ "value": true, "reason": "Candidate expressed strong interest in joining the company." }},
            "discipline": {{ "value": true, "reason": "Candidate described consistent work habits." }},
            "job_understanding": {{ "value": true, "reason": "Candidate accurately described the role's key responsibilities." }},
            "right_person_for_job": {{ "value": true, "reason": "Overall responses align with role requirements." }}
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
    # Use a regex to find the content inside a JSON markdown block
    json_match = re.search(r'```json\n(.*?)\n```', raw_text, re.DOTALL)
    
    if json_match:
        raw_json_str = json_match.group(1).strip()
    else:
        # If no markdown block is found, assume the entire text is the JSON
        raw_json_str = raw_text.strip()
    
    # Remove trailing commas that might cause parsing errors
    raw_json_str = re.sub(r',\s*\]', ']', raw_json_str)
    raw_json_str = re.sub(r',\s*\}', '}', raw_json_str)

    try:
        return json.loads(raw_json_str)
    except json.JSONDecodeError as e:
        st.error(f"❌ Failed to parse JSON. Error: {e}")
        st.write("--- Raw Model Output ---")
        st.code(raw_text) # Display the raw output for debugging
        st.write("--- Attempted to parse: ---")
        st.code(raw_json_str)
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

    # Candidate Feedback — flatten personality_assessment
    candidate_feedback = data.get("candidate_feedback", {})
    personality = candidate_feedback.get("personality_assessment", {})
    flat_personality = {}
    for trait, detail in personality.items():
        flat_personality[f"{trait}_value"] = detail.get("value")
        flat_personality[f"{trait}_reason"] = detail.get("reason")
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

