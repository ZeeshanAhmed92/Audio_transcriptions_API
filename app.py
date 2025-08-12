from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import time
import json
from dotenv import load_dotenv
from utils.transciptions import (upload_to_gcs,process_audio_with_gemini,generate_report,
                                 clean_and_parse_json,export_report_to_single_excel)
load_dotenv()
os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
app = Flask(__name__)

# Set base folders
UPLOAD_FOLDER = 'uploads'
REPORT_FOLDER = 'reports'
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")  
LOCATION = "us-central1"        
GCS_BUCKET_NAME = "bengali-transcription-bucket" 
TEMP_DIR = "temp_uploads"
GCS_FOLDER_NAME = "Interview_audios"  

# openai.api_key = os.getenv("OPENAI_API_KEY")
os.getenv("GOOGLE_APPLICATION_CREDENTIALS")



os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['REPORT_FOLDER'] = REPORT_FOLDER


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_audio():
    if 'audio_file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['audio_file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    return jsonify({"message": "File uploaded successfully"})


@app.route('/list_files', methods=['GET'])
def list_files():
    files = os.listdir(app.config['UPLOAD_FOLDER'])
    return jsonify(files)


@app.route('/generate_report', methods=['POST'])
def generate_report_route():
    data = request.json
    filename = data.get("filename")

    if not filename:
        return jsonify({"error": "No file specified"}), 400

    audio_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(audio_path):
        return jsonify({"error": "File not found"}), 404

    # Paths
    meta_path = os.path.join(TEMP_DIR, f"{filename}_meta.json")
    transcription_path = os.path.join(TEMP_DIR, f"{filename}transcription.json")
    report_json_path = os.path.join(TEMP_DIR, f"{filename}_interview_report.json")
    report_excel_path = os.path.join(TEMP_DIR, f"{filename}_interview_report.xlsx")

    # Load or create metadata
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
    else:
        meta = {
            "gcs_uri": None,
            "transcription_done": False,
            "report_done": False
        }

    try:
        # Step 1: Upload to GCS (skip if already done)
        if not meta.get("gcs_uri"):
            print("Uploading to GCS...")
            destination_blob_name = f"{GCS_FOLDER_NAME}/{filename}"
            source_file_path = os.path.join(UPLOAD_FOLDER, filename)
            gcs_uri = upload_to_gcs(
                GCS_BUCKET_NAME,
                source_file_path,
                destination_blob_name
            )
            meta["gcs_uri"] = gcs_uri
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=4)
            print("File uploaded to GCS:", gcs_uri)
        else:
            print("Skipping GCS upload — already done:", meta["gcs_uri"])

        # Step 2: Transcription (skip if already done)
        if not meta.get("transcription_done") or not os.path.exists(transcription_path):
            print("Transcribing audio...")
            raw_transcription = process_audio_with_gemini(
                PROJECT_ID,
                LOCATION,
                meta["gcs_uri"]
            )
            transcription_json = clean_and_parse_json(raw_transcription)
            if transcription_json:
                os.makedirs(os.path.dirname(transcription_path), exist_ok=True)
                with open(transcription_path, "w", encoding="utf-8") as f:
                    json.dump(transcription_json, f, ensure_ascii=False, indent=4)
                meta["transcription_done"] = True
                with open(meta_path, "w") as f:
                    json.dump(meta, f, indent=4)
                print("Transcription complete.")
            else:
                return jsonify({"message": "Transcription failed", "filename": filename}), 500
        else:
            print("Skipping transcription — already done.")

        # Step 3: Report Generation (skip if already done)
        if not meta.get("report_done") or not os.path.exists(report_json_path) or not os.path.exists(report_excel_path):
            print("Generating report...")
            with open(transcription_path, "r", encoding="utf-8") as f:
                transcription_json = json.load(f)

            report_text = generate_report(json.dumps(transcription_json))
            report_json = clean_and_parse_json(report_text)

            if report_json:
                os.makedirs(os.path.dirname(report_json_path), exist_ok=True)
                with open(report_json_path, "w", encoding="utf-8") as f:
                    json.dump(report_json, f, ensure_ascii=False, indent=4)

                export_report_to_single_excel(report_json, report_excel_path)

                meta["report_done"] = True
                with open(meta_path, "w") as f:
                    json.dump(meta, f, indent=4)

                print("Report generated.")
            else:
                return jsonify({"message": "Report generation failed", "filename": filename}), 500
        else:
            print("Skipping report generation — already done.")

        return jsonify({
            "message": "Report generated successfully",
            "report_filename": os.path.basename(report_json_path),
            "report_excel": os.path.basename(report_excel_path),
            "gcs_uri": meta["gcs_uri"]
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"message": "Report generation failed", "error": str(e)}), 500



@app.route('/download_report/<path:filename>', methods=['GET'])
def download_report(filename):
    return send_from_directory(TEMP_DIR, filename, as_attachment=True)

@app.route("/delete_file/<filename>", methods=["DELETE"])
def delete_file(filename):
    files = [
        os.path.join(TEMP_DIR, f"{filename}_meta.json"),
        os.path.join(TEMP_DIR, f"{filename}transcription.json"),
        os.path.join(TEMP_DIR, f"{filename}_interview_report.json"),
        os.path.join(TEMP_DIR, f"{filename}_interview_report.xlsx"),
        os.path.join(UPLOAD_FOLDER, filename)
        ]
    removed = 0
    for file in files:
        if os.path.exists(file):
            os.remove(file)
            removed+=1
        else:
            print({file}+" doesnt exits")
    if removed>0:
        return "", 200
    return "", 404

if __name__ == '__main__':
    os.makedirs(GCS_FOLDER_NAME, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(REPORT_FOLDER, exist_ok=True)
    app.run(debug=True)
