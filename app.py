from flask import Flask, render_template, request, jsonify, send_from_directory,make_response,send_file
import threading, queue, os, uuid, json
import pandas as pd
import os
import time
import json
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from dotenv import load_dotenv
from utils.transciptions import (process_audio_with_gemini, generate_report, clean_and_parse_json, split_and_upload, 
                                 export_report_to_single_excel)
import shutil
from openpyxl import load_workbook, Workbook
from openpyxl.styles import Font, PatternFill, Border, Alignment, Protection
import copy

load_dotenv()
os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
app = Flask(__name__)


app.config["JWT_SECRET_KEY"] = os.getenv("JWT_SECRET_KEY")
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRES", 3600))
app.config["JWT_TOKEN_LOCATION"] = ["cookies"]
app.config["JWT_COOKIE_NAME"] = "access_token_cookie"
app.config["JWT_COOKIE_SECURE"] = False       # True if using HTTPS
app.config["JWT_COOKIE_CSRF_PROTECT"] = False # optional for simplicity


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
REPORTS_FOLDER = "reports"

jwt = JWTManager(app)

JOBS_FILE = "jobs.json"
if not os.path.exists(JOBS_FILE):
    with open(JOBS_FILE, "w") as f:
        json.dump([], f)


# Job queue
report_queue = queue.Queue()
jobs_status = {}  # job_id: {"status": "pending|processing|done|error", "file": filename, "job_folder": ...}



@app.route("/all_jobs_status")
@jwt_required()
def all_jobs_status():
    return jsonify(jobs_status)

# Job queue
report_queue = queue.Queue()
jobs_status = {}  # job_id: {status, file, job_folder, report_excel?, error?}





def read_jobs():
    if os.path.exists(JOBS_FILE):
        with open(JOBS_FILE, "r") as f:
            return json.load(f)
    return []

def write_jobs(jobs):
    with open(JOBS_FILE, "w") as f:
        json.dump(jobs, f, indent=4)

def delete_job_folders(job_id):
    report_path = os.path.join(REPORT_FOLDER, f"job_{job_id}")
    upload_path = os.path.join(UPLOAD_FOLDER, f"job_{job_id}")

    for path in [report_path, upload_path]:
        if os.path.exists(path):
            shutil.rmtree(path)  # Recursively delete folder and contents


@app.route("/get_questions", methods=["GET"])
def get_questions():
    questions_dir = "./utils/questionaires"
    if not os.path.exists(questions_dir):
        return jsonify([])

    files = [
        f for f in os.listdir(questions_dir)
        if os.path.isfile(os.path.join(questions_dir, f))
    ]
    return jsonify(files)

@app.route("/delete_job/<int:job_id>", methods=["DELETE"])
def delete_job(job_id):
    jobs = read_jobs()
    updated_jobs = [job for job in jobs if job.get("id") != job_id]

    if len(updated_jobs) == len(jobs):
        return jsonify({"msg": "Job not found"}), 404
    delete_job_folders(job_id)
    write_jobs(updated_jobs)
    return jsonify({"msg": "Job deleted successfully"}), 200



def report_worker():
    while True:
        job_id, filename, job_id_folder,questionaire = report_queue.get()
        try:
            jobs_status[job_id]["status"] = "processing"
            print(f"[Worker] Start job_id={job_id}, file={filename}, folder={job_id_folder}")

            audio_path = os.path.join(UPLOAD_FOLDER, f"job_{job_id_folder}", filename)
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")

            report_job_folder = os.path.join(REPORT_FOLDER, f"job_{job_id_folder}")
            os.makedirs(report_job_folder, exist_ok=True)

            meta_path = os.path.join(report_job_folder, f"{filename}_meta.json")
            transcription_path = os.path.join(report_job_folder, f"{filename}_transcription.json")
            report_json_path = os.path.join(report_job_folder, f"{filename}_interview_report.json")
            report_excel_path = os.path.join(report_job_folder, f"{filename}_interview_report.xlsx")

            # Load or create metadata
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    meta = json.load(f)
            else:
                meta = {"gcs_uri": None, "transcription_done": False, "report_done": False,"questionaire":questionaire}
            if meta.get("questionaire")!=questionaire:
                meta = {"gcs_uri": None, "transcription_done": False, "report_done": False,"questionaire":questionaire}
    
            # --- Step 1: Upload to GCS ---
            if not meta.get("gcs_uri"):
                jobs_status[job_id]["status"] = "Uploading to GCS"
                print(f"[Worker] Uploading {filename} to GCS...")
                gcs_uris = split_and_upload(GCS_BUCKET_NAME, audio_path, f"{GCS_FOLDER_NAME}/{filename}")
                meta["gcs_uri"] = gcs_uris
                with open(meta_path, "w") as f:
                    json.dump(meta, f, indent=4)
                print(f"[Worker] Uploaded: {gcs_uris}")
            else:
                print("[Worker] GCS upload already done.")

            # --- Step 2: Transcription ---
            if not meta.get("transcription_done") or not os.path.exists(transcription_path):
                jobs_status[job_id]["status"] = "Transcribing Audio"
                print(f"[Worker] Transcribing {filename}...")
                all_transcriptions = []
                for uri in meta["gcs_uri"]:
                    raw_transcription = process_audio_with_gemini(PROJECT_ID, LOCATION, uri)
                    parsed = clean_and_parse_json(raw_transcription)
                    if parsed:
                        all_transcriptions.extend(parsed)
                if all_transcriptions:
                    with open(transcription_path, "w", encoding="utf-8") as f:
                        json.dump(all_transcriptions, f, ensure_ascii=False, indent=4)
                    meta["transcription_done"] = True
                    with open(meta_path, "w") as f:
                        json.dump(meta, f, indent=4)
                    print(f"[Worker] Transcription done: {transcription_path}")
                else:
                    jobs_status[job_id]["status"] = "error"
                    jobs_status[job_id]["error"] = "Transcription failed"
                    continue
            else:
                print("[Worker] Transcription already done.")

            # --- Step 3: Report Generation ---
            if not meta.get("report_done") or not os.path.exists(report_excel_path):
                jobs_status[job_id]["status"] = "Generating Report"
                print(f"[Worker] Generating report for {filename}...")
                with open(transcription_path, "r", encoding="utf-8") as f:
                    transcription_json = json.load(f)
                report_text = generate_report(json.dumps(transcription_json),questionaire)
                report_json = clean_and_parse_json(report_text)
                if report_json:
                    with open(report_json_path, "w", encoding="utf-8") as f:
                        json.dump(report_json, f, ensure_ascii=False, indent=4)
                    export_report_to_single_excel(report_json, report_excel_path)
                    meta["report_done"] = True
                    with open(meta_path, "w") as f:
                        json.dump(meta, f, indent=4)
                    print(f"[Worker] Report generated: {report_excel_path}")
                else:
                    jobs_status[job_id]["status"] = "error"
                    jobs_status[job_id]["error"] = "Report generation failed"
                    continue
            else:
                print("[Worker] Report already generated.")

            jobs_status[job_id]["status"] = "done"
            jobs_status[job_id]["report_excel"] = os.path.basename(report_excel_path)
        except Exception as e:
            print(f"[Worker] Error: {e}")
            jobs_status[job_id]["status"] = "error"
            jobs_status[job_id]["error"] = str(e)
        finally:
            report_queue.task_done()


threading.Thread(target=report_worker, daemon=True).start()


@app.route("/generate_report", methods=["POST"])
def generate_report_route():
    data = request.json
    filename = data.get("filename")
    questionaire = data.get("questionnaire")
    job_id_folder = str(data.get("job_id"))

    if not filename or not job_id_folder:
        return jsonify({"error": "Filename or job_id missing"}), 400

    job_upload_folder = os.path.join(UPLOAD_FOLDER, f"job_{job_id_folder}")
    if not os.path.exists(os.path.join(job_upload_folder, filename)):
        return jsonify({"error": "File not found"}), 404

    # If report already exists, return done
    report_job_folder = os.path.join(REPORT_FOLDER, f"job_{job_id_folder}")
    meta_path = os.path.join(report_job_folder, f"{filename}_meta.json")
    if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    meta = json.load(f)

    report_folder = os.path.join(REPORT_FOLDER, f"job_{job_id_folder}")
    os.makedirs(report_folder, exist_ok=True)
    report_excel_path = os.path.join(report_folder, f"{filename}_interview_report.xlsx")
    if os.path.exists(report_excel_path)and meta.get("questionaire") == questionaire:
        job_id = str(uuid.uuid4())
        jobs_status[job_id] = {
            "status": "done",
            "file": filename,
            "job_folder": report_folder,
            "report_excel": os.path.basename(report_excel_path)
        }
        return jsonify({"job_id": job_id, "status": "done", "report_excel": os.path.basename(report_excel_path)})
    
    
    job_upload_chunks_folder = os.path.join(app.config['UPLOAD_FOLDER'], f"job_{data.get("job_id")}", "chunks")
    job_report_folder = os.path.join(app.config['REPORT_FOLDER'], f"job_{data.get("job_id")}")

    # Delete all files in the chunks folder
    if os.path.exists(job_upload_chunks_folder):
        for f in os.listdir(job_upload_chunks_folder):
            file_path = os.path.join(job_upload_chunks_folder, f)
            if os.path.isfile(file_path):
                os.remove(file_path)

    # Delete report files
    files_to_delete = [
        os.path.join(job_report_folder, f"{filename}_meta.json"),
        os.path.join(job_report_folder, f"{filename}transcription.json"),
        os.path.join(job_report_folder, f"{filename}_interview_report.json"),
        os.path.join(job_report_folder, f"{filename}_interview_report.xlsx")
    ]

    for file_path in files_to_delete:
        if os.path.exists(file_path):
            os.remove(file_path)


    # Otherwise enqueue
    job_id = str(uuid.uuid4())
    jobs_status[job_id] = {"status": "pending", "file": filename, "job_folder": report_folder}
    report_queue.put((job_id, filename, job_id_folder,questionaire))
    return jsonify({"job_id": job_id, "status": "queued"})




@app.route('/')
def index():
    return render_template('index.html')




@app.route('/merged_report', methods=['POST'])
@jwt_required()
def merge_reports():
    data = request.get_json()
    job_id = data.get("job_id")
    files = data.get("filenames", [])

    if not job_id or not files:
        return jsonify({"msg": "job_id and files are required"}), 400

    job_report_folder = os.path.join(app.config['REPORT_FOLDER'], f"job_{job_id}")
    if not os.path.exists(job_report_folder):
        return jsonify({"msg": f"Job {job_id} report folder not found"}), 404

    merged_wb = Workbook()
    merged_ws = merged_wb.active
    merged_ws.title = "Merged Report"

    current_row = 1

    for filename in files:
        report_path = os.path.join(job_report_folder, f"{filename}_interview_report.xlsx")
        if not os.path.exists(report_path):
            return jsonify({"msg": f"Report not found for {filename}"}), 404
        
        try:
            wb = load_workbook(report_path)
            ws = wb.active

            # Header for each file section
            merged_ws.cell(row=current_row, column=1, value=f"Report for {filename}")
            merged_ws.cell(row=current_row, column=1).font = Font(bold=True)
            current_row += 1

            # Copy cells with formatting
            for row in ws.iter_rows():
                for col_index, cell in enumerate(row, start=1):
                    new_cell = merged_ws.cell(row=current_row, column=col_index, value=cell.value)
                    
                    if cell.has_style:
                        new_cell.font = copy.copy(cell.font)
                        new_cell.fill = copy.copy(cell.fill)
                        new_cell.border = copy.copy(cell.border)
                        new_cell.alignment = copy.copy(cell.alignment)
                        new_cell.number_format = cell.number_format
                        new_cell.protection = copy.copy(cell.protection)

                current_row += 1
            
            # Blank row between reports
            current_row += 1

        except Exception as e:
            return jsonify({"msg": f"Error reading {report_path}: {str(e)}"}), 500

    merged_path = os.path.join(job_report_folder, f"merged_{job_id}.xlsx")
    merged_wb.save(merged_path)

    return send_file(merged_path, as_attachment=True)

@app.route("/report/<int:job_id>")
@jwt_required()
def report_page(job_id):
    current_user = get_jwt_identity()
    # Ensure the job exists
    with open(JOBS_FILE) as f:
        jobs = json.load(f)
    job = next((j for j in jobs if j["id"] == job_id), None)
    if not job:
        return "Job not found", 404

    # Create job-specific upload folder if not exists
    job_folder = os.path.join(app.config['UPLOAD_FOLDER'], f"job_{job_id}")
    os.makedirs(job_folder, exist_ok=True)

    return render_template("report.html", job_id=job_id, user=current_user)


# Login endpoint
@app.route("/login", methods=["POST"])
def login():
    data = request.json
    username = data.get("username")
    password = data.get("password")
    
    if username == os.getenv('APP_USERNAME') and password == os.getenv('APP_PASSWORD'):
        access_token = create_access_token(identity=username)
        resp = make_response(jsonify({"msg": "Login successful"}))
        resp.set_cookie(
            "access_token_cookie",
            access_token,
            httponly=True,
            samesite="Lax"
        )
        return resp
    
    return jsonify({"msg": "Bad username or password"}), 401


@app.route("/create_job", methods=["POST"])
@jwt_required()
def create_job():
    data = request.json
    job_name = data.get("job_name")
    if not job_name:
        return jsonify({"msg": "Job name required"}), 400

    # Load existing jobs
    with open(JOBS_FILE) as f:
        jobs = json.load(f)

    job_id = len(jobs) + 1
    job = {"id": job_id, "name": job_name}
    jobs.append(job)

    # Save jobs
    with open(JOBS_FILE, "w") as f:
        json.dump(jobs, f, indent=4)

    return jsonify({"msg": "Job created", "job": job})



# Protected route
@app.route("/protected")
@jwt_required()
def protected():
    current_user = get_jwt_identity()
    return jsonify(logged_in_as=current_user)

# Logout
@app.route("/logout", methods=["POST"])
def logout():
    resp = make_response(jsonify({"msg": "Logged out"}))
    resp.delete_cookie("access_token_cookie")
    return resp


@app.route('/upload/<int:job_id>', methods=['POST'])
@jwt_required()
def upload_audio(job_id):
    if 'audio_file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['audio_file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    job_folder = os.path.join(app.config['UPLOAD_FOLDER'], f"job_{job_id}")
    os.makedirs(job_folder, exist_ok=True)
    
    filepath = os.path.join(job_folder, file.filename)
    file.save(filepath)
    return jsonify({"message": "File uploaded successfully"})


@app.route('/list_files/<int:job_id>', methods=['GET'])
@jwt_required()
def list_files(job_id):
    job_folder = os.path.join(app.config['UPLOAD_FOLDER'], f"job_{job_id}")
    if not os.path.exists(job_folder):
        return jsonify([])
    
    files = [
        f for f in os.listdir(job_folder)
        if os.path.isfile(os.path.join(job_folder, f))
    ]
    
    return jsonify(files)


@app.route("/report_status/<job_id>")
def report_status(job_id):
    job = jobs_status.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    # If job status is done but report_excel missing, try to find it
    if job["status"] == "done" and "report_excel" not in job:
        report_folder = job["job_folder"]
        filename = job["file"]
        report_excel_path = os.path.join(report_folder, f"{filename}_interview_report.xlsx")
        if os.path.exists(report_excel_path):
            job["report_excel"] = os.path.basename(report_excel_path)

    return jsonify(job)

@app.route("/download_report/<job_id>/<filename>")
def download_report(job_id, filename):
    job_folder = os.path.join(REPORTS_FOLDER, f"job_{job_id}")
    if not os.path.exists(os.path.join(job_folder, filename)):
        return "File not found", 404
    return send_from_directory(job_folder, filename, as_attachment=True)


@app.route("/delete_file/<int:job_id>/<filename>", methods=["DELETE"])
@jwt_required()
def delete_file(job_id, filename):
    job_upload_folder = os.path.join(app.config['UPLOAD_FOLDER'], f"job_{job_id}")
    job_report_folder = os.path.join(app.config['UPLOAD_FOLDER'], f"job_{job_id}")
    files_to_delete = [
        os.path.join(job_upload_folder, filename),
        os.path.join(job_report_folder, f"{filename}_meta.json"),
        os.path.join(job_report_folder, f"{filename}transcription.json"),
        os.path.join(job_report_folder, f"{filename}_interview_report.json"),
        os.path.join(job_report_folder, f"{filename}_interview_report.xlsx")
    ]

    removed = 0
    for file_path in files_to_delete:
        if os.path.exists(file_path):
            os.remove(file_path)
            removed += 1

    if removed > 0:
        return "", 200
    return "", 404


@app.route("/jobs")
@jwt_required()
def list_jobs():
    current_user = get_jwt_identity()
    with open(JOBS_FILE) as f:
        jobs = json.load(f)

    jobs_with_counts = []
    for job in jobs:
        job_id = job["id"]
        job_folder = os.path.join(app.config['UPLOAD_FOLDER'], f"job_{job_id}")

        # Count total audio files
        total_audios = len([
            f for f in os.listdir(job_folder)
            if f.lower().endswith(".wav")
        ]) if os.path.exists(job_folder) else 0

        # Count total runs in queue or processing for this job_id
        active_runs = sum(
            1 for item in list(report_queue.queue)
            if str(item[2]) == str(job_id)
        ) + sum(
            1 for jid, status in jobs_status.items()
            if str(status.get("job_folder", "")).endswith(f"job_{job_id}")
            and status.get("status") in ["pending", "processing", "Uploading to GCS", "Transcribing Audio", "Generating Report"]
        )

        jobs_with_counts.append({
            "id": job_id,
            "name": job["name"],
            "total_audios": total_audios,
            "active_runs": active_runs,
            "current_user": current_user
        })

    return render_template("jobs.html", jobs=jobs_with_counts)



if __name__ == '__main__':
    os.makedirs(GCS_FOLDER_NAME, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(REPORT_FOLDER, exist_ok=True)
    app.run(debug=True)
