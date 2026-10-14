import os
import subprocess
import sys

def start_mlflow_server():
    
    """
    Starts the MLflow tracking server.
    In a production environment, this would be handled by a
    dedicated process manager or Docker, but this script
    allows for easy local testing and CI smoke tests.
    """
    
    print("--- Initializing MLflow Tracking Server ---")

    # Configuration for MLflow (env-overridable for Docker Compose volumes).
    backend_store_uri = os.getenv("MLFLOW_BACKEND_STORE_URI", "sqlite:///mlflow.db")
    default_artifact_root = os.getenv("MLFLOW_ARTIFACT_ROOT", "artifacts")
    host = os.getenv("MLFLOW_HOST", "0.0.0.0")
    port = os.getenv("MLFLOW_PORT", "5000")
    allowed_hosts = os.getenv(
        "MLFLOW_ALLOWED_HOSTS", "localhost,127.0.0.1,mlflow-server"
    )

    os.makedirs(default_artifact_root, exist_ok=True)
    if backend_store_uri.startswith("sqlite:///"):
        db_dir = os.path.dirname(backend_store_uri.replace("sqlite:///", "", 1))
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

    # Command to start the MLflow server
    # --host 0.0.0.0 makes it accessible from outside the container
    # --backend-store-uri defines where the experiment data is saved
    # --default-artifact-root defines where the actual model files go
    cmd = [
        "mlflow", "server",
        "--host", host,
        "--port", port,
        "--backend-store-uri", backend_store_uri,
        "--default-artifact-root", default_artifact_root,
        "--allowed-hosts", allowed_hosts,
    ]

    print(f"Executing: {' '.join(cmd)}")

    try:
        # Start the server as a subprocess; inherit stdout/stderr so container
        # logs are visible (a PIPE here can fill and block the server).
        process = subprocess.Popen(cmd)
        print(f"MLflow server started successfully on port {port}...")
        return process
    except Exception as e:
        print(f"Failed to start MLflow server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    server_process = start_mlflow_server()
    # Keep the script running so the server stays alive
    try:
        server_process.wait()
    except KeyboardInterrupt:
        print("\nStopping MLflow server...")
        server_process.terminate()
