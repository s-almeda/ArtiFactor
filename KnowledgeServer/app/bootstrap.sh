echo "Starting Flask API with Gunicorn..."

# Set the FLASK_APP environment variable
export FLASK_APP=../main/index.py

# Run Gunicorn with the specified configuration
exec gunicorn -c ./gunicorn_config.py index:app --timeout 300 -w 1