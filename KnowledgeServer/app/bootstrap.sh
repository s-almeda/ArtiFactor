
#!/bin/bash
# Check if test mode is requested
if [ "$1" = "test" ]; then
    echo "Starting Flask API in TEST MODE with Gunicorn development server..."
    # Set the FLASK_APP environment variable
    export FLASK_APP=../main/index.py

    # Set Flask environment to development
    export FLASK_ENV=development
    export FLASK_DEBUG=1

    # Start the worker in the background
    echo "Starting job worker in background..."
    rm -f ./jobs/jobs.db
    python3 ./jobs/worker.py &
    WORKER_PID=$!

    # Run Gunicorn with reload option for development
    exec gunicorn -c ./gunicorn_config.py index:app --timeout 300 -w 1 --reload
else
    echo "Starting Flask API with Gunicorn..."
    # Set the FLASK_APP environment variable
    export FLASK_APP=../main/index.py

    # Start the worker in the background
    echo "Starting job worker in background..."
    python3 ./jobs/worker.py &
    WORKER_PID=$!

    # Run Gunicorn with the specified configuration
    exec gunicorn -c ./gunicorn_config.py index:app --timeout 300 -w 1
fi
