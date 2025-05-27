from flask import Blueprint, render_template, request, jsonify, current_app
import json

health_check_bp = Blueprint('health_check', __name__)

@health_check_bp.route('/health_check', methods=['GET', 'POST'])
def health_check():
    """Interactive health check page with forms to test different components"""
    
    if request.method == 'GET':
        # Show the form page
        return render_template('health_check.html')
    
    # Handle POST request - run the tests
    results = {}
    errors = {}
    
    # Get form data
    test_text = request.form.get('test_text', '').strip()
    threshold = float(request.form.get('threshold', 0.3))
    text_query = request.form.get('text_query', '').strip()
    top_k = int(request.form.get('top_k', 5))
    image_url = request.form.get('image_url', '').strip()
    
    # Import the functions we need (adjust imports based on your app structure)
    from index import keyword_check, handle_text, handle_image, get_db
    
    # Test 1: Keyword Check (if text provided)
    if test_text:
        try:
            keyword_results = keyword_check(test_text, threshold=threshold)
            results['keyword_check'] = {
                'input': test_text,
                'threshold': threshold,
                'words': keyword_results
            }
        except Exception as e:
            errors['keyword_check'] = str(e)
    
    # Test 2: Text Lookup (if query provided)
    if text_query:
        try:
            with current_app.test_request_context(json={'query': text_query, 'top_k': top_k}):
                text_response = handle_text()
                results['text_lookup'] = {
                    'query': text_query,
                    'top_k': top_k,
                    'results': text_response.get_json()
                }
        except Exception as e:
            errors['text_lookup'] = str(e)
    
    # Test 3: Image Lookup (if image URL provided)
    if image_url:
        try:
            with current_app.test_request_context(json={'image': image_url}):
                image_response = handle_image()
                results['image_lookup'] = {
                    'query_image': image_url,
                    'results': image_response.get_json()
                }
        except Exception as e:
            errors['image_lookup'] = str(e)
    
    # Test 4: Database connectivity check (always run)
    try:
        db = get_db()
        
        # Check text entries
        text_cursor = db.execute("SELECT COUNT(*) as count FROM text_entries")
        text_count = text_cursor.fetchone()['count']
        
        # Check image entries
        image_cursor = db.execute("SELECT COUNT(*) as count FROM image_entries")
        image_count = image_cursor.fetchone()['count']
        
        results['database_stats'] = {
            'text_entries': text_count,
            'image_entries': image_count
        }
    except Exception as e:
        errors['database'] = str(e)
    
    # Return JSON response for AJAX or render template for form submission
    if request.headers.get('Content-Type') == 'application/json' or request.args.get('format') == 'json':
        return jsonify({
            'results': results,
            'errors': errors,
            'success': len(errors) == 0
        })
    
    # Render the template with results
    return render_template('health_check.html', 
                         results=results, 
                         errors=errors,
                         form_data={
                             'test_text': test_text,
                             'threshold': threshold,
                             'text_query': text_query,
                             'top_k': top_k,
                             'image_url': image_url
                         })