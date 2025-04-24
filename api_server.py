from flask import Flask, jsonify, request, make_response, send_from_directory, send_file
import sqlite3
import pandas as pd
import json
import os
import datetime
from flask_cors import CORS

app = Flask(__name__, 
            static_folder='web/static',
            static_url_path='/static')

CORS(app, resources={r"/*": {"origins": "*"}})

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET,POST,PUT,DELETE,OPTIONS'
    return response

DB_PATH = "fish_trash_data.db"
DETECTION_IMAGES_DIR = "detection_images"

os.makedirs(DETECTION_IMAGES_DIR, exist_ok=True)

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/api/detections', methods=['GET'])
def get_detections():
    try:
        limit = request.args.get('limit', default=100, type=int)
        offset = request.args.get('offset', default=0, type=int)
        start_date = request.args.get('start_date', default=None, type=str)
        end_date = request.args.get('end_date', default=None, type=str)

        conn = get_db_connection()
        cursor = conn.cursor()

        query = "SELECT * FROM detections"
        params = []

        if start_date or end_date:
            query += " WHERE "
            if start_date:
                query += "timestamp >= ?"
                params.append(start_date)
            if start_date and end_date:
                query += " AND "
            if end_date:
                query += "timestamp <= ?"
                params.append(end_date)

        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor.execute(query, params)
        rows = cursor.fetchall()

        detections = []
        for row in rows:
            detection = dict(row)
            if detection['detection_types']:
                try:
                    detection['detection_types'] = json.loads(detection['detection_types'])
                except:
                    detection['detection_types'] = []
            detections.append(detection)

        cursor.execute("SELECT COUNT(*) FROM detections")
        total_count = cursor.fetchone()[0]

        conn.close()

        return jsonify({
            'detections': detections,
            'total': total_count,
            'limit': limit,
            'offset': offset
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/detection/<int:id>', methods=['GET'])
def get_detection(id):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM detections WHERE id = ?", (id,))
        row = cursor.fetchone()

        if not row:
            return jsonify({'error': 'Detection not found'}), 404

        detection = dict(row)
        if detection['detection_types']:
            try:
                detection['detection_types'] = json.loads(detection['detection_types'])
            except:
                detection['detection_types'] = []

        conn.close()

        return jsonify(detection)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    try:
        conn = get_db_connection()

        stats = {}

        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM detections")
        stats['total_records'] = cursor.fetchone()[0]

        cursor.execute("SELECT SUM(trash_count) FROM detections")
        stats['total_trash_count'] = cursor.fetchone()[0] or 0

        cursor.execute("""
            SELECT location_name, COUNT(*) as count, SUM(trash_count) as total_trash
            FROM detections 
            GROUP BY location_name 
            ORDER BY total_trash DESC
            LIMIT 10
        """)
        stats['top_locations'] = [dict(row) for row in cursor.fetchall()]

        cursor.execute("""
            SELECT weather_condition, COUNT(*) as count, AVG(trash_count) as avg_trash
            FROM detections 
            GROUP BY weather_condition
            ORDER BY count DESC
        """)
        stats['weather_stats'] = [dict(row) for row in cursor.fetchall()]

        cursor.execute("""
            SELECT 
                timestamp as date, 
                trash_count,
                1 as detection_count,
                detection_types
            FROM detections 
            ORDER BY timestamp
        """)
        stats['time_series'] = [dict(row) for row in cursor.fetchall()]

        cursor.execute("SELECT detection_types FROM detections")
        rows = cursor.fetchall()

        type_counts = {}
        for row in rows:
            if row['detection_types']:
                try:
                    types = json.loads(row['detection_types'])
                    for t in types:
                        if t in type_counts:
                            type_counts[t] += 1
                        else:
                            type_counts[t] = 1
                except:
                    pass

        stats['trash_types'] = [{'type': k, 'count': v} for k, v in type_counts.items()]
        stats['trash_types'].sort(key=lambda x: x['count'], reverse=True)

        conn.close()

        return jsonify(stats)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/locations', methods=['GET'])
def get_locations():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT 
                location_name, 
                latitude, 
                longitude, 
                COUNT(*) as detection_count,
                SUM(trash_count) as trash_count
            FROM detections 
            GROUP BY location_name, latitude, longitude
        """)

        locations = [dict(row) for row in cursor.fetchall()]
        conn.close()

        return jsonify(locations)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/images/<path:filename>')
def get_image(filename):
    return send_from_directory(DETECTION_IMAGES_DIR, filename)

@app.route('/')
def serve_index():
    return send_from_directory('web/templates', 'index.html')

@app.route('/<path:path>')
def catch_all(path):
    if path == 'favicon.ico':
        if os.path.exists(os.path.join('web/static', path)):
            return send_from_directory('web/static', path)
        else:
            return send_from_directory('web/static', 'favicon.ico', mimetype='image/x-icon')

    if path.endswith('.html'):
        if os.path.exists(os.path.join('web/templates', path)):
            return send_from_directory('web/templates', path)
        return send_from_directory('web/templates', 'index.html')

    if os.path.exists(os.path.join('web/templates', path)):
        return send_from_directory('web/templates', path)
    elif os.path.exists(os.path.join('web/static', path)):
        return send_from_directory('web/static', path)

    return send_from_directory('web/templates', 'index.html')

@app.route('/api/detections', methods=['POST'])
def add_detection():
    try:
        data = request.json

        required_fields = ['timestamp', 'latitude', 'longitude', 'location_name', 
                          'trash_count', 'detection_types']

        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        if isinstance(data['detection_types'], (list, dict)):
            detection_types = json.dumps(data['detection_types'])
        else:
            detection_types = data['detection_types']

        if not data.get('timestamp'):
            data['timestamp'] = datetime.datetime.now().isoformat()

        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute('''
        INSERT INTO detections 
        (timestamp, latitude, longitude, location_name, weather_condition, temperature, trash_count, detection_types, image_path)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            data['timestamp'],
            data['latitude'],
            data['longitude'],
            data['location_name'],
            data.get('weather_condition', ''),
            data.get('temperature', 0.0),
            data['trash_count'],
            detection_types,
            data.get('image_path', '')
        ))

        conn.commit()
        new_id = cursor.lastrowid
        conn.close()

        return jsonify({'id': new_id, 'message': 'Detection added successfully'}), 201

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Fish AI Trash Detection API Server...")
    print(f"Database path: {os.path.abspath(DB_PATH)}")

    port = 5000

    try:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(('0.0.0.0', port))
        s.close()
    except socket.error:
        print(f"Port {port} is in use, trying alternative port 8080")
        port = 8080

    print(f"Web dashboard will be available at: http://127.0.0.1:{port}")
    print("Press Ctrl+C to stop the server")

    app.run(debug=True, host='0.0.0.0', port=port)
