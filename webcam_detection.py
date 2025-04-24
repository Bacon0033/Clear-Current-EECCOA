import cv2
import time
import datetime
from ultralytics import YOLO
import numpy as np
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
import requests
import json
import sqlite3
import os
from geopy.geocoders import Nominatim
import pandas as pd
import folium
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import flask
from flask import Flask, render_template, request
import webbrowser
import socket
import re
import platform
import threading
import http.server
import socketserver
from urllib.parse import urlparse, parse_qs

SHOW_MODEL_INFO = True

MODEL_PATH = "mainmodel/train/weights/best.pt"
DB_PATH = "fish_trash_data.db"
DETECTION_IMAGES_DIR = "detection_images"

GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.4
DISTANCE_THRESHOLD = 50
ALLOWED_NAMES = {'battery', 'can', 'cardboard', 'drink carton', 'glass bottle', 'paper', 'plastic bottle', 'plastic bottle cap', 'pop tab', 'Fish'}
TRASH_CATEGORY = "Trash"

WEATHER_API_KEY = "1003863e0a160d71605dcc67d44701db"
WEATHER_API_URL = "https://api.openweathermap.org/data/2.5/weather"

def setup_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        latitude REAL,
        longitude REAL,
        location_name TEXT,
        weather_condition TEXT,
        temperature REAL,
        trash_count INTEGER,
        detection_types TEXT,
        image_path TEXT
    )
    ''')
    
    conn.commit()
    conn.close()

def get_location():
    try:
        last_location = get_last_location()
        
        location_method = simpledialog.askstring(
            "Location Method", 
            "Choose location detection method:\n\n" +
            "1. Browser-based detection (recommended)\n" +
            "2. Manual coordinates entry\n\n" +
            "Enter 1 or 2:", 
            initialvalue="1"
        )
        
        location_data = {"received": False, "lat": 0.0, "lng": 0.0, "accuracy": 1000}
        
        if location_method == "1":
            print("Using browser-based location detector...")
            
            import threading
            import http.server
            import socketserver
            from urllib.parse import urlparse, parse_qs
            
            location_data = {"received": False, "lat": 0.0, "lng": 0.0, "accuracy": 1000}
            
            class LocationHandler(http.server.SimpleHTTPRequestHandler):
                def do_GET(self):
                    nonlocal location_data
                    if self.path.startswith('/location'):
                        parsed_url = urlparse(self.path)
                        params = parse_qs(parsed_url.query)
                        
                        try:
                            if 'lat' in params and 'lng' in params:
                                location_data["lat"] = float(params['lat'][0])
                                location_data["lng"] = float(params['lng'][0])
                                if 'acc' in params:
                                    location_data["accuracy"] = float(params['acc'][0])
                                location_data["received"] = True
                                
                                self.send_response(200)
                                self.send_header('Content-type', 'text/plain')
                                self.send_header('Access-Control-Allow-Origin', '*')
                                self.end_headers()
                                self.wfile.write(b"Location received successfully")
                                print(f"Received location: {location_data['lat']}, {location_data['lng']} (accuracy: {location_data['accuracy']}m)")
                                return
                        except Exception as e:
                            print(f"Error parsing location data: {e}")
                    elif self.path == '/' or self.path == '/index.html' or self.path == '/get_gps_location.html':
                        location_html = """<!DOCTYPE html>
<html>
<head>
    <title>Location Detection</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="Content-Security-Policy" content="default-src 'self' 'unsafe-inline' data: gap: https://ssl.gstatic.com https://*.openstreetmap.org https://unpkg.com https://nominatim.openstreetmap.org https://ipinfo.io http://localhost:* http://127.0.0.1:*">
    <style>
        body { 
            font-family: Arial, sans-serif; 
            margin: 20px;
            background-color: #f5f5f5;
        }
        .container { 
            max-width: 600px; 
            margin: 0 auto; 
            text-align: center; 
            background-color: white;
            padding: 25px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .status { 
            margin: 20px 0; 
            padding: 10px; 
            border-radius: 5px; 
        }
        .success { background-color: #d4edda; color: #155724; }
        .error { background-color: #f8d7da; color: #721c24; }
        .warning { background-color: #fff3cd; color: #856404; }
        .waiting { background-color: #cce5ff; color: #004085; }
        button { 
            padding: 10px 20px; 
            background: #4CAF50; 
            color: white; 
            border: none; 
            cursor: pointer;
            border-radius: 5px;
            margin: 5px;
        }
        button.secondary {
            background: #6c757d;
        }
        .progress-container {
            width: 100%;
            background-color: #f1f1f1;
            border-radius: 5px;
            margin: 10px 0;
        }
        .progress-bar {
            width: 0%;
            height: 30px;
            background-color: #4CAF50;
            border-radius: 5px;
            text-align: center;
            line-height: 30px;
            color: white;
        }
        .alternate-methods {
            margin-top: 20px;
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 5px;
            text-align: left;
        }
        #map {
            height: 350px;
            margin: 15px 0;
            border-radius: 5px;
            display: none;
        }
        .input-group {
            margin: 10px 0;
        }
        .input-group label {
            display: block;
            margin-bottom: 5px;
            text-align: left;
        }
        .input-group input {
            width: 100%;
            padding: 8px;
            border: 1px solid #ced4da;
            border-radius: 4px;
        }
        h4 {
            margin-top: 20px;
            margin-bottom: 10px;
        }
        #search-container {
            margin-bottom: 10px;
        }
        #search-input {
            width: 70%;
            padding: 8px;
            border: 1px solid #ced4da;
            border-radius: 4px 0 0 4px;
        }
        #search-btn {
            padding: 8px 15px;
            background: #0d6efd;
            color: white;
            border: none;
            border-radius: 0 4px 4px 0;
            cursor: pointer;
        }
        .map-instructions {
            background-color: #e9ecef;
            padding: 10px;
            border-radius: 4px;
            margin-bottom: 10px;
            font-size: 14px;
        }
        .method-button {
            display: block;
            width: 100%;
            margin: 10px 0;
            padding: 15px;
            font-size: 16px;
            text-align: left;
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 5px;
            cursor: pointer;
            transition: all 0.2s;
        }
        .method-button:hover {
            background: #e9ecef;
        }
        .method-button i {
            margin-right: 10px;
        }
    </style>
    <!-- Include Leaflet.js for map -->
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>
</head>
<body>
    <div class="container">
        <h1>Location Detection</h1>
        <p>Please select a method to provide your location:</p>
        
        <div id="method-selection">
            <button class="method-button" onclick="getLocation()">
                <i>📍</i> Use my current location (GPS)
            </button>
            <button class="method-button" onclick="showMapSelection()">
                <i>🗺️</i> Select a location on map
            </button>
            <button class="method-button" onclick="useIPLocation()">
                <i>🌐</i> Use approximate location (IP-based)
            </button>
        </div>
        
        <div id="status" class="status waiting" style="display:none">Waiting for location...</div>
        
        <div class="progress-container" style="display:none">
            <div id="progress" class="progress-bar">0%</div>
        </div>
        
        <div id="location-info"></div>
        <button id="retry-btn" style="display:none" onclick="getLocation()">Try Again</button>
        <button id="confirm-btn" style="display:none" onclick="confirmLocation()">Use This Location</button>
        
        <!-- Map selection -->
        <div id="map-container" style="display:none">
            <div class="map-instructions">
                <strong>Instructions:</strong> Click anywhere on the map to select a location. You can use the search box to find places.
            </div>
            <div id="search-container">
                <input type="text" id="search-input" placeholder="Search for a place...">
                <button id="search-btn" onclick="searchLocation()">Search</button>
            </div>
            <div id="map"></div>
            <div class="input-group">
                <label for="lat-input">Latitude (-90 to 90):</label>
                <input type="text" id="lat-input" placeholder="Enter latitude" value="0.0">
            </div>
            <div class="input-group">
                <label for="lng-input">Longitude (-180 to 180):</label>
                <input type="text" id="lng-input" placeholder="Enter longitude" value="0.0">
            </div>
            <button onclick="useManualCoordinates()">Use These Coordinates</button>
        </div>
    </div>
    
    <script>
        var lat = 0, lng = 0;
        var locationAccuracy = 1000;
        var progress = 0;
        var progressInterval;
        var map, marker;
        var highAccuracyOptions = {
            enableHighAccuracy: true,
            timeout: 15000,
            maximumAge: 0
        };
        var serverAttempts = 0;
        var maxServerAttempts = 10;
        
        function sendLocationToServer(latitude, longitude, accuracy) {
            serverAttempts++;
            
            const urls = [
                `http://localhost:""" + str(port) + """/location?lat=${latitude}&lng=${longitude}&acc=${accuracy}`,
                `http://127.0.0.1:""" + str(port) + """/location?lat=${latitude}&lng=${longitude}&acc=${accuracy}`
            ];
            
            Promise.all(urls.map(url => 
                fetch(url, { mode: 'no-cors' })
                    .catch(err => console.log(`Attempt with ${url} failed: ${err}`))
            ))
            .then(() => {
                console.log("Location data sent via HTTP");
                document.getElementById('status').innerHTML = '<span class="success">✓ Location saved successfully!</span>';
                document.getElementById('status').style.display = 'block';
                setTimeout(() => {
                    if (window.opener) {
                        try {
                            window.opener.postMessage({
                                type: 'location_data',
                                latitude: latitude,
                                longitude: longitude,
                                accuracy: accuracy
                            }, '*');
                        } catch (e) {
                            console.log("Error sending message to opener:", e);
                        }
                    }
                    
                    window.close();
                }, 2000);
            })
            .catch(error => {
                console.error("Error sending location:", error);
                
                if (serverAttempts < maxServerAttempts) {
                    document.getElementById('status').innerHTML = 
                        `<span class="waiting">Sending location data... (Attempt ${serverAttempts}/${maxServerAttempts})</span>`;
                    document.getElementById('status').style.display = 'block';
                    
                    setTimeout(() => {
                        sendLocationToServer(latitude, longitude, accuracy);
                    }, 1000);
                } else {
                    document.getElementById('status').className = 'status warning';
                    document.getElementById('status').innerHTML = 
                        'Unable to connect to the application. ' +
                        'Your location information is saved in your browser. ' +
                        'You can close this window now.';
                    document.getElementById('status').style.display = 'block';
                        
                    localStorage.setItem('last_location_lat', latitude.toString());
                    localStorage.setItem('last_location_lng', longitude.toString());
                    localStorage.setItem('last_location_acc', accuracy.toString());
                    localStorage.setItem('last_location_time', new Date().toISOString());
                }
            });
            
            localStorage.setItem('selected_lat', latitude.toString());
            localStorage.setItem('selected_lng', longitude.toString());
            localStorage.setItem('location_accuracy', accuracy.toString());
        }
        
        function updateProgress() {
            progress += 1;
            if (progress <= 100) {
                document.getElementById('progress').style.width = progress + '%';
                document.getElementById('progress').innerHTML = progress + '%';
            } else {
                clearInterval(progressInterval);
            }
        }
        
        function confirmLocation() {
            if (lat !== 0 || lng !== 0) {
                sendLocationToServer(lat, lng, locationAccuracy);
            } else {
                alert("No location selected. Please select a location first.");
            }
        }
        
        function getLocation() {
            // Hide other containers
            document.getElementById('method-selection').style.display = 'none';
            document.getElementById('map-container').style.display = 'none';
            
            // Show progress
            document.getElementById('status').className = 'status waiting';
            document.getElementById('status').innerHTML = 'Waiting for location...';
            document.getElementById('status').style.display = 'block';
            document.querySelector('.progress-container').style.display = 'block';
            document.getElementById('retry-btn').style.display = 'none';
            document.getElementById('location-info').innerHTML = '';
            
            progress = 0;
            document.getElementById('progress').style.width = '0%';
            document.getElementById('progress').innerHTML = '0%';
            
            progressInterval = setInterval(updateProgress, 150);
            
            if (navigator.permissions && navigator.permissions.query) {
                navigator.permissions.query({ name: 'geolocation' })
                    .then(function(permissionStatus) {
                        if (permissionStatus.state === 'denied') {
                            handleLocationError({ code: 1, message: "Location permission is denied in your browser settings." });
                            return;
                        }
                        
                        permissionStatus.onchange = function() {
                            if (this.state === 'granted') {
                                getLocation();
                            } else if (this.state === 'denied') {
                                handleLocationError({ code: 1, message: "Location permission was denied." });
                            }
                        };
                        
                        attemptGeolocation();
                    })
                    .catch(function(error) {
                        console.error("Permission check error:", error);
                        attemptGeolocation();
                    });
            } else {
                attemptGeolocation();
            }
        }
        
        function attemptGeolocation() {
            if (navigator.geolocation) {
                if (window.isSecureContext === false) {
                    console.warn("Not in a secure context, geolocation might be blocked");
                    document.getElementById('status').innerHTML = 
                        'Warning: This page is not in a secure context (HTTPS). ' +
                        'Some browsers like Chrome require HTTPS for geolocation to work. ' +
                        'Attempting to get location anyway...';
                }
                
                navigator.geolocation.getCurrentPosition(
                    successCallback, 
                    errorHighAccuracy, 
                    highAccuracyOptions
                );
            } else {
                handleNoGeolocation("Geolocation is not supported by this browser.");
            }
        }
        
        function errorHighAccuracy(error) {
            if (error.code === error.TIMEOUT) {
                console.log("High accuracy location timed out, trying with lower accuracy");
                navigator.geolocation.getCurrentPosition(
                    successCallback, 
                    errorLowAccuracy, 
                    { enableHighAccuracy: false, timeout: 20000, maximumAge: 0 }
                );
                return;
            }
            
            handleLocationError(error);
        }
        
        function errorLowAccuracy(error) {
            handleLocationError(error);
        }
        
        function handleLocationError(error) {
            clearInterval(progressInterval);
            document.getElementById('progress').style.width = '100%';
            document.getElementById('progress').innerHTML = 'Failed';
            
            let errorMessage = '';
            switch(error.code) {
                case 1:
                    errorMessage = "Location access was denied. Please check your browser settings and ensure location services are enabled.";
                    break;
                case 2:
                    errorMessage = "Location information is unavailable on this device or browser.";
                    break;
                case 3:
                    errorMessage = "The request to get location timed out. Please try again or use an alternative method.";
                    break;
                default:
                    errorMessage = "An unknown error occurred: " + (error.message || '');
                    break;
            }
            
            document.getElementById('status').className = 'status error';
            document.getElementById('status').innerHTML = 'Error: ' + errorMessage;
            document.getElementById('retry-btn').style.display = 'inline-block';
            
            document.getElementById('method-selection').style.display = 'block';
        }
        
        function handleNoGeolocation(errorMessage) {
            clearInterval(progressInterval);
            document.getElementById('progress').style.width = '100%';
            document.getElementById('progress').innerHTML = 'Failed';
            
            document.getElementById('status').className = 'status error';
            document.getElementById('status').innerHTML = errorMessage;
            document.getElementById('retry-btn').style.display = 'inline-block';
            
            document.getElementById('method-selection').style.display = 'block';
        }
        
        function successCallback(position) {
            clearInterval(progressInterval);
            document.getElementById('progress').style.width = '100%';
            document.getElementById('progress').innerHTML = 'Done!';
            
            lat = position.coords.latitude;
            lng = position.coords.longitude;
            locationAccuracy = position.coords.accuracy;
            
            console.log(`Location obtained: ${lat}, ${lng} (accuracy: ${locationAccuracy}m)`);
            
            let accuracyText = '';
            if (locationAccuracy < 100) {
                accuracyText = 'Good (±' + Math.round(locationAccuracy) + 'm)';
                document.getElementById('status').className = 'status success';
            } else if (locationAccuracy < 500) {
                accuracyText = 'Moderate (±' + Math.round(locationAccuracy) + 'm)';
                document.getElementById('status').className = 'status waiting';
            } else {
                accuracyText = 'Poor (±' + Math.round(locationAccuracy) + 'm)';
                document.getElementById('status').className = 'status warning';
            }
            
            document.getElementById('status').innerHTML = 'Location found! Accuracy: ' + accuracyText;
            document.getElementById('location-info').innerHTML = 
                '<h3>Your Location</h3>' +
                '<p>Latitude: ' + lat.toFixed(6) + '</p>' +
                '<p>Longitude: ' + lng.toFixed(6) + '</p>' +
                '<p><em>No download needed - click "Use This Location" to continue</em></p>';
            
            document.getElementById('confirm-btn').style.display = 'inline-block';
            document.getElementById('retry-btn').style.display = 'inline-block';
            
            showMapWithLocation(lat, lng);
            
            localStorage.setItem('selected_lat', lat.toString());
            localStorage.setItem('selected_lng', lng.toString());
            localStorage.setItem('location_accuracy', locationAccuracy.toString());
        }
        
        function showMapSelection() {
            // Hide other elements
            document.getElementById('method-selection').style.display = 'none';
            document.getElementById('status').style.display = 'none';
            document.querySelector('.progress-container').style.display = 'none';
            
            // Show map container
            document.getElementById('map-container').style.display = 'block';
            
            // Initialize map if not already done
            if (!map) {
                // Try to get a better starting point from localStorage or default to center of world map
                let startLat = 0, startLng = 0, zoom = 2;
                
                // Try to use stored location if available
                const storedLat = localStorage.getItem('selected_lat');
                const storedLng = localStorage.getItem('selected_lng');
                if (storedLat && storedLng) {
                    startLat = parseFloat(storedLat);
                    startLng = parseFloat(storedLng);
                    zoom = 10;
                }
                
                // Initialize map
                map = L.map('map').setView([startLat, startLng], zoom);
                L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                }).addTo(map);
                
                // If we have a stored location, add a marker
                if (storedLat && storedLng) {
                    marker = L.marker([startLat, startLng]).addTo(map);
                    lat = startLat;
                    lng = startLng;
                    
                    // Update form fields
                    document.getElementById('lat-input').value = lat.toFixed(6);
                    document.getElementById('lng-input').value = lng.toFixed(6);
                    
                    document.getElementById('location-info').innerHTML = 
                        '<h3>Selected Location</h3>' +
                        '<p>Latitude: ' + lat.toFixed(6) + '</p>' +
                        '<p>Longitude: ' + lng.toFixed(6) + '</p>' +
                        '<p><em>Click "Use This Location" when ready</em></p>';
                    
                    document.getElementById('confirm-btn').style.display = 'inline-block';
                }
                
                // Add click handler for map
                map.on('click', function(e) {
                    lat = e.latlng.lat;
                    lng = e.latlng.lng;
                    
                    // Update marker
                    if (marker) {
                        marker.setLatLng(e.latlng);
                    } else {
                        marker = L.marker(e.latlng).addTo(map);
                    }
                    
                    // Update form fields
                    document.getElementById('lat-input').value = lat.toFixed(6);
                    document.getElementById('lng-input').value = lng.toFixed(6);
                    
                    // Update info and show confirm button
                    document.getElementById('location-info').innerHTML = 
                        '<h3>Selected Location</h3>' +
                        '<p>Latitude: ' + lat.toFixed(6) + '</p>' +
                        '<p>Longitude: ' + lng.toFixed(6) + '</p>' +
                        '<p><em>Click "Use This Location" when ready</em></p>';
                    
                    document.getElementById('confirm-btn').style.display = 'inline-block';
                    
                    // Default accuracy for map selection
                    locationAccuracy = 10000;
                });
                
                // Add handler for search input (Enter key)
                document.getElementById('search-input').addEventListener('keypress', function(e) {
                    if (e.key === 'Enter') {
                        searchLocation();
                    }
                });
            }
            
            // Make sure map is visible and update its size
            document.getElementById('map').style.display = 'block';
            setTimeout(function() {
                map.invalidateSize();
            }, 100);
        }
        
        function searchLocation() {
            const searchTerm = document.getElementById('search-input').value;
            if (!searchTerm) return;
            
            // Show status while searching
            document.getElementById('status').className = 'status waiting';
            document.getElementById('status').innerHTML = 'Searching for location...';
            document.getElementById('status').style.display = 'block';
            
            // Use Nominatim search API
            fetch('https://nominatim.openstreetmap.org/search?format=json&q=' + encodeURIComponent(searchTerm))
                .then(response => response.json())
                .then(data => {
                    document.getElementById('status').style.display = 'none';
                    
                    if (data && data.length > 0) {
                        lat = parseFloat(data[0].lat);
                        lng = parseFloat(data[0].lon);
                        
                        // Update map view
                        map.setView([lat, lng], 13);
                        
                        // Update marker
                        if (marker) {
                            marker.setLatLng([lat, lng]);
                        } else {
                            marker = L.marker([lat, lng]).addTo(map);
                        }
                        
                        // Update form fields
                        document.getElementById('lat-input').value = lat.toFixed(6);
                        document.getElementById('lng-input').value = lng.toFixed(6);
                        
                        // Update info
                        document.getElementById('location-info').innerHTML = 
                            '<h3>Selected Location</h3>' +
                            '<p>Search result: ' + data[0].display_name + '</p>' +
                            '<p>Latitude: ' + lat.toFixed(6) + '</p>' +
                            '<p>Longitude: ' + lng.toFixed(6) + '</p>' +
                            '<p><em>Click "Use This Location" when ready</em></p>';
                        
                        document.getElementById('confirm-btn').style.display = 'inline-block';
                        
                        // Default accuracy for searched locations
                        locationAccuracy = 5000;
                    } else {
                        alert('Location not found. Please try a different search term.');
                    }
                })
                .catch(error => {
                    document.getElementById('status').style.display = 'none';
                    console.error('Error searching location:', error);
                    alert('Error searching for location. Please try again.');
                });
        }
        
        function useManualCoordinates() {
            try {
                const latInput = parseFloat(document.getElementById('lat-input').value);
                const lngInput = parseFloat(document.getElementById('lng-input').value);
                
                if (isNaN(latInput) || isNaN(lngInput)) {
                    alert('Please enter valid numeric coordinates');
                    return;
                }
                
                if (latInput < -90 || latInput > 90 || lngInput < -180 || lngInput > 180) {
                    alert('Coordinates out of range. Latitude must be between -90 and 90, and longitude between -180 and 180.');
                    return;
                }
                
                lat = latInput;
                lng = lngInput;
                
                // Update map
                map.setView([lat, lng], 13);
                
                if (marker) {
                    marker.setLatLng([lat, lng]);
                } else {
                    marker = L.marker([lat, lng]).addTo(map);
                }
                
                // Update info
                document.getElementById('location-info').innerHTML = 
                    '<h3>Selected Location</h3>' +
                    '<p>Latitude: ' + lat.toFixed(6) + '</p>' +
                    '<p>Longitude: ' + lng.toFixed(6) + '</p>' +
                    '<p><em>Click "Use This Location" when ready</em></p>';
                
                document.getElementById('confirm-btn').style.display = 'inline-block';
                
                // Default accuracy for manual coordinates
                locationAccuracy = 10000;
            } catch (e) {
                alert('Error processing coordinates: ' + e.message);
            }
        }
        
        function useIPLocation() {
            // Hide other elements
            document.getElementById('method-selection').style.display = 'none';
            document.getElementById('map-container').style.display = 'none';
            
            // Show status
            document.getElementById('status').className = 'status waiting';
            document.getElementById('status').innerHTML = 'Fetching approximate location...';
            document.getElementById('status').style.display = 'block';
            
            // Try IP-based geolocation using ipinfo.io
            fetch('https://ipinfo.io/json')
                .then(response => response.json())
                .then(data => {
                    if (data && data.loc) {
                        const coords = data.loc.split(',');
                        lat = parseFloat(coords[0]);
                        lng = parseFloat(coords[1]);
                        locationAccuracy = 5000; // Approximate accuracy for IP (5km)
                        
                        document.getElementById('status').className = 'status warning';
                        document.getElementById('status').innerHTML = 'Location found using IP address (low accuracy: ±5km)';
                        
                        document.getElementById('location-info').innerHTML = 
                            '<h3>Approximate Location</h3>' +
                            '<p>Latitude: ' + lat.toFixed(6) + '</p>' +
                            '<p>Longitude: ' + lng.toFixed(6) + '</p>' +
                            '<p>Location: ' + (data.city || '') + ', ' + (data.region || '') + ', ' + (data.country || '') + '</p>' +
                            '<p><em>Click "Use This Location" to continue</em></p>';
                        
                        document.getElementById('confirm-btn').style.display = 'inline-block';
                        
                        showMapWithLocation(lat, lng);
                        
                        // Store in localStorage as backup
                        localStorage.setItem('selected_lat', lat.toString());
                        localStorage.setItem('selected_lng', lng.toString());
                        localStorage.setItem('location_accuracy', locationAccuracy.toString());
                    } else {
                        document.getElementById('status').className = 'status error';
                        document.getElementById('status').innerHTML = 'Could not determine location from IP. Please try another method.';
                        document.getElementById('method-selection').style.display = 'block';
                    }
                })
                .catch(error => {
                    console.error("IP location error:", error);
                    document.getElementById('status').className = 'status error';
                    document.getElementById('status').innerHTML = 'Error getting location from IP. Please try another method.';
                    document.getElementById('method-selection').style.display = 'block';
                });
        }
        
        function showMapWithLocation(latitude, longitude) {
            document.getElementById('map-container').style.display = 'block';
            document.getElementById('map').style.display = 'block';
            
            if (!map) {
                // Initialize map
                map = L.map('map').setView([latitude, longitude], 13);
                L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                }).addTo(map);
                
                marker = L.marker([latitude, longitude]).addTo(map);
                
                // Add click handler to update if user wants to adjust
                map.on('click', function(e) {
                    lat = e.latlng.lat;
                    lng = e.latlng.lng;
                    
                    if (marker) {
                        marker.setLatLng(e.latlng);
                    } else {
                        marker = L.marker(e.latlng).addTo(map);
                    }
                    
                    // Update input fields if they exist
                    const latInput = document.getElementById('lat-input');
                    const lngInput = document.getElementById('lng-input');
                    if (latInput && lngInput) {
                        latInput.value = lat.toFixed(6);
                        lngInput.value = lng.toFixed(6);
                    }
                    
                    document.getElementById('location-info').innerHTML = 
                        '<h3>Selected Location</h3>' +
                        '<p>Latitude: ' + lat.toFixed(6) + '</p>' +
                        '<p>Longitude: ' + lng.toFixed(6) + '</p>' +
                        '<p><em>Click "Use This Location" when ready</em></p>';
                    
                    locationAccuracy = 10000; // Default accuracy for map clicks (10km)
                    document.getElementById('confirm-btn').style.display = 'inline-block';
                });
            } else {
                map.setView([latitude, longitude], 13);
                if (marker) {
                    marker.setLatLng([latitude, longitude]);
                } else {
                    marker = L.marker([latitude, longitude]).addTo(map);
                }
            }
            
            // Make sure the map is properly sized
            setTimeout(function() {
                map.invalidateSize();
            }, 100);
        }
        
        window.onload = function() {
            console.log("Location detector started");
            // Don't automatically get location, wait for user to choose method
        };
    </script>
</body>
</html>"""
                        
                        self.send_response(200)
                        self.send_header('Content-type', 'text/html')
                        self.end_headers()
                        self.wfile.write(location_html.encode('utf-8'))
                        return
                
                def log_message(self, format, *args):
                    return
            
            port = 5123
            server = None
            
            for attempt in range(10):
                try:
                    server = socketserver.TCPServer(("", port), LocationHandler)
                    break
                except OSError:
                    port += 1
            
            if not server:
                messagebox.showerror("Server Error", "Could not start location server. Please try manual entry.")
            else:
                server_thread = threading.Thread(target=server.serve_forever)
                server_thread.daemon = True
                server_thread.start()
                print(f"Started location server on port {port}")
                
                browser_url = f"http://127.0.0.1:{port}/"
                
                try:
                    webbrowser.open(browser_url)
                    print(f"Opened browser with URL: {browser_url}")
                except Exception as e:
                    print(f"Error opening browser with URL: {e}")
                
                messagebox.showinfo(
                    "Location Detection", 
                    "A browser window has opened to detect your location.\n\n" +
                    "Please allow location access if prompted, or use the alternatives (map selection, IP-based, manual entry).\n\n" +
                    "Click 'Use This Location' when you're satisfied with the selected location."
                )
                
                wait_dialog = tk.Toplevel(root)
                wait_dialog.title("Waiting for Location")
                wait_dialog.geometry("400x150")
                wait_dialog.resizable(False, False)
                
                tk.Label(wait_dialog, text="Waiting for location data from browser...", pady=10).pack()
                progress_var = tk.DoubleVar()
                progress_bar = tk.ttk.Progressbar(wait_dialog, variable=progress_var, maximum=100)
                progress_bar.pack(fill=tk.X, padx=20, pady=10)
                
                cancel_var = tk.BooleanVar(value=False)
                def on_cancel():
                    cancel_var.set(True)
                    wait_dialog.destroy()
                
                cancel_btn = tk.Button(wait_dialog, text="Cancel", command=on_cancel)
                cancel_btn.pack(pady=10)
                
                import random
                progress = 0
                
                timeout = 120
                start_time = time.time()
                
                while not location_data["received"] and not cancel_var.get():
                    if time.time() - start_time > timeout:
                        messagebox.showwarning("Timeout", "Location detection timed out. Please try again or use manual entry.")
                        break
                    
                    progress = min(progress + random.uniform(0.5, 1.5), 95)
                    progress_var.set(progress)
                    
                    wait_dialog.update()
                    time.sleep(0.1)
                
                try:
                    wait_dialog.destroy()
                except:
                    pass
                
                server.shutdown()
                server.server_close()
                
                if location_data["received"]:
                    latitude = location_data["lat"]
                    longitude = location_data["lng"]
                    
                    try:
                        geolocator = Nominatim(user_agent="fish_trash_detector")
                        location = geolocator.reverse(f"{latitude}, {longitude}", exactly_one=True)
                        location_name = location.address if location else f"Location: {latitude:.6f}, {longitude:.6f}"
                    except Exception as e:
                        print(f"Error getting location name: {e}")
                        location_name = f"Browser Location: {latitude:.6f}, {longitude:.6f}"
                    
                    save_last_location(latitude, longitude, location_name)
                    
                    return latitude, longitude, location_name
                else:
                    messagebox.showinfo("Location Detection", "Location detection cancelled or failed. Trying manual entry.")
            
        if location_method == "2" or not location_data.get("received", False):
            valid_coords = False
            while not valid_coords:
                lat_input = simpledialog.askstring("Location Input", "Enter latitude (-90 to 90):", initialvalue="0.0")
                if lat_input is None:
                    break
                    
                lon_input = simpledialog.askstring("Location Input", "Enter longitude (-180 to 180):", initialvalue="0.0")
                if lon_input is None:
                    break
                
                try:
                    latitude = float(lat_input.strip())
                    longitude = float(lon_input.strip())
                    
                    if -90 <= latitude <= 90 and -180 <= longitude <= 180:
                        valid_coords = True
                    else:
                        messagebox.showerror("Invalid Coordinates", "Latitude must be between -90 and 90, and longitude between -180 and 180.")
                except ValueError:
                    messagebox.showerror("Invalid Format", "Please enter valid numeric coordinates (e.g., -34.5 or 45.67).")
            
            if valid_coords:
                custom_name = simpledialog.askstring(
                    "Location Name", 
                    "Enter a name for this location (leave blank to use automatic name):",
                    initialvalue=""
                )
                
                if custom_name and custom_name.strip():
                    location_name = custom_name.strip()
                else:
                    try:
                        geolocator = Nominatim(user_agent="fish_trash_detector")
                        location = geolocator.reverse(f"{latitude}, {longitude}", exactly_one=True)
                        location_name = location.address if location else f"Manual Coordinates: {latitude:.6f}, {longitude:.6f}"
                    except Exception as e:
                        print(f"Error getting location name: {e}")
                        location_name = f"Manual Coordinates: {latitude:.6f}, {longitude:.6f}"
                
                save_last_location(latitude, longitude, location_name)
                
                return latitude, longitude, location_name
        
        messagebox.showwarning("Location Detection", "Could not determine location. Using default coordinates.")
        return 0.0, 0.0, "Unknown Location"
        
    except Exception as e:
        print(f"Error in get_location: {e}")
        messagebox.showwarning("Location Error", f"Error detecting location: {e}\nUsing default coordinates.")
        return 0.0, 0.0, "Error Location"

def save_last_location(latitude, longitude, location_name):
    try:
        settings_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "location_settings.json")
        settings = {
            "latitude": latitude,
            "longitude": longitude,
            "location_name": location_name,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        with open(settings_file, "w") as f:
            json.dump(settings, f)
            
        print(f"Saved location: {location_name}")
        return True
    except Exception as e:
        print(f"Error saving location settings: {e}")
        return False

def get_last_location():
    try:
        settings_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "location_settings.json")
        
        if os.path.exists(settings_file):
            with open(settings_file, "r") as f:
                settings = json.load(f)
                
            latitude = settings.get("latitude")
            longitude = settings.get("longitude")
            location_name = settings.get("location_name")
            
            if latitude is not None and longitude is not None and location_name:
                return float(latitude), float(longitude), location_name
                
    except Exception as e:
        print(f"Error reading last location: {e}")
        
    return None

def get_weather(latitude, longitude):
    try:
        params = {
            'lat': latitude,
            'lon': longitude,
            'appid': WEATHER_API_KEY,
            'units': 'imperial'
        }
        
        response = requests.get(WEATHER_API_URL, params=params)
        data = response.json()
        
        if response.status_code == 200:
            weather_condition = data['weather'][0]['main']
            temperature = data['main']['temp']
            return weather_condition, temperature
        else:
            print(f"Weather API error: {data.get('message', 'Unknown error')}")
            return "Unknown", 0.0
    except Exception as e:
        print(f"Error getting weather data: {e}")
        return "Unknown", 0.0

def save_detection_data(timestamp, latitude, longitude, location_name, weather, temp, trash_count, detection_types, image_path=None):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO detections 
        (timestamp, latitude, longitude, location_name, weather_condition, temperature, trash_count, detection_types, image_path)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (timestamp, latitude, longitude, location_name, weather, temp, trash_count, detection_types, image_path))
        
        conn.commit()
        conn.close()
        print(f"Data saved to database: {trash_count} trash items at {location_name}")
    except Exception as e:
        print(f"Error saving to database: {e}")

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = MODEL_PATH

if not os.path.exists(model_path):
    print(f"Model not found at {model_path}")
    for alt_model in ["yolov8n.pt", "yolov8s.pt", "yolo11n.pt"]:
        alt_path = os.path.join(current_dir, alt_model)
        if os.path.exists(alt_path):
            print(f"Using alternative model: {alt_model}")
            model_path = alt_path
            break
    else:
        print("No suitable model found. Please check the model path.")
        exit(1)

model = YOLO(model_path)

class_mappings = {
    'bottle': 'plastic bottle',
    'plastic_bottle': 'plastic bottle',
    'glass': 'glass bottle',
    'glass_bottle': 'glass bottle',
    'paper': 'paper',
    'cardboard': 'cardboard',
    'can': 'can',
    'metal_can': 'can',
    'aluminum_can': 'can',
    'battery': 'battery',
    'bottle_cap': 'plastic bottle cap',
    'cap': 'plastic bottle cap',
    'pop_tab': 'pop tab',
    'pull_tab': 'pop tab',
    'drink_carton': 'drink carton',
    'carton': 'drink carton',
    'fish': 'Fish'
}

print("Available detection classes in the model:")
for cls_id, cls_name in model.names.items():
    print(f"  {cls_id}: {cls_name}")
print()

actual_model_path = model_path
model_name = os.path.basename(actual_model_path)
print(f"Using model: {model_name} ({actual_model_path})")

tracked_trash = []
trash_count_total = 0

setup_database()

latitude, longitude, location_name = get_location()
print(f"Location: {location_name} ({latitude}, {longitude})")

if WEATHER_API_KEY != "your_api_key_here":
    weather_condition, temperature = get_weather(latitude, longitude)
    print(f"Weather: {weather_condition}, Temperature: {temperature}°F")
else:
    weather_condition, temperature = "API Key Not Set", 0.0
    print("Weather API key not set. Weather data will not be collected.")

os.makedirs("detection_images", exist_ok=True)

def get_centroid(coords):
    x1, y1, x2, y2 = map(int, coords)
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    return (cx, cy)

def update_tracked_trash(detections, current_time, frame=None, detection_types=None):
    global tracked_trash, trash_count_total
    new_tracked = []
    used = set()
    for obj in tracked_trash:
        matched = False
        for i, (centroid, coords) in enumerate(detections):
            if i in used:
                continue
            dist = np.linalg.norm(np.array(obj['centroid']) - np.array(centroid))
            if dist < DISTANCE_THRESHOLD:
                matched = True
                used.add(i)
                obj['centroid'] = centroid
                new_tracked.append(obj)
                break
    for i, (centroid, coords) in enumerate(detections):
        if i not in used:
            new_tracked.append({'centroid': centroid, 'continuous_start': current_time, 'counted': False})
    
    prev_count = trash_count_total
    for obj in new_tracked:
        if not obj['counted'] and (current_time - obj['continuous_start'] >= 1.0):
            trash_count_total += 1
            obj['counted'] = True
    
    if trash_count_total > prev_count and frame is not None:
        if detection_types:
            detection_types_list = list(detection_types)
            if not detection_types_list or (len(detection_types_list) == 1 and 'Fish' in detection_types_list):
                detection_types_list = [TRASH_CATEGORY]
        else:
            detection_types_list = [TRASH_CATEGORY]
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = f"detection_images/detection_{timestamp}.jpg"
        cv2.imwrite(image_path, frame)
        
        detection_types_str = json.dumps(detection_types_list)
        timestamp_iso = datetime.datetime.now().isoformat()
        save_detection_data(
            timestamp_iso, 
            latitude, 
            longitude, 
            location_name,
            weather_condition,
            temperature,
            trash_count_total,
            detection_types_str,
            image_path
        )
    
    tracked_trash = new_tracked

last_save_time = time.time()
SAVE_INTERVAL = 60

mode = input("Select mode (1 for live camera, 2 for video file, 3 for still image): ")
cap = None
still_image = None
water_filter_enabled = False
dark_percent = 0
if mode == "1":
    cap = cv2.VideoCapture(0)
elif mode == "2":
    root = tk.Tk()
    root.withdraw()
    video_path = filedialog.askopenfilename(title="Select video file", filetypes=[("MP4 files", "*.mp4"), ("AVI files", "*.avi"), ("MOV files", "*.mov")])
    if not video_path:
        print("No file selected. Exiting.")
        exit()
    water_input = input("Apply water filters to the video? (true/false): ").strip().lower()
    if water_input in ["true", "yes", "1"]:
        water_filter_enabled = True
        try:
            dark_percent = float(input("Enter darkness percentage (0-100): "))
            if dark_percent < 0 or dark_percent > 100:
                dark_percent = 0
        except ValueError:
            dark_percent = 0
    cap = cv2.VideoCapture(video_path)
elif mode == "3":
    root = tk.Tk()
    root.withdraw()
    image_path = filedialog.askopenfilename(title="Select image file", filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
    if not image_path:
        print("No image selected. Exiting.")
        exit()
    still_image = cv2.imread(image_path)
    if still_image is None:
        print("Error loading image. Exiting.")
        exit()
else:
    print("Invalid mode selected. Exiting.")
    exit()

prev_time = time.time()
if mode in ["1", "2"]:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        results = model(frame, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD)
        boxes = results[0].boxes
        valid_detections = 0
        trash_detections = []
        detection_types = set()
        
        for box in boxes:
            coords = box.xyxy[0].cpu().numpy()
            conf = box.conf.cpu().numpy()[0]
            cls = int(box.cls.cpu().numpy()[0])
            original_label = model.names[cls]
            
            lower_label = original_label.lower()
            if lower_label in class_mappings:
                label = class_mappings[lower_label]
            elif any(mapping_key in lower_label for mapping_key in class_mappings.keys()):
                for key in class_mappings:
                    if key in lower_label:
                        label = class_mappings[key]
                        break
                else:
                    label = original_label
            else:
                label = original_label
            
            if conf < CONFIDENCE_THRESHOLD:
                continue
            
            if label != 'Fish' and label not in ALLOWED_NAMES:
                print(f"Ignoring detected item: {original_label} (mapped to {label})")
                continue
            
            x1, y1, x2, y2 = map(int, coords)
            box_width = x2 - x1
            box_height = y2 - y1
            box_area = box_width * box_height
            frame_area = frame.shape[0] * frame.shape[1]
            box_percentage = (box_area / frame_area) * 100
            
            if box_percentage > 90:
                print(f"Ignoring {label}: Covers {box_percentage:.1f}% of frame (too large)")
                continue
            
            valid_detections += 1
            
            display_label = label
            if label != 'Fish':
                detection_types.add(label)
                centroid = get_centroid(coords)
                trash_detections.append((centroid, coords))
            else:
                detection_types.add(label)
            
            color = GREEN if label == 'Fish' else RED
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{display_label} {conf*100:.1f}%", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        update_tracked_trash(trash_detections, curr_time, frame, detection_types)
        
        if curr_time - last_save_time >= SAVE_INTERVAL and valid_detections > 0:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            image_path = f"detection_images/periodic_{timestamp}.jpg"
            cv2.imwrite(image_path, frame)
            
            if detection_types:
                detection_types_list = list(detection_types)
                if not detection_types_list or (len(detection_types_list) == 1 and 'Fish' in detection_types_list):
                    detection_types_list = [TRASH_CATEGORY]
            else:
                detection_types_list = [TRASH_CATEGORY]
            
            detection_types_str = json.dumps(detection_types_list)
            timestamp_iso = datetime.datetime.now().isoformat()
            save_detection_data(
                timestamp_iso, 
                latitude, 
                longitude, 
                location_name,
                weather_condition,
                temperature,
                trash_count_total,
                detection_types_str,
                image_path
            )
            last_save_time = curr_time
        
        cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 240, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, WHITE, 2)
        cv2.putText(frame, f"Detections: {valid_detections}", (frame.shape[1] - 240, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, WHITE, 2)
        cv2.putText(frame, f"Trash Count: {trash_count_total}", (frame.shape[1] - 240, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, WHITE, 2)
        cv2.putText(frame, f"Loc: {location_name[:20]}...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
        cv2.putText(frame, f"Weather: {weather_condition}, {temperature:.1f}°F", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
        
        if SHOW_MODEL_INFO:
            cv2.putText(frame, f"Model: {model_name}", (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)
            cv2.putText(frame, f"Path: {actual_model_path}", (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
        
        if mode == "2" and water_filter_enabled:
            frame = cv2.convertScaleAbs(frame, alpha=(1 - dark_percent/100), beta=0)
        cv2.imshow("Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
else:
    while True:
        frame = still_image.copy()
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        results = model(frame, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD)
        boxes = results[0].boxes
        valid_detections = 0
        trash_detections = []
        detection_types = set()
        
        for box in boxes:
            coords = box.xyxy[0].cpu().numpy()
            conf = box.conf.cpu().numpy()[0]
            cls = int(box.cls.cpu().numpy()[0])
            original_label = model.names[cls]
            
            lower_label = original_label.lower()
            if lower_label in class_mappings:
                label = class_mappings[lower_label]
            elif any(mapping_key in lower_label for mapping_key in class_mappings.keys()):
                for key in class_mappings:
                    if key in lower_label:
                        label = class_mappings[key]
                        break
                else:
                    label = original_label
            else:
                label = original_label
            
            if conf < CONFIDENCE_THRESHOLD:
                continue
            
            if label != 'Fish' and label not in ALLOWED_NAMES:
                print(f"Ignoring detected item: {original_label} (mapped to {label})")
                continue
            
            x1, y1, x2, y2 = map(int, coords)
            box_width = x2 - x1
            box_height = y2 - y1
            box_area = box_width * box_height
            frame_area = frame.shape[0] * frame.shape[1]
            box_percentage = (box_area / frame_area) * 100
            
            if box_percentage > 90:
                print(f"Ignoring {label}: Covers {box_percentage:.1f}% of frame (too large)")
                continue
            
            valid_detections += 1
            
            display_label = label
            if label != 'Fish':
                detection_types.add(label)
                centroid = get_centroid(coords)
                trash_detections.append((centroid, coords))
            else:
                detection_types.add(label)
            
            color = GREEN if label == 'Fish' else RED
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{display_label} {conf*100:.1f}%", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        update_tracked_trash(trash_detections, curr_time, frame, detection_types)
        
        if curr_time - last_save_time >= 0.5:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            image_path = f"detection_images/still_{timestamp}.jpg"
            cv2.imwrite(image_path, frame)
            
            if detection_types:
                detection_types_list = list(detection_types)
                if not detection_types_list or (len(detection_types_list) == 1 and 'Fish' in detection_types_list):
                    detection_types_list = [TRASH_CATEGORY]
            else:
                detection_types_list = [TRASH_CATEGORY]
            
            detection_types_str = json.dumps(detection_types_list)
            timestamp_iso = datetime.datetime.now().isoformat()
            save_detection_data(
                timestamp_iso, 
                latitude, 
                longitude, 
                location_name,
                weather_condition,
                temperature,
                trash_count_total,
                detection_types_str,
                image_path
            )
            last_save_time = time.time() + 1000
        
        cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 240, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, WHITE, 2)
        cv2.putText(frame, f"Detections: {valid_detections}", (frame.shape[1] - 240, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, WHITE, 2)
        cv2.putText(frame, f"Trash Count: {trash_count_total}", (frame.shape[1] - 240, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, WHITE, 2)
        cv2.putText(frame, f"Loc: {location_name[:20]}...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
        cv2.putText(frame, f"Weather: {weather_condition}, {temperature:.1f}°F", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
        
        if SHOW_MODEL_INFO:
            cv2.putText(frame, f"Model: {model_name}", (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)
            cv2.putText(frame, f"Path: {actual_model_path}", (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
        
        cv2.imshow("Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
cv2.destroyAllWindows()

print("\nDetection complete!")
print("===========================================")
print("To view the data in the interactive dashboard:")
print("1. Run the API server with: python api_server.py")
print("2. Open your browser to: http://127.0.0.1:8080")
print("===========================================")
print(f"Detection data has been saved to: {DB_PATH}")
print(f"Detection images are in: {DETECTION_IMAGES_DIR}")
print("Script Ending!")