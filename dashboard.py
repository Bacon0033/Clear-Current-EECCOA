import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import sqlite3
import json
import os
from flask import Flask, send_from_directory
import numpy as np
from datetime import datetime

def load_data():
    conn = sqlite3.connect('fish_trash_data.db')
    query = "SELECT * FROM detections ORDER BY timestamp DESC"
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date
        df['time'] = df['timestamp'].dt.time
        df['hour'] = df['timestamp'].dt.hour
        df['detection_types_list'] = df['detection_types'].apply(
            lambda x: json.loads(x) if x else []
        )
        all_types = set()
        for types_list in df['detection_types_list']:
            all_types.update(types_list)
        for trash_type in all_types:
            df[f'count_{trash_type}'] = df['detection_types_list'].apply(
                lambda x: x.count(trash_type) if x else 0
            )
    
    return df

server = Flask(__name__, static_folder='detection_images')
app = dash.Dash(__name__, server=server)

app.layout = html.Div([
    html.H1("Fish AI Trash Detection Dashboard", style={'textAlign': 'center', 'color': '#2c3e50', 'marginTop': 20}),
    
    html.Div([
        html.Div([
            html.H3("Detection Statistics", style={'textAlign': 'center', 'color': '#2c3e50'}),
            dcc.Graph(id='trash-count-graph'),
            dcc.Graph(id='trash-type-pie'),
        ], className='six columns', style={'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '10px', 'boxShadow': '0px 0px 10px rgba(0,0,0,0.1)'}),
        
        html.Div([
            html.H3("Detection Map", style={'textAlign': 'center', 'color': '#2c3e50'}),
            dcc.Graph(id='detection-map'),
            html.H4("Weather Impact", style={'textAlign': 'center', 'color': '#2c3e50', 'marginTop': 20}),
            dcc.Graph(id='weather-impact'),
        ], className='six columns', style={'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '10px', 'boxShadow': '0px 0px 10px rgba(0,0,0,0.1)'}),
    ], className='row', style={'margin': '20px'}),
    
    html.Div([
        html.H3("Recent Detections", style={'textAlign': 'center', 'color': '#2c3e50'}),
        html.Div(id='detection-gallery', style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'center'}),
    ], style={'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '10px', 'boxShadow': '0px 0px 10px rgba(0,0,0,0.1)', 'margin': '20px'}),
    
    dcc.Interval(
        id='interval-component',
        interval=10*1000,
        n_intervals=0
    )
], style={'fontFamily': 'Arial', 'margin': '0', 'padding': '0', 'backgroundColor': '#ecf0f1'})

@app.callback(
    [Output('trash-count-graph', 'figure'),
     Output('trash-type-pie', 'figure'),
     Output('detection-map', 'figure'),
     Output('weather-impact', 'figure'),
     Output('detection-gallery', 'children')],
    [Input('interval-component', 'n_intervals')]
)
def update_graphs(n):
    df = load_data()
    
    if df.empty:
        empty_fig = px.scatter(title="No data available yet")
        empty_pie = px.pie(title="No detection data")
        empty_map = px.scatter_mapbox(title="No location data")
        empty_weather = px.bar(title="No weather data")
        return empty_fig, empty_pie, empty_map, empty_weather, html.Div("No images available yet")
    
    fig_count = px.line(
        df.sort_values('timestamp'),
        x='timestamp',
        y='trash_count',
        title='Trash Count Over Time',
        labels={'trash_count': 'Total Trash Count', 'timestamp': 'Time'},
    )
    fig_count.update_layout(xaxis_title="Time", yaxis_title="Trash Count")
    
    type_columns = [col for col in df.columns if col.startswith('count_')]
    if type_columns:
        type_data = pd.melt(
            df[type_columns].sum().reset_index(),
            id_vars='index',
            value_name='count'
        )
        type_data.columns = ['trash_type', 'count']
        type_data['trash_type'] = type_data['trash_type'].apply(lambda x: x.replace('count_', ''))
        
        fig_pie = px.pie(
            type_data,
            values='count',
            names='trash_type',
            title='Trash Type Distribution',
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
    else:
        fig_pie = px.pie(title="No type data available")
    
    fig_map = px.scatter_mapbox(
        df,
        lat='latitude',
        lon='longitude',
        size='trash_count',
        color='trash_count',
        hover_name='location_name',
        hover_data=['timestamp', 'trash_count', 'weather_condition', 'temperature'],
        title='Detection Locations',
        zoom=10,
        height=500,
        color_continuous_scale=px.colors.sequential.Viridis
    )
    fig_map.update_layout(
        mapbox_style="open-street-map",
        margin={"r": 0, "t": 40, "l": 0, "b": 0}
    )
    
    weather_data = df.groupby('weather_condition').agg({
        'trash_count': 'mean',
        'id': 'count'
    }).reset_index()
    weather_data.columns = ['Weather Condition', 'Average Trash Count', 'Number of Detections']
    
    fig_weather = px.bar(
        weather_data,
        x='Weather Condition',
        y='Average Trash Count',
        title='Average Trash Count by Weather Condition',
        color='Average Trash Count',
        text='Number of Detections'
    )
    fig_weather.update_layout(xaxis_title="Weather Condition", yaxis_title="Average Trash Count")
    
    gallery_items = []
    recent_detections = df.sort_values('timestamp', ascending=False).head(8)
    
    for _, row in recent_detections.iterrows():
        image_path = row['image_path']
        if image_path and os.path.exists(image_path):
            image_filename = os.path.basename(image_path)
            item = html.Div([
                html.Img(src=f'/detection_images/{image_filename}', style={'width': '100%', 'borderRadius': '5px'}),
                html.P(f"Location: {row['location_name'][:20]}..."),
                html.P(f"Time: {row['timestamp'].strftime('%Y-%m-%d %H:%M')}"),
                html.P(f"Trash count: {row['trash_count']}")
            ], style={
                'width': '22%', 
                'margin': '10px', 
                'padding': '10px', 
                'borderRadius': '10px', 
                'boxShadow': '0px 0px 5px rgba(0,0,0,0.2)',
                'backgroundColor': '#f8f9fa'
            })
            gallery_items.append(item)
    
    if not gallery_items:
        gallery_items = [html.Div("No images available yet")]
    
    return fig_count, fig_pie, fig_map, fig_weather, gallery_items

@server.route('/detection_images/<path:path>')
def serve_image(path):
    return send_from_directory('detection_images', path)

if __name__ == '__main__':
    app.run_server(debug=True)
