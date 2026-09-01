"""
StreamSense FastAPI Main Application.
Provides RESTful controls and real-time WebSockets for data drift simulation and feed monitoring.
"""

import os
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.config import DriftConfig, FeedConfig
from src.data_loader import DataLoader
from src.drift_engine import DriftEngine
from src.feed_simulator import FeedSimulator

app = FastAPI(
    title="StreamSense API",
    description="Real-time Social Feed Summarization & Sentiment Analysis with Synthetic Semantic Data Drift",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Core singletons
data_loader = DataLoader(mode="sample")
drift_engine = DriftEngine()
feed_simulator = FeedSimulator(data_loader=data_loader, drift_engine=drift_engine)

CONTROL_PANEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "control_panel")

if os.path.exists(CONTROL_PANEL_DIR):
    app.mount("/static", StaticFiles(directory=CONTROL_PANEL_DIR), name="static")


@app.on_event("startup")
async def startup_event():
    print("[StreamSense] Backend initialized successfully.")


@app.on_event("shutdown")
async def shutdown_event():
    feed_simulator.stop()


@app.get("/")
async def serve_control_panel():
    index_path = os.path.join(CONTROL_PANEL_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h1>StreamSense Control Panel building...</h1>")


@app.get("/api/status")
async def get_status():
    return feed_simulator.get_status()


@app.post("/api/drift/configure")
async def configure_drift(config: DriftConfig):
    drift_engine.update_config(config)
    return {"status": "ok", "drift_config": drift_engine.config.model_dump()}


@app.post("/api/feed/configure")
async def configure_feed(config: FeedConfig):
    feed_simulator.update_feed_config(config)
    return {"status": "ok", "feed_config": feed_simulator.config.model_dump()}


@app.post("/api/feed/start")
async def start_feed():
    feed_simulator.start()
    return {"status": "ok", "is_running": True}


@app.post("/api/feed/stop")
async def stop_feed():
    feed_simulator.stop()
    return {"status": "ok", "is_running": False}


@app.post("/api/feed/reset")
async def reset_feed():
    feed_simulator.reset()
    return {"status": "ok", "is_reset": True}


@app.get("/api/dataset/meta")
async def get_dataset_meta():
    return data_loader.get_info()


@app.websocket("/ws/feed")
async def websocket_feed_endpoint(websocket: WebSocket):
    await websocket.accept()
    queue = feed_simulator.register_message_listener()
    try:
        while True:
            msg = await queue.get()
            await websocket.send_json(msg)
    except WebSocketDisconnect:
        feed_simulator.unregister_message_listener(queue)
    except Exception:
        feed_simulator.unregister_message_listener(queue)


@app.websocket("/ws/metrics")
async def websocket_metrics_endpoint(websocket: WebSocket):
    await websocket.accept()
    queue = feed_simulator.register_metrics_listener()
    try:
        while True:
            metrics = await queue.get()
            await websocket.send_json(metrics)
    except WebSocketDisconnect:
        feed_simulator.unregister_metrics_listener(queue)
    except Exception:
        feed_simulator.unregister_metrics_listener(queue)
