"""
StreamSense Feed Simulator.
Manages real-time message streaming, async interval timing, and dual-trigger windowing (N messages or T seconds).
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, Set
from src.config import FeedConfig, DriftConfig
from src.data_loader import DataLoader
from src.drift_engine import DriftEngine
from src.drift_metrics import DriftMetricsCalculator

class FeedSimulator:
    def __init__(self, data_loader: DataLoader, drift_engine: DriftEngine):
        self.data_loader = data_loader
        self.drift_engine = drift_engine
        self.config = FeedConfig()
        self.metrics_calculator = DriftMetricsCalculator(burn_in_windows=self.config.burn_in_windows)

        self.is_running = False
        self.task: Optional[asyncio.Task] = None

        # Dual-trigger Windowing state
        self.current_window: List[Dict[str, Any]] = []
        self.window_start_time = time.time()
        self.total_messages_streamed = 0
        self.total_windows_processed = 0

        # Subscriptions for WebSockets
        self.message_listeners: Set[asyncio.Queue] = set()
        self.metrics_listeners: Set[asyncio.Queue] = set()

    def update_feed_config(self, new_config: FeedConfig):
        self.config = new_config
        if new_config.burn_in_windows != self.metrics_calculator.burn_in_windows_needed:
            self.metrics_calculator.burn_in_windows_needed = new_config.burn_in_windows
        if new_config.dataset_mode != self.data_loader.mode:
            self.data_loader.mode = new_config.dataset_mode
            self.data_loader.reload_data()

    def register_message_listener(self) -> asyncio.Queue:
        q = asyncio.Queue(maxsize=100)
        self.message_listeners.add(q)
        return q

    def unregister_message_listener(self, q: asyncio.Queue):
        self.message_listeners.discard(q)

    def register_metrics_listener(self) -> asyncio.Queue:
        q = asyncio.Queue(maxsize=50)
        self.metrics_listeners.add(q)
        return q

    def unregister_metrics_listener(self, q: asyncio.Queue):
        self.metrics_listeners.discard(q)

    async def broadcast_message(self, message: Dict[str, Any]):
        dead_queues = set()
        for q in self.message_listeners:
            try:
                if q.full():
                    try:
                        q.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                q.put_nowait(message)
            except Exception:
                dead_queues.add(q)
        self.message_listeners -= dead_queues

    async def broadcast_metrics(self, metrics: Dict[str, Any]):
        dead_queues = set()
        for q in self.metrics_listeners:
            try:
                if q.full():
                    try:
                        q.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                q.put_nowait(metrics)
            except Exception:
                dead_queues.add(q)
        self.metrics_listeners -= dead_queues

    async def _flush_window(self, trigger_reason: str):
        if not self.current_window:
            return

        window_items = list(self.current_window)
        self.current_window = []
        self.window_start_time = time.time()
        self.total_windows_processed += 1

        # Compute drift metrics across window
        metrics = self.metrics_calculator.compute_window_metrics(window_items)
        metrics["window_index"] = self.total_windows_processed
        metrics["trigger_reason"] = trigger_reason
        metrics["timestamp"] = time.strftime("%H:%M:%S")

        await self.broadcast_metrics(metrics)

    async def _run_stream_loop(self):
        self.window_start_time = time.time()
        while self.is_running:
            try:
                # 1. Fetch next raw review
                raw_review = self.data_loader.get_next_review()

                # 2. Process through drift engine
                drifted_item = self.drift_engine.process_review(raw_review)
                self.total_messages_streamed += 1

                # 3. Append to window buffer
                self.current_window.append(drifted_item)

                # 4. Broadcast live review item
                await self.broadcast_message(drifted_item)

                # 5. Check Dual Window Trigger (N messages OR T seconds)
                elapsed_window_time = time.time() - self.window_start_time
                if len(self.current_window) >= self.config.window_message_count:
                    await self._flush_window(f"Hit message limit ({self.config.window_message_count} msgs)")
                elif elapsed_window_time >= self.config.window_time_seconds:
                    await self._flush_window(f"Hit time interval ({self.config.window_time_seconds:.1f}s)")

                # Sleep to enforce message rate
                rate = max(0.1, self.config.messages_per_second)
                delay = 1.0 / rate
                await asyncio.sleep(delay)

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[FeedSimulator Error] {e}")
                await asyncio.sleep(1.0)

    def start(self):
        if not self.is_running:
            self.is_running = True
            self.config.is_running = True
            self.task = asyncio.create_task(self._run_stream_loop())

    def stop(self):
        if self.is_running:
            self.is_running = False
            self.config.is_running = False
            if self.task:
                self.task.cancel()
                self.task = None

    def reset(self):
        self.stop()
        self.current_window = []
        self.window_start_time = time.time()
        self.total_messages_streamed = 0
        self.total_windows_processed = 0
        self.data_loader.reset_cursor()
        self.drift_engine.step_count = 0
        self.metrics_calculator = DriftMetricsCalculator(burn_in_windows=self.config.burn_in_windows)

    def get_status(self) -> Dict[str, Any]:
        return {
            "is_running": self.is_running,
            "total_messages_streamed": self.total_messages_streamed,
            "total_windows_processed": self.total_windows_processed,
            "current_window_buffered": len(self.current_window),
            "window_time_elapsed": round(time.time() - self.window_start_time, 1) if self.is_running else 0.0,
            "config": self.config.model_dump(),
            "drift_config": self.drift_engine.config.model_dump(),
            "dataset_info": self.data_loader.get_info(),
            "active_listeners": len(self.message_listeners)
        }
