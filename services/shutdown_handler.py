#!/usr/bin/env python3
"""
Shutdown Handler Service
Handles graceful shutdown requests from browser close detection
"""
import threading
import logging
import signal
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class ShutdownHandler(BaseHTTPRequestHandler):
    """HTTP handler for shutdown requests."""

    def do_POST(self):
        """Handle POST request for shutdown."""
        parsed_path = urlparse(self.path)

        if parsed_path.path == "/shutdown-app":
            logger.info("Received shutdown request from browser")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(b"Shutdown initiated")

            # Trigger shutdown in a separate thread
            def shutdown_app():
                logger.info("Initiating graceful shutdown...")
                # Send SIGTERM to the main process
                os.kill(os.getpid(), signal.SIGTERM)

            threading.Thread(target=shutdown_app, daemon=True).start()
        else:
            self.send_response(404)
            self.end_headers()

    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, format, *args):
        """Suppress default HTTP server logging."""
        pass


class ShutdownServer:
    """Simple HTTP server for handling shutdown requests."""

    def __init__(self, port=8502):
        self.port = port
        self.server = None
        self.thread = None

    def start(self):
        """Start the shutdown server."""
        try:
            self.server = HTTPServer(("localhost", self.port), ShutdownHandler)
            self.thread = threading.Thread(
                target=self.server.serve_forever, daemon=True, name="ShutdownServer"
            )
            self.thread.start()
            logger.info(f"Shutdown server started on port {self.port}")
        except Exception as e:
            logger.warning(f"Could not start shutdown server: {e}")

    def stop(self):
        """Stop the shutdown server."""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            logger.info("Shutdown server stopped")


# Global shutdown server instance
shutdown_server = ShutdownServer()


def start_shutdown_server():
    """Start the shutdown server."""
    shutdown_server.start()


def stop_shutdown_server():
    """Stop the shutdown server."""
    shutdown_server.stop()
