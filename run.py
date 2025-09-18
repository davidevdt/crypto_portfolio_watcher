#!/usr/bin/env python3
"""
Enhanced Application Runner with Background Data Service
Starts the Streamlit app along with background data collection
"""

import asyncio
import threading
import logging
import signal
import sys
import subprocess
import time
import webbrowser
from pathlib import Path

from services.background_data_service import background_service
from services.shutdown_handler import start_shutdown_server, stop_shutdown_server
from database.models import create_database

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class EnhancedAppRunner:
    """Enhanced app runner with background services."""

    def __init__(self):
        self.background_task = None
        self.streamlit_process = None
        self.loop = None
        self.shutdown_flag = False
        self.browser_opened = False

    def start_background_service(self):
        """Start the background data collection service with configurable interval."""
        try:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

            try:
                from services.portfolio_manager import PortfolioManager

                pm = PortfolioManager()
                refresh_interval_str = pm.get_setting(
                    "background_refresh_interval", "60"
                )
                background_refresh_seconds = int(refresh_interval_str)
                logger.info(
                    f"Using user's background refresh interval setting: {background_refresh_seconds} seconds"
                )
            except Exception as e:
                logger.warning(
                    f"Could not load user refresh interval setting: {e}. Using default."
                )
                background_refresh_seconds = 60

            logger.info(
                f"Starting background data collection service (every {background_refresh_seconds} seconds)..."
            )
            self.background_task = self.loop.create_task(
                background_service.start_background_process(
                    refresh_interval_seconds=background_refresh_seconds
                )
            )

            self.loop.run_until_complete(self.background_task)

        except asyncio.CancelledError:
            logger.info("Background service cancelled")
        except Exception as e:
            logger.error(f"Background service error: {e}")
        finally:
            if self.loop:
                self.loop.close()

    def start_streamlit_app(self):
        """Start the Streamlit application."""
        try:
            logger.info("Starting Streamlit application...")

            app_dir = Path(__file__).parent
            app_file = app_dir / "app.py"

            cmd = [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(app_file),
                "--server.address",
                "0.0.0.0",
                "--server.port",
                "8501",
                "--server.headless",
                "true",
                "--server.runOnSave",
                "true",
                "--theme.base",
                "dark",
            ]

            self.streamlit_process = subprocess.Popen(
                cmd,
                cwd=str(app_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
            )

            while not self.shutdown_flag:
                output = self.streamlit_process.stdout.readline()
                if output:
                    if "You can now view your Streamlit app" in output:
                        logger.info("✅ Streamlit app is ready!")
                        logger.info("🌐 App URL: http://localhost:8501")
                        # Auto-open browser
                        if not self.browser_opened:
                            try:
                                webbrowser.open("http://localhost:8501")
                                logger.info("🚀 Browser opened automatically!")
                                self.browser_opened = True
                            except Exception as e:
                                logger.warning(f"Could not auto-open browser: {e}")
                    elif "WARNING" not in output and "Stopping" not in output:
                        logger.info(f"Streamlit: {output.strip()}")

                if self.streamlit_process.poll() is not None:
                    break

                time.sleep(0.1)

        except Exception as e:
            logger.error(f"Error starting Streamlit app: {e}")

    def signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.shutdown()

    def shutdown(self):
        """Shutdown all services gracefully."""
        self.shutdown_flag = True

        logger.info("Shutting down services...")

        # Stop shutdown handler server
        stop_shutdown_server()

        if self.background_task and not self.background_task.done():
            background_service.stop_background_process()
            self.background_task.cancel()
            logger.info("Background service stopped")

        if self.streamlit_process:
            self.streamlit_process.terminate()
            try:
                self.streamlit_process.wait(timeout=10)
                logger.info("Streamlit app stopped")
            except subprocess.TimeoutExpired:
                logger.warning("Streamlit app did not stop gracefully, forcing...")
                self.streamlit_process.kill()

        if hasattr(self, "background_thread") and self.background_thread.is_alive():
            self.background_thread.join(timeout=5.0)
            if self.background_thread.is_alive():
                logger.warning("Background thread did not stop gracefully")

        logger.info("All services stopped")

    def run(self):
        """Run the complete application with background services."""
        try:
            logger.info("Initializing database...")
            create_database()

            # Start shutdown handler server
            logger.info("Starting shutdown handler...")
            start_shutdown_server()

            signal.signal(signal.SIGINT, self.signal_handler)
            signal.signal(signal.SIGTERM, self.signal_handler)

            logger.info("🚀 Starting Crypto Portfolio Tracker with Enhanced Features")
            logger.info("=" * 60)

            self.background_thread = threading.Thread(
                target=self.start_background_service,
                daemon=True,
                name="BackgroundDataService",
            )
            self.background_thread.start()

            time.sleep(2)

            logger.info("🌐" + "=" * 58)
            logger.info("🌐 STREAMLIT APP WILL BE AVAILABLE AT:")
            logger.info("🌐 http://localhost:8501")
            logger.info("🌐" + "=" * 58)

            self.start_streamlit_app()

        except KeyboardInterrupt:
            logger.info("Received Ctrl+C, shutting down...")
        except Exception as e:
            logger.error(f"Application error: {e}")
        finally:
            self.shutdown()


def main():
    """Main entry point."""
    logger.info("🔧 Crypto Portfolio Tracker - Enhanced Edition")
    logger.info("=" * 50)
    logger.info("Features:")
    logger.info("✅ Modular page-based architecture")
    logger.info("✅ Background data collection")
    logger.info("✅ Real-time portfolio value tracking")
    logger.info("✅ Advanced technical analysis")
    logger.info("✅ Multi-channel notifications")
    logger.info("✅ Enhanced take profit strategies")
    logger.info("✅ Comprehensive monitoring")
    logger.info("=" * 50)
    logger.info("🌐 App will be available at: http://localhost:8501")
    logger.info("⏳ Starting services... Please wait...")
    logger.info("=" * 50)

    runner = EnhancedAppRunner()
    runner.run()


if __name__ == "__main__":
    main()
