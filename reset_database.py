#!/usr/bin/env python3
"""
Reset Database Script - Cleans the database for a fresh empty start
Run this to remove all existing data and start with a completely empty database.
"""

import os
import shutil
from pathlib import Path


def reset_database():
    """
    Remove all database files to start fresh.

    Searches for database files in multiple locations and removes them,
    equivalent to Settings > Delete All Records but faster.

    Returns:
        bool: True if any files were removed, False otherwise
    """
    print("🗑️  Resetting database to completely empty state...")
    print("📝 This is equivalent to Settings > Delete All Records but faster")

    project_root = Path(__file__).parent

    db_locations = []

    legacy_files = [
        project_root / "portfolio.db",
        project_root / "portfolio.db-shm",
        project_root / "portfolio.db-wal",
    ]
    db_locations.extend(legacy_files)

    common_db_folders = ["db_data", "database", "data", "db"]
    for folder_name in common_db_folders:
        db_folder = project_root / folder_name
        if db_folder.exists() and db_folder.is_dir():
            db_files = [
                db_folder / "portfolio.db",
                db_folder / "portfolio.db-shm",
                db_folder / "portfolio.db-wal",
            ]
            db_locations.extend(db_files)
            print(f"📁 Found database folder: {folder_name}/")

    removed_count = 0
    removed_folders = set()

    for db_file in db_locations:
        if db_file.exists():
            try:
                db_file.unlink()
                print(f"✅ Removed: {db_file}")
                removed_count += 1

                if db_file.parent != project_root:
                    removed_folders.add(db_file.parent)

            except Exception as e:
                print(f"❌ Failed to remove {db_file}: {e}")
        else:
            if db_file.parent == project_root:
                print(f"ℹ️  Not found: {db_file.name}")

    for folder in removed_folders:
        try:
            if not any(folder.iterdir()):
                print(f"📁 Database folder is empty: {folder.name}/")
            else:
                print(f"📁 Database folder cleaned: {folder.name}/")
        except:
            pass

    if removed_count > 0:
        print(f"\n🎉 Database reset complete! Removed {removed_count} files.")
        print("💾 This removed all:")
        print("   • Portfolios and assets")
        print("   • Transaction history")
        print("   • Historical price data")
        print("   • User settings and preferences")
        print("   • Watchlist items")
        print("   • Take profit levels")
        print("   • Cached data")
        print("\n📝 Next app launch will start with completely empty database.")
        print("🚀 Run 'python run.py' or 'streamlit run app.py' to start fresh.")
        print(
            "\n💡 This is the same as Settings > Delete All Records but removes files entirely"
        )
    else:
        print("\n✨ Database was already empty - no files to remove.")

    return removed_count > 0


if __name__ == "__main__":
    reset_database()
