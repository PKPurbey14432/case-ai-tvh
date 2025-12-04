"""
Main entry point for TVH Product Findability system.

This script orchestrates the data pipeline and launches the application.
It automatically runs all required pipeline steps if files are missing.
Run this to set up and start the system.
"""
import os
import subprocess
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def run_pipeline_step(script_name, description, required=True):
    """
    Run a pipeline step script.
    
    Args:
        script_name: Name of the Python script to run
        description: Description of what this step does
        required: Whether this step is required (True) or optional (False)
    
    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Script: {script_name}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            cwd=Path(__file__).parent
        )
        print(f"Completed: {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}: {e}")
        if required:
            print(f"This step is required. Please fix the error and try again.")
            return False
        else:
            print(f"This step is optional. Continuing...")
            return True
    except FileNotFoundError:
        print(f"Script {script_name} not found. Skipping...")
        return not required


def check_and_run_pipelines(auto_run=False, skip_optional=False):
    """
    Check what pipeline steps need to be run and optionally run them.
    
    Args:
        auto_run: If True, automatically run missing pipeline steps
        skip_optional: If True, skip optional steps (descriptions, copurchase)
    
    Returns:
        True if all required files exist or were created, False otherwise
    """
    print("\nChecking pipeline status...")
    print("="*60)
    
    steps_to_run = []
    
    # Step 1: Extract catalog
    if not Path("data/catalog_clean.csv").exists():
        steps_to_run.append(("extract_catalog.py", "Extract PDF catalog data", True))
    else:
        print("Found: data/catalog_clean.csv")
    
    # Step 2: Generate descriptions (optional)
    if not Path("data/catalog_with_descriptions.csv").exists():
        if not skip_optional:
            steps_to_run.append(("generate_descriptions.py", "Generate LLM descriptions", False))
        else:
            print("Skipping: Generate descriptions (optional, skipped)")
    else:
        print("Found: data/catalog_with_descriptions.csv")
    
    # Step 3: Build embeddings
    if not Path("data/embeddings.pkl").exists():
        steps_to_run.append(("build_embeddings.py", "Build embeddings", True))
    else:
        print("Found: data/embeddings.pkl")
    
    # Step 4: Generate co-purchase data (optional)
    if not Path("data/co_purchase.csv").exists():
        if not skip_optional:
            steps_to_run.append(("generate_copurchase.py", "Generate co-purchase data", False))
        else:
            print("Skipping: Generate co-purchase data (optional, skipped)")
    else:
        print("Found: data/co_purchase.csv")
    
    if not steps_to_run:
        print("\nAll pipeline steps completed. Ready to launch!")
        return True
    
    print(f"\nFound {len(steps_to_run)} pipeline step(s) that need to be run:")
    for i, (script, desc, req) in enumerate(steps_to_run, 1):
        req_text = "REQUIRED" if req else "OPTIONAL"
        print(f"  {i}. {desc} ({req_text}) - {script}")
    
    if not auto_run:
        print("\nTo run these steps automatically, use: python3 main.py --run-pipelines")
        print("Or run them manually:")
        for script, desc, _ in steps_to_run:
            print(f"  python3 {script}")
        return False
    
    # Run pipeline steps
    print("\nRunning pipeline steps automatically...")
    for script, desc, required in steps_to_run:
        success = run_pipeline_step(script, desc, required)
        if not success and required:
            print(f"\nFailed to complete required step: {desc}")
            return False
    
    print("\n" + "="*60)
    print("All pipeline steps completed successfully!")
    print("="*60)
    return True


def main():
    """Main function to launch the application."""
    import argparse
    
    parser = argparse.ArgumentParser(description="TVH Product Findability System")
    parser.add_argument(
        "--run-pipelines",
        action="store_true",
        help="Automatically run all missing pipeline steps"
    )
    parser.add_argument(
        "--skip-optional",
        action="store_true",
        help="Skip optional pipeline steps (descriptions, copurchase)"
    )
    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="Skip pipeline checks and launch app directly"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port to run Streamlit on (default: 8501)"
    )
    parser.add_argument(
        "--address",
        type=str,
        default="0.0.0.0",
        help="Address to bind Streamlit to (default: 0.0.0.0)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("TVH Product Findability System")
    print("=" * 60)
    print()
    
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY environment variable not set")
        print("   The application will not work without it.")
        print("   Set it with: export OPENAI_API_KEY='your-key-here'")
        print("   Or create a .env file with: OPENAI_API_KEY=your-key-here")
        print()
        if not args.run_pipelines:
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                sys.exit(1)
    
    # Check and run pipelines
    if not args.skip_check:
        pipelines_ready = check_and_run_pipelines(
            auto_run=args.run_pipelines,
            skip_optional=args.skip_optional
        )
        
        if not pipelines_ready:
            if not args.run_pipelines:
                print("\n" + "="*60)
                print("To automatically run all missing pipeline steps:")
                print("  python3 main.py --run-pipelines")
                print("="*60)
            sys.exit(1)
    
    # Final check for required files
    required_files = ["data/catalog_clean.csv", "data/embeddings.pkl"]
    missing_required = [f for f in required_files if not Path(f).exists()]
    
    if missing_required:
        print("\nError: Required files still missing:")
        for f in missing_required:
            print(f"  - {f}")
        print("\nPlease run the pipeline steps manually or use --run-pipelines")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("All requirements met!")
    print("Launching Streamlit application...")
    print("="*60)
    print()
    
    # Launch Streamlit
    try:
        subprocess.run([
            "streamlit", "run", "app.py",
            "--server.port", str(args.port),
            "--server.address", args.address
        ])
    except KeyboardInterrupt:
        print("\nApplication stopped by user.")
    except Exception as e:
        print(f"\nError launching application: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
