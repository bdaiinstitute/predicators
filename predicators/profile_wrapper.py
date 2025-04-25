import sys
import cProfile
import atexit
import runpy

def main():
    if len(sys.argv) < 2:
        print("Usage: python profile_wrapper.py target_script.py [args]")
        sys.exit(1)

    target_script = sys.argv[1]
    script_args = sys.argv[2:]

    # Replace sys.argv so the target script sees the correct arguments
    sys.argv = [target_script] + script_args

    profiler = cProfile.Profile()
    profiler.enable()

    def save_profile():
        profiler.disable()
        print("Saving profile to profile.out...")
        profiler.dump_stats("profile.out")

    atexit.register(save_profile)

    # Run the target script as __main__
    runpy.run_path(target_script, run_name="__main__")

if __name__ == "__main__":
    main()
