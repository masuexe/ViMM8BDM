"""Start one local test client with a separate configuration and no remote server."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Start a local ViMM8BDM control test")
    for name in ("engine", "iwad", "base", "classes", "gondola"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--map", default="MM1CUT")
    parser.add_argument("--run-dir", type=Path, required=True, help="New local run directory")
    args = parser.parse_args()
    paths = {}
    for name in ("engine", "iwad", "base", "classes", "gondola"):
        path = getattr(args, name).resolve()
        if not path.is_file():
            parser.error(f"The {name} file does not exist")
        paths[name] = str(path)
    if not re.fullmatch(r"[A-Za-z0-9_]+", args.map):
        parser.error("The map name must contain letters, digits, or underscores")
    run = args.run_dir.resolve()
    try:
        run.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error("The run directory already exists. Select a new directory")
    config = run / "probe.cfg"
    config.write_text("unbind f8\nunbind f9\n"
                      "bind w +forward\nbind a +moveleft\nbind s +back\nbind d +moveright\n"
                      "bind mouse1 +attack\nbind space +jump\nfreelook true\n"
                      "mouse_sensitivity 1\nm_filter false\nm_pitch 1\nm_yaw 1\n"
                      "cl_capfps true\n", encoding="utf-8")
    command = [paths["engine"], "-noautoload", "-nosound", "-nomusic", "-config",
               str(run / "zandronum.ini"), "-iwad", paths["iwad"], "-file",
               paths["base"], paths["classes"], paths["gondola"],
               "+set", "fullscreen", "0", "+set", "vid_defwidth", "640",
               "+set", "vid_defheight", "480", "+exec", str(config),
               "+deathmatch", "1", "+timelimit", "0", "+fraglimit", "0", "+map", args.map]
    environment = os.environ.copy()
    if sys.platform.startswith("linux"):
        old_path = environment.get("LD_LIBRARY_PATH")
        environment["LD_LIBRARY_PATH"] = str(Path(paths["engine"]).parent) + (os.pathsep + old_path if old_path else "")
    with (run / "engine.log").open("x", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=run, env=environment, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    with (run / "launch.local.json").open("x", encoding="utf-8") as record:
        json.dump({"pid": process.pid, "arguments": command}, record, indent=2)
    print("The local test client started.")
    print("Process ID:", process.pid)
    print("Local records:", run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
