"""Run on the Pi in a second SSH session while live_cascade.py is running.

Standard library only: python3 monitor_pi.py [--once]
Read-only; does not change clocks, cooling, power settings, or the cascade.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time
from urllib.request import urlopen


FLAG_NAMES = {0: "undervoltage NOW", 1: "frequency capped NOW",
              2: "throttling NOW", 3: "soft temperature limit NOW",
              16: "undervoltage occurred", 17: "frequency cap occurred",
              18: "throttling occurred", 19: "soft temperature limit occurred"}


def command(*args):
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL, timeout=1).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def read_number(path, divisor=1):
    try:
        return float(Path(path).read_text().strip()) / divisor
    except (OSError, ValueError):
        return None


def decode_flags(value):
    return [label for bit, label in FLAG_NAMES.items() if value & (1 << bit)]


def read_text(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def fan_status(sys_root="/sys"):
    """Discover fan telemetry without relying on a fixed hwmon device number.

    RPM is measured feedback. PWM and thermal cooling state are requests,
    not proof that the fan physically rotates. GPIO-only fans may expose none.
    """
    root = Path(sys_root)
    fans, controllers = [], []
    for hwmon in sorted((root / "class/hwmon").glob("hwmon*")):
        name = read_text(hwmon / "name") or hwmon.name
        inputs = sorted(hwmon.glob("fan*_input"))
        # Some pwm-fan versions expose a control without a tachometer reading.
        if not inputs and "fan" in name.lower() and (hwmon / "pwm1").exists():
            inputs = [hwmon / "fan1_input"]
        for input_path in inputs:
            number_text = input_path.name.removeprefix("fan").removesuffix("_input")
            rpm = read_number(input_path)
            pwm = read_number(hwmon / f"pwm{number_text}")
            if pwm is not None and not 0 <= pwm <= 255:
                pwm = None
            status = ("spinning" if rpm > 0 else "not spinning") if rpm is not None else "RPM unavailable"
            fans.append({"name": name, "channel": int(number_text), "status": status,
                         "rpm": rpm, "pwm_raw": pwm,
                         "requested_pwm_percent": pwm / 255 * 100 if pwm is not None else None,
                         "hwmon_path": str(hwmon)})
    for device in sorted((root / "class/thermal").glob("cooling_device*")):
        name = read_text(device / "type") or ""
        if "fan" in name.lower():
            controllers.append({"name": name, "state": read_number(device / "cur_state"),
                                "max_state": read_number(device / "max_state"),
                                "path": str(device)})
    return {"available": bool(fans or controllers), "fans": fans, "controllers": controllers}


def fan_lines(info):
    if not info["available"]:
        return ["Fan: unavailable (no fan telemetry exposed; this does not prove the fan is off)."]
    lines = []
    for fan in info["fans"]:
        line = f"Fan {fan['name']} #{fan['channel']}: {fan['status']} | {number(fan['rpm'], ' RPM')}"
        if fan["requested_pwm_percent"] is not None:
            line += f" | requested PWM {fan['requested_pwm_percent']:.0f}%"
            if fan["rpm"] == 0 and fan["pwm_raw"] > 0:
                line += " (requested on; no rotation reported)"
        lines.append(line)
    for controller in info["controllers"]:
        state, maximum = controller["state"], controller["max_state"]
        state_text = "unavailable" if state is None else str(int(state))
        maximum_text = "?" if maximum is None else str(int(maximum))
        lines.append(f"Fan controller {controller['name']}: requested cooling state {state_text}/{maximum_text}")
    if not info["fans"]:
        lines.append("Fan RPM unavailable; cooling state alone does not confirm rotation.")
    return lines


def measure(url):
    temp = command("vcgencmd", "measure_temp")
    throttle = command("vcgencmd", "get_throttled")
    clock = command("vcgencmd", "measure_clock", "arm")
    data = {"temperature_c": read_number("/sys/class/thermal/thermal_zone0/temp", 1000),
            "clock_mhz": None, "throttled_hex": None, "flags": [],
            "fan": fan_status(), "cascade": None}
    if temp and "=" in temp:
        try:
            data["temperature_c"] = float(temp.split("=")[1].split("'")[0])
        except ValueError:
            pass
    if throttle and "=" in throttle:
        try:
            flags = int(throttle.split("=")[1], 16)
            data["throttled_hex"] = hex(flags)
            data["flags"] = decode_flags(flags)
        except ValueError:
            pass
    if clock and "=" in clock:
        try:
            data["clock_mhz"] = int(clock.split("=")[1]) / 1_000_000
        except ValueError:
            pass
    if data["clock_mhz"] is None:
        data["clock_mhz"] = read_number("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq", 1000)
    try:
        with urlopen(url, timeout=1) as response:
            status = json.load(response)
            keys = ("mode", "processing_fps", "detection_ms", "heatmap_and_drawing_ms",
                    "cascade_ms", "detected_flowers", "processed_flowers", "skipped_flowers",
                    "candidate_tips", "frame_age_ms", "last_result_age_ms")
            data["cascade"] = {key: status.get(key) for key in keys}
    except (OSError, ValueError):
        pass
    return data


def number(value, suffix=""):
    return "unavailable" if value is None else f"{value:.1f}{suffix}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="Print one JSON report for sharing")
    parser.add_argument("--url", default="http://127.0.0.1:8000/status")
    parser.add_argument("--interval", type=float, default=2)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("Interval must be positive")
    try:
        while True:
            data = measure(args.url)
            if args.once:
                print(json.dumps(data, indent=2))
                return
            print(f"Temperature {number(data['temperature_c'], ' C')} | "
                  f"ARM {number(data['clock_mhz'], ' MHz')} | throttled={data['throttled_hex']}")
            if data["throttled_hex"] == "0x0":
                print("  No current or recorded throttling/undervoltage flags.")
            elif data["flags"]:
                print("  " + "; ".join(data["flags"]))
            else:
                print("  Throttling status unavailable; check vcgencmd manually.")
            for line in fan_lines(data["fan"]):
                print("  " + line)
            status = data["cascade"]
            if status:
                print(f"  FPS {number(status.get('processing_fps'))} | "
                      f"YOLO {number(status.get('detection_ms'), ' ms')} | "
                      f"heatmap/drawing {number(status.get('heatmap_and_drawing_ms'), ' ms')} | "
                      f"processed flowers {status.get('processed_flowers', 0)} | "
                      f"last result {number(status.get('last_result_age_ms'), ' ms')} ago")
            else:
                print("  Live cascade status unavailable; temperature monitoring still works.")
            print(flush=True)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
