"""Run one process group with fail-closed CPU/VE temperature supervision.

Uses CPU coretemp millidegrees and NEC VE1 sensor 15..18,20..28 microdegrees.
No fan, clock, device or system settings are modified. Requires Python >=3.6.
"""
import argparse
import csv
from datetime import datetime, timezone
import math
import os
import re
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]


def limits(args, label):
    kind = 'cpu' if label.startswith('cpu/') else 've'
    defaults = (80, 85) if kind == 'cpu' else (70, 75)
    warn = getattr(args, 'warn', None)
    stop = getattr(args, 'stop', None)
    return (warn if warn is not None else getattr(args, kind+'_warn', defaults[0]),
            stop if stop is not None else getattr(args, kind+'_stop', defaults[1]))


def discover_fans(sysroot=Path('/sys')):
    result = []
    for hw in sorted((sysroot/'class/hwmon').glob('hwmon*')):
        for path in sorted(hw.glob('*')):
            if re.fullmatch(r'fan[0-9]+_input|pwm[0-9]+(?:_enable|_mode|_target_temp|'
                            r'_temp_sel|_step_up_time|_step_down_time|_auto_point[0-9]+_(?:temp|pwm))?',
                            path.name):
                result.append((hw.name+'/'+path.name, path))
    # sensor_29 is VE1 fan RPM per NEC driver table/decoder; often unavailable.
    for node in sorted((sysroot/'class/ve').glob('ve[0-9]*')):
        if (node/'model').exists() and (node/'model').read_text().strip() == '1':
            result.append((node.name+'/fan_rpm', node/'sensor_29'))
    return result


def fan_sample(channels):
    result = {}
    for label, path in channels:
        try:
            result[label] = str(int(path.read_text().strip()))
        except (OSError, ValueError, TypeError):
            result[label] = 'unavailable'
    return result


def fan_services():
    result = {}
    for service in ('nec-ve-fan-ctl.service', 'nec-ve-fan-ctl-mon.service'):
        try:
            p = subprocess.run(['systemctl', 'is-active', service], stdout=subprocess.PIPE,
                               stderr=subprocess.DEVNULL, timeout=2, universal_newlines=True)
            state = p.stdout.strip()
            result[service] = state if state in ('active', 'inactive', 'failed', 'activating',
                                                'deactivating', 'unknown') else 'unavailable'
        except (OSError, subprocess.TimeoutExpired):
            result[service] = 'unavailable'
    return result


def bmc_fans():
    """Read only fan SDR records; never prompt or retain credentials."""
    try:
        p = subprocess.run(['sudo', '-n', 'ipmitool', 'sdr', 'type', 'Fan'],
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                           timeout=3, universal_newlines=True)
        if p.returncode:
            return {'bmc/read_status': 'unavailable'}
        result = {'bmc/read_status': 'ok'}
        for line in p.stdout.splitlines():
            fields = [part.strip() for part in line.split('|')]
            if len(fields) == 5 and re.fullmatch(r'[A-Za-z0-9 _-]{1,40}', fields[0]):
                match = re.fullmatch(r'([0-9]+) RPM', fields[4])
                if match:
                    result['bmc/'+fields[0]+'/rpm'] = match.group(1)
        return result
    except (OSError, subprocess.TimeoutExpired):
        return {'bmc/read_status': 'unavailable'}


def discover(sysroot=Path('/sys'), cpu_only=False):
    channels = []
    for hw in sorted((sysroot/'class/hwmon').glob('hwmon*')):
        name = hw/'name'
        if name.exists() and name.read_text().strip() == 'coretemp':
            for sensor in sorted(hw.glob('temp*_input')):
                channels.append(('cpu/'+hw.name+'/'+sensor.stem, sensor, 1000))
    if not channels:
        raise RuntimeError('CPU coretemp sensors unavailable')
    if not cpu_only:
        nodes = sorted((sysroot/'class/ve').glob('ve[0-9]*'))
        if not nodes:
            raise RuntimeError('VE temperature nodes unavailable')
        for node in nodes:
            # Verified against NEC VE1 driver sensor table and decoders.
            if (node/'model').read_text().strip() != '1':
                raise RuntimeError('temperature mapping only verified for VE1')
            for index in list(range(15, 19))+list(range(20, 29)):
                path = node/('sensor_'+str(index))
                channels.append((node.name+'/sensor_'+str(index), path, 1000000))
    return channels


def sample(channels):
    result = {}
    for label, path, divisor in channels:
        try:
            value = float(path.read_text().strip())/divisor
        except (OSError, ValueError, TypeError):
            raise RuntimeError('temperature reading unavailable: '+label)
        if not math.isfinite(value) or not 0 < value < 130:
            raise RuntimeError('invalid temperature: '+label)
        result[label] = value
    return result


def stop_group(proc):
    # Only the group created by this runner; never signal unrelated processes.
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        pass
    # Terminate any surviving descendants even if the group leader has exited.
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()


def run(args):
    channels = discover(cpu_only=args.cpu_only)
    initial = sample(channels)  # No command starts without valid readings.
    peak = dict(initial)
    fans = discover_fans()
    folder = ROOT/'build/results'
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    fd, name = tempfile.mkstemp(prefix=stamp+'-temperature-', suffix='.csv', dir=str(folder))
    cpu_warn,cpu_stop = limits(args,'cpu/')
    ve_warn,ve_stop = limits(args,'ve')
    print('Temperature guard: CPU warn/stop={}/{}C VE warn/stop={}/{}C interval={}s log={}'
          .format(cpu_warn,cpu_stop,ve_warn,ve_stop,args.interval,
                  Path(name).relative_to(ROOT)), file=sys.stderr)
    proc = None
    warned = set()
    fan_path = Path(name).with_name(Path(name).stem+'-fans.csv')
    fan_initial,fan_last,fan_changes = {},{},set()
    fan_service_previous = 0
    with os.fdopen(fd, 'w', newline='') as handle, open(str(fan_path),'w',newline='') as fan_handle:
        writer = csv.writer(handle)
        writer.writerow(['utc', 'sensor', 'temperature_c', 'warn_c', 'stop_c'])
        fan_writer = csv.writer(fan_handle)
        fan_writer.writerow(['utc','channel','raw_value'])

        def record_fans():
            nonlocal fan_service_previous
            values = fan_sample(fans)
            if time.monotonic()-fan_service_previous >= 5 or not fan_initial:
                values.update(fan_services())
                if getattr(args, 'bmc_fans', False):
                    values.update(bmc_fans())
                fan_service_previous = time.monotonic()
            utc = datetime.now(timezone.utc).isoformat()
            for label,value in values.items():
                if label not in fan_initial:
                    fan_initial[label]=value
                if label in fan_last and fan_last[label]!=value:
                    fan_changes.add(label)
                fan_last[label]=value
                fan_writer.writerow([utc,label,value])
            fan_handle.flush()

        def record(values):
            utc = datetime.now(timezone.utc).isoformat()
            for label, value in values.items():
                warn,stop=limits(args,label)
                writer.writerow([utc, label, value, warn, stop])
                peak[label] = max(peak.get(label, value), value)
                if value >= warn and label not in warned:
                    print('Temperature warning: {} {:.2f}C'.format(label, value), file=sys.stderr)
                    warned.add(label)
                if value < warn:
                    warned.discard(label)
            handle.flush()
            if any(value >= limits(args,label)[1] for label,value in values.items()):
                raise RuntimeError('temperature reached stop threshold')

        try:
            record(initial)
            record_fans()
            if not args.command:
                return 0
            proc = subprocess.Popen(args.command, cwd=str(ROOT), start_new_session=True)
            completion_code = None
            post_until = 0
            while True:
                # One final sample also covers jobs shorter than the interval.
                record(sample(channels))
                record_fans()
                code = proc.poll()
                if code is not None and completion_code is None:
                    stop_group(proc)
                    completion_code=code
                    post_until=time.monotonic()+getattr(args,'post_seconds',0)
                if completion_code is not None and time.monotonic()>=post_until:
                    return completion_code if completion_code >= 0 else 128-completion_code
                time.sleep(args.interval)
        except (OSError, ValueError, RuntimeError, KeyboardInterrupt) as error:
            if proc is not None:
                stop_group(proc)
            print('Temperature guard stopped command: {}'.format(error), file=sys.stderr)
            return 125
        finally:
            groups = sorted(set(label.split('/')[0] for label in peak))
            summary = ['{} peak={:.2f}C'.format(group, max(v for k, v in peak.items()
                                                          if k.split('/')[0] == group))
                       for group in groups]
            print('Temperature summary: '+', '.join(summary), file=sys.stderr)
            print('Fan observation: {} channels, {} observed changes; log={}'
                  .format(len(fan_initial),len(fan_changes),fan_path.relative_to(ROOT)),file=sys.stderr)
            for label in sorted(fan_changes):
                print('Fan channel changed: {} {} -> {} (may have intermediate changes)'
                      .format(label,fan_initial[label],fan_last[label]),file=sys.stderr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--warn', type=float, help='override warning for both CPU and VE')
    parser.add_argument('--stop', type=float, help='override stop threshold for both CPU and VE')
    parser.add_argument('--cpu-warn', type=float, default=80)
    parser.add_argument('--cpu-stop', type=float, default=85)
    parser.add_argument('--ve-warn', type=float, default=70)
    parser.add_argument('--ve-stop', type=float, default=75)
    parser.add_argument('--post-seconds', type=float, default=0,
                        help='continue monitoring after completion (0..30 seconds)')
    parser.add_argument('--interval', type=float, default=0.5)
    parser.add_argument('--cpu-only', action='store_true')
    parser.add_argument('--bmc-fans', action='store_true',
                        help='read fan RPM using sudo -n ipmitool (cached authorization required)')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command[:1] == ['--']:
        args.command.pop(0)
    if not (all(10 <= limits(args,label)[0] < limits(args,label)[1] <= 90
                for label in ('cpu/','ve')) and 0.05 <= args.interval <= 1
            and 0 <= args.post_seconds <= 30):
        parser.error('require 10 <= warn < stop <= 90 per device, interval 0.05..1s, post 0..30s')
    try:
        def interrupted(signum, frame):
            raise KeyboardInterrupt()
        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGHUP, interrupted)
        sys.exit(run(args))
    except (OSError, ValueError, RuntimeError) as error:
        print('Temperature guard cannot start: {}'.format(error), file=sys.stderr)
        sys.exit(125)
