"""Test protective behavior with fake files, without heating real hardware."""
import argparse
import importlib.util
from pathlib import Path
import sys
import tempfile
import threading
import time
from unittest.mock import patch

root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('guard', root/'scripts/temperature_guard.py')
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)

with tempfile.TemporaryDirectory(dir=root/'build') as tmp:
    folder = Path(tmp)
    sensor = folder/'sensor'
    channels = [('cpu/test', sensor, 1000)]
    marker = folder/'started'
    command = [sys.executable, '-c',
               'from pathlib import Path; import time; '
               'Path({!r}).write_text("started"); time.sleep(10)'.format(str(marker))]
    args = argparse.Namespace(cpu_only=True, warn=70, stop=75, interval=0.05, command=command)
    with patch.object(guard, 'ROOT', folder), patch.object(guard, 'discover', return_value=channels):
        sensor.write_text('80000')
        assert guard.run(args) == 125
        assert not marker.exists(), 'overheated precheck started workload'

        sensor.write_text('50000')
        def heat():
            time.sleep(0.15)
            sensor.write_text('80000')
        thread = threading.Thread(target=heat)
        thread.start()
        start = time.monotonic()
        assert guard.run(args) == 125
        thread.join()
        assert marker.exists() and time.monotonic()-start < 3

        sensor.write_text('50000')
        def fail_sensor():
            time.sleep(0.15)
            sensor.unlink()
        thread = threading.Thread(target=fail_sensor)
        thread.start()
        assert guard.run(args) == 125
        thread.join()

        sensor.write_text('50000')
        args.command = [sys.executable, '-c', 'raise SystemExit(7)']
        assert guard.run(args) == 7
        args.command = []
        assert guard.run(args) == 0

    fake_sys = folder/'sys'
    cpu = fake_sys/'class/hwmon/hwmon0'
    cpu.mkdir(parents=True)
    (cpu/'name').write_text('coretemp')
    (cpu/'temp1_input').write_text('60000')
    ve = fake_sys/'class/ve/ve0'
    ve.mkdir(parents=True)
    (ve/'model').write_text('1')
    for index in list(range(15, 19))+list(range(20, 29)):
        (ve/('sensor_'+str(index))).write_text('42000000')
    readings = guard.sample(guard.discover(fake_sys))
    assert readings['cpu/hwmon0/temp1_input'] == 60
    assert readings['ve0/sensor_15'] == 42
    (cpu/'fan1_input').write_text('1800')
    (cpu/'pwm1').write_text('127')
    (cpu/'pwm1_enable').write_text('1')
    fan_readings=guard.fan_sample(guard.discover_fans(fake_sys))
    assert fan_readings['hwmon0/fan1_input']=='1800'
    assert fan_readings['hwmon0/pwm1_enable']=='1'
    assert fan_readings['ve0/fan_rpm']=='unavailable'
    new_args=argparse.Namespace(warn=None,stop=None)
    assert guard.limits(new_args,'cpu/test')==(80,85)
    assert guard.limits(new_args,'ve0/sensor_15')==(70,75)
print('temperature precheck/abort/sensor-loss/units/exit-code tests: PASS')
