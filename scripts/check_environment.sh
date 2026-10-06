#!/usr/bin/env bash
# Read-only capability check; intentionally excludes IDs, addresses and secrets.
set -u
printf 'CPU architecture: '
uname -m
printf 'CPU logical count: '
getconf _NPROCESSORS_ONLN
for tool in gcc ncc ve_exec uv; do
    if command -v "$tool" >/dev/null 2>&1; then
        printf '%s: available\n' "$tool"
    else
        printf '%s: unavailable\n' "$tool"
    fi
done
if command -v lspci >/dev/null 2>&1; then
    lspci | sed -E 's/^[^ ]+ //' | sed -n '/NEC.*Vector Engine/p; /VGA\|3D controller\|Display controller/p'
fi
status=0
count=0
for node in /sys/class/ve/ve[0-9]*; do
    [ -d "$node" ] || continue
    name=$(basename "$node")
    printf 'Hardware sysfs node: %s (not a runtime slot ID)\n' "$name"
done
for slot in /dev/veslot[0-9]*; do
    [ -L "$slot" ] || continue
    count=$((count+1))
    id=${slot#/dev/veslot}
    printf 'Runtime slot %s: ' "$id"
    if [ -c "$slot" ] && [ -r "$slot" ] && [ -w "$slot" ] &&
       [ -S "/var/opt/nec/ve/veos/veos${id}.sock" ]; then
        printf 'device and VEOS socket present; execution still needs smoke test\n'
    else
        printf 'device or VEOS socket inaccessible\n'
        status=1
    fi
done
if [ "$count" -eq 0 ]; then
    printf 'No /dev/veslotN links visible: VE execution blocked in this session\n'
    status=1
fi
exit "$status"
