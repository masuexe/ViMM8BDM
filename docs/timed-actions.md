# Timed action execution

The input worker now accepts a timed window through `InputWorker.submit_timed`.
This path preserves short key presses and splits larger mouse motions into bounded packets.
The existing `submit` method retains its previous action limit.
Model policies do not use the new path automatically.

## Window contract

A `TimedWindow` contains full input states with ordered nanosecond offsets.
Its duration must not exceed 50 milliseconds.
It contains at most 1024 packets, including start and end states.
Each mouse packet contains at most 24 counts per axis.
The start and end packets contain no mouse motion.
The caller must specify the intended end state.

The worker accepts one pending window at a time.
It rejects another window while packets remain pending.
Only an accepted window renews the 200-millisecond input lease.
Later packets and status requests do not renew that lease.
The worker checks focus and the lease before each later packet.

Pause, focus loss, parent loss, and lease expiry cancel pending packets.
These conditions also release owned inputs.
An old generation cannot submit input after pause and restart.
An ordinary action cancels pending timed input and clears fractional mouse remainders.
If a packet is more than 20 milliseconds late, the worker pauses with `timing_overrun`.

The worker polls approximately every millisecond while a window remains pending.
Windows scheduling can delay a poll.
These offsets specify requested times, not guaranteed game response times.
A desktop API call can also delay execution.

## Compile recorded labels

`compile_label` accepts a recorded action label and two explicit experimental mouse scales.
It preserves key transitions, mouse event order, and their receipt-time offsets.
It checks the recorded mouse totals and final held state.
Unsupported keys, modifier bindings, secondary buttons, and wheel events cause an error.

The compiler splits each scaled mouse event into packets of at most 23 counts per axis.
The extra count allows for integer rounding with a fractional remainder.
Packets from one mouse event retain the same requested time.
The worker sends those packets in order, subject to focus, lease, and timing checks.
It does not distribute them across later game ticks.

The worker retains fractional counts across completed timed windows.
Pause and cancellation clear those remainders.
Integer output can differ from the cumulative requested motion by at most half a count per axis before cancellation.
This arithmetic property does not establish physical mouse-unit equivalence.
The compiler does not change source recordings or labels.

## Offline checks

All 2005 navigation samples compiled with candidate scales 1.06 and 0.92.
They produced 10229 packets, with at most 24 packets in one window.
Every sample retained its scaled mouse total before integer conversion.
The cumulative integer output differed by approximately 0.22 horizontal counts and zero vertical counts.
The source hashes matched.
This check sent no desktop input.

Linux and Windows each passed 113 tests.
The tests include exact clock checks and an independent input-process check.
They check short taps, fractional remainders, stale images, focus loss, lease expiry, and pause cancellation.
They also check overload rejection and cancellation after a scheduling stall.
These tests do not establish game movement accuracy or training readiness.

## Attended game check

Start the official Windows client with the original packages.
Use a local test map with clear space around the player.
Run this command with a new output directory:

```powershell
.local\windows-venv\Scripts\python -m vimm8bdm.timed_check `
  --window-id 0x12345 --run-dir runs/timed-check
```

Replace the example window identifier with the current game window identifier.
The tool waits for F8 without an initial timeout.

1. Select the game window.
2. Close menus.
3. Keep the real mouse still.
4. Press F8.
5. Observe short right and left strafes, followed by horizontal and vertical turns.
6. Check that the player stops when the sequence ends.

The sequence sends no attack, jump, or weapon input.
Esc, F8, or focus loss stops this check after its first activation.
F9 exits the tool.
Start a new run to repeat the check.
Keep logs and screenshots private.
Record the operator result separately from packet timing measurements.
