# Independent input watchdog

Live control now uses two Python processes.
The parent captures images and runs the policy.
The input worker owns keyboard input, mouse input, focus checks, and control hotkeys.
The processes exchange small messages through a local pipe.
They do not use an engine bridge or inspect game packets.

## Input ownership

The input worker starts paused.
The run duration starts after the first user enable command.
The separate initial waiting limit defaults to 180 seconds.
Use `--arm-timeout` to change that limit.
Only its user hotkeys can enable desktop input.
F8 enables or pauses input.
Esc pauses input.
F9 stops the worker and releases its inputs.
The policy process has no live enable command.

The worker rejects images older than 150 ms.
An accepted action renews a 200 ms input lease.
Status requests do not renew that lease.
The worker checks the lease and focus independently from inference.

Each enable or pause changes the action generation.
The parent attaches the observed generation to each action.
The worker rejects a result from an older generation.
Thus, a late result cannot restore input after takeover and re-enabling.

## Failure behavior

If inference stops producing valid actions, the lease expires.
The worker releases held keys and attack, then remains paused.
The user must press F8 to enable input again.
If the parent process exits, the worker releases its inputs and exits.
Normal shutdown also releases input before closing the worker.

The worker has a separate interpreter and event loop.
An inference stall does not block its hotkeys or deadline checks.
This replaces the earlier watchdog thread inside the policy process.

## Local fault test

Use an attended local game with a stock loopback server.
Do not use this test in a human multiplayer session.
The test briefly holds forward movement and attack before the requested fault.
The default policy never injects a fault.

Start the stall test from the repository root in PowerShell:

```powershell
.local\windows-venv\Scripts\python -m vimm8bdm --live --capture window `
  --window-id 0xYOUR_WINDOW_ID --policy watchdog_check --watchdog-fault stall `
  --seconds 90 --frames 1800 --hz 20 `
  --output runs/watchdog-stall.jsonl --input-trace runs/watchdog-stall-worker.jsonl
```

Select the game window.
Press F8 to start the test.
The parent pauses for 750 ms during a held action.
The input worker must release input without waiting for that pause to end.
The controller must not resume automatically.
Press F8 to check manual re-enabling.
Press F9 to finish.

For the parent-exit test, use a new pair of output files.
Replace `--watchdog-fault stall` with `--watchdog-fault exit`.
The parent exits with code 86 without normal cleanup.
The worker audit must still record input release and worker shutdown.
The parent does not print a final summary in this test.

## Audit interpretation

`--input-trace` enables a separate worker audit.
The worker queues audit events rather than waiting for file writes during input handling.
The trace records lease deadlines, generations, and the inputs that the worker owns.
It does not read player or opponent state.

Keep raw traces in the excluded `runs/` directory.
Check the final `trace_end` event before using the audit as complete evidence.
If the trace reports dropped events, the audit is incomplete.

The parent `input_ms` stage now includes the pipe round trip.
It is not a direct measurement of game response.
Worker release timestamps follow the input-release calls.
They do not identify the exact game tick that processed those calls.

## Test scope

Linux and Windows each passed 50 automated tests.
The process tests use a simulated input receiver in real spawned processes.
They cover inference stalls, forced parent termination, stale generations, status traffic, and F9 release.
A standalone Windows test also exits the parent without cleanup under the WSL launcher.
Desktop fault tests used the stock Windows alpha and its loopback server.

| Test | Observed release-call completion | Result |
| --- | --- | --- |
| 750 ms inference stall | 217.35 ms after the last accepted action | The worker released movement and attack. The operator could enable control again. |
| Parent exit without cleanup | 12.77 ms after the recorded fault marker | The worker recorded `parent_lost`, released input, and exited. |

Both worker traces ended with zero dropped events.
The release records contain no held keys and no held attack button.
The operator observed one automatic primary attack in the parent-exit test.
The test did not measure visible movement-stop latency.

The first desktop exit attempt accepted no actions and reached its waiting-time limit.
It did not exercise the fault.
The corrected initial waiting phase then permitted the exit test.
See the [sanitized evidence](evidence/input-watchdog.json).

The worker itself must remain responsive.
This design does not guarantee release after an input-worker crash, a blocked OS input API, or an OS failure.
Scheduling delays also prevent a hard real-time guarantee.
Calibrated mouse control and long-session acceptance remain separate work.
