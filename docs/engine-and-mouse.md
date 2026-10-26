# Engine selection and mouse behavior

Use the official `3.3-alpha-r260925-2352` build for the primary runtime.
This revision matches the normal-play environment.
Use Windows native Python for capture, inference, and input.
Keep WSL for development, research, and suitable training tasks.
Keep 3.2.1 as a separate compatibility baseline.

The stable or alpha label alone does not determine the suitable engine.
The target server and normal-play environment determine the required revision.
Record the exact engine version and file digests for each experiment.
Do not transfer an acceptance result between revisions or operating systems without another test.

## Official artifacts

The tests used archives from the official Zandronum download host:

- [Linux alpha archive](https://zandronum.com/downloads/testing/3.3/ZandroDev3.3-260925-2352linux-x86_64.tar.bz2)
- [Windows alpha archive](https://zandronum.com/downloads/testing/3.3/ZandroDev3.3-260925-2352windows-x86_64.zip)

The local installation remains outside the public file set.
The public evidence records filenames, sizes, and digests.
The tests did not change either engine executable.
No MCP engine participated in these input tests.

## Test comparison

| Runtime | Result | Limit |
| --- | --- | --- |
| Linux 3.2.1 under WSLg | The operator observed movement, jump, camera movement, and attack. Pause and re-enabling worked. | The report did not identify every camera direction separately. |
| Linux alpha under WSLg | Input phases executed. The operator reported camera flicker with automatic and physical mouse movement. | Aim acceptance failed. The cause remains unresolved. |
| Windows alpha with Windows Python | The operator reported normal physical mouse movement and normal automatic actions without flicker. | This establishes a local interface test, not autonomous combat ability. |

All clients loaded the original MM8BDM v6a, CBM v9fh, and Gondola v2.7h2 packages.
The controlled tests used stock loopback servers and an ordinary Player client.
They did not join external servers.

## Linux failure isolation

The Linux camera flicker continued after both the controller and capture process stopped.
This excludes those running processes as a necessary condition for the reproduced symptom.
It does not identify a specific engine defect.

The installed SDL 1.2 interface uses `sdl12-compat` version 1.2.68-2.
It forwards calls to SDL 2.30.0.
The [SDL compatibility project](https://github.com/libsdl-org/sdl12-compat) describes this library relationship.
The test used X11 windows through WSLg.

Relative input, pointer recentering, the compatibility library, and the compositor remain candidate causes.
A [WSLg discussion](https://github.com/microsoft/wslg/issues/1361) reports limits around host pointer repositioning.
That report supports further investigation but does not establish the cause of this test failure.
No workaround received a successful Linux camera test.

The project selected the Windows runtime after the operator reported normal physical mouse behavior there.
The new Windows test then established normal automatic actions.
This avoids the observed environment failure without an engine patch.

## Windows input path

The mouse backend sends signed relative deltas with Win32 `SendInput`.
It does not set an absolute cursor position for camera movement.
Zero deltas produce no mouse event.
The backend reports a rejected input event as an error.
See the [Microsoft input documentation](https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-sendinput).

The capture backend uses the selected client area.
It checks the foreground window before capture.
It follows client movement and blocks input after a size change.
It rejects hidden or minimized windows through the window metadata check.
Process DPI handling keeps capture coordinates and client dimensions consistent.

These controls do not establish calibrated aim or protection after every possible process failure.
An independent input watchdog remains necessary before long unattended sessions.

## Timing limit

The first Windows trace used a clock with approximately 16 ms steps.
Do not interpret its zero-duration samples as zero input latency.
The controller now uses the high-resolution performance counter for frame age and stage timing.
The local Windows check reports a 100 ns counter resolution.
This does not establish the accuracy of visible response measurements.
Separate accepted-action timings from inactive frames when comparing performance.
