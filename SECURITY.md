# Security policy

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub:
[Report a vulnerability](https://github.com/kelsayed17/fnquant-kernels/security/advisories/new). Do not open a public
issue.

Include what you found, how to reproduce it, and the impact you expect. You will get an acknowledgement within a
week, and a fix or a plan within 30 days for a confirmed issue.

## Supported versions

fnquant-kernels is pre-1.0: only the latest release receives fixes.

## Scope

The kernels are a native extension called on caller-owned arrays, so these are in scope:

- a crash, out-of-bounds access or memory corruption reachable from Python through any public kernel;
- a Rust panic that reaches Python as anything but `fnquant_kernels.KernelPanic`, or that aborts the interpreter;
- an input (a length, a period, an index) that makes a kernel read or write outside the arrays it was given.

A kernel whose output differs from TA-Lib is a bug, not a vulnerability: please open an issue.
