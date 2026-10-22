# Bundled binaries

`linux-x86_64/busybox` and `linux-aarch64/busybox` are static builds of BusyBox,
used only when a Linux system has no `sh` of its own (a distroless container, for
example). The runtime invokes the `sh` applet; nothing else from these binaries
is used.

| File | Source | Size | SHA-256 |
| --- | --- | --- | --- |
| `linux-x86_64/busybox` | `https://dl-cdn.alpinelinux.org/alpine/v3.21/main/x86_64/busybox-static-1.37.0-r14.apk`, member `bin/busybox.static` | 1,034,152 | `2571bdcaa3bc1bcc0b5d01c71d1c3b9ed21981f6bf1e98c458a86551cbd4545c` |
| `linux-aarch64/busybox` | `https://dl-cdn.alpinelinux.org/alpine/v3.21/main/aarch64/busybox-static-1.37.0-r14.apk`, member `bin/busybox.static` | 1,115,944 | `e383c8bc25a1137b8ee88718cc6df1f1e84c54521d6045fc837385995dcdf031` |

Both are extracted unchanged from Alpine's `busybox-static` package. They are
statically linked, so they run on any Linux kernel of the matching architecture
without a C library, and they are never executed on another platform: the
resolver returns them only for `sys.platform == "linux"` and a matching
`platform.machine()`.

Licensing: BusyBox is GPL-2.0-only. `COPYING.busybox` is the license text, and
the source is the upstream release at `https://busybox.net/downloads/busybox-1.37.0.tar.bz2`,
of which these are unmodified builds. Keep both files beside the binaries when
you redistribute them.

To refresh them for a new Alpine release, download the two `busybox-static`
packages, extract `bin/busybox.static`, and update this table. The tests read the
ELF header and fail when a file is not the architecture its directory names.
