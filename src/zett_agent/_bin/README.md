# Bundled bubblewrap

`linux-x86_64/bwrap` and `linux-aarch64/bwrap` are static builds of
[bubblewrap](https://github.com/containers/bubblewrap), used only on Linux when no
`bwrap` is on `PATH`. They are what the sandbox launches; nothing else in the
runtime runs them.

| File | Built from | Size | SHA-256 |
| --- | --- | --- | --- |
| `linux-x86_64/bwrap` | bubblewrap 0.11.0, Alpine 3.21, musl, `-Dselinux=disabled -Dc_link_args=-static`, stripped | 729,584 | `b5369792bab3d5ed69a80fd446ac29e2e789b35776fbb3958f675bb1241e78e8` |
| `linux-aarch64/bwrap` | the same source and options, built for aarch64 | 198,496 | `e3cc9d7deec54c262460e26244a0cb8690485fcc1c4e0a20943d33e380d10351` |

Both are statically linked against musl, so they need no C library on the target,
and the resolver only returns them on Linux with a matching `platform.machine()`.
SELinux support is compiled out; the sandbox never relied on it.

A statically linked `bwrap` is still only as strong as the kernel lets it be:
unprivileged user namespaces must work (on Ubuntu 24.04, set
`kernel.apparmor_restrict_unprivileged_userns=0` or load the bwrap AppArmor
profile), the same requirement the system package has. `LinuxSandbox.check()`
probes that and reports it rather than running unsandboxed.

Licensing: bubblewrap is LGPL-2.0-or-later. `COPYING.bwrap` is the license text
that ships with the release; the source is
`https://github.com/containers/bubblewrap/releases/download/v0.11.0/bubblewrap-0.11.0.tar.xz`,
of which these are unmodified builds.

To rebuild for a new release, from a machine with Docker:

```bash
docker run --rm --platform linux/amd64 -v "$PWD/out:/out" alpine:3.21 sh -c '
  apk add -q build-base meson ninja musl-dev libcap-dev libcap-static pkgconf curl xz
  curl -sSL https://github.com/containers/bubblewrap/releases/download/vX.Y.Z/bubblewrap-vX.Y.Z.tar.xz | tar xJ -C /tmp
  cd /tmp/bubblewrap-*/
  meson setup build --prefer-static -Dman=disabled -Dtests=false -Dselinux=disabled \
    -Dzsh_completion=disabled -Dbash_completion=disabled -Dc_link_args=-static
  ninja -C build && strip build/bwrap && cp build/bwrap /out/bwrap'
```

Repeat with `--platform linux/arm64`. The tests read each file's ELF header and
fail when it does not match the architecture its directory names.
