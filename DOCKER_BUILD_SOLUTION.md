# Docker Build Notes

You may want to use host networking for `docker build` in case you are behind a firewall or VPN. Without it, Debian package resolution may stall for a long time and then fail with `Temporary failure resolving 'deb.debian.org'` during `apt-get update`.

Working build command:

```bash
docker build --no-cache --network host --build-arg HTTP_PROXY="" --build-arg HTTPS_PROXY="" -t oslidar-dev .
```

## Important dependency note

`ouster-sdk==0.16.1` declares `rosbags==0.9.23`, which is too old for the newer ROS 2 bag format used here. The Dockerfile avoids this by:

- installing the required `ouster-sdk` Python dependencies explicitly
- installing `ouster-sdk` with `pip install --no-deps .`
- re-pinning `rosbags==0.11.0`

If bag analysis fails with a message saying the installed `rosbags` package is too old, rebuild the image from the current Dockerfile and confirm the version inside the image:

```bash
docker run --rm oslidar-dev /opt/venv/bin/python -c "import importlib.metadata as m; print(m.version('rosbags'))"
```

Expected output:

```text
0.11.0
```

## Build compatibility note

The builder stage also patches `ouster-sdk` to force C++20 and builds `robin-map` explicitly before installing the Python package. This is not cosmetic. It avoids the earlier SDK wheel build failure where bundled code hit missing `std::launder` and `std::clamp` support during compilation.

The current Dockerfile keeps the fix in three parts:

- build and install `robin-map` in the builder image first
- prepend `set(CMAKE_CXX_STANDARD 20)` and `set(CMAKE_CXX_STANDARD_REQUIRED ON)` to `ouster-sdk/CMakeLists.txt`
- install the SDK wheel with explicit C/C++ flags during the build step

If a future Dockerfile cleanup removes that logic, expect the `ouster-sdk` wheel build to regress before runtime validation even starts.

## Validation commands

Mount the workspace and the known bag directory when validating the image:

```bash
docker run --rm --network host \
  -v $(pwd):/workspace \
  -v $(pwd)/ouster_bags:/data/ouster_bags \
  oslidar-dev \
  /opt/venv/bin/python /workspace/ouster_analysis.py \
  --input /data/ouster_bags/<recording> \
  --bag-mode walking --log silent
```

