import time


class StageTimer:
    """
    Lightweight stage timer.

    For each stage stores:
      timing_<stage>_wall_s : elapsed wall-clock time
      timing_<stage>_cpu_s  : CPU time consumed by this process

    Repeated stages are accumulated.
    """

    def __init__(self):
        self._active = {}
        self._data = {}

    def start(self, name):
        name = str(name)

        if name in self._active:
            raise RuntimeError(
                f"Timing stage already active: {name}"
            )

        self._active[name] = (
            time.perf_counter(),
            time.process_time(),
        )

    def stop(self, name):
        name = str(name)

        started = self._active.pop(name, None)

        if started is None:
            return None

        wall0, cpu0 = started

        wall = float(
            time.perf_counter() - wall0
        )

        cpu = float(
            time.process_time() - cpu0
        )

        wall_key = f"timing_{name}_wall_s"
        cpu_key = f"timing_{name}_cpu_s"

        self._data[wall_key] = (
            float(self._data.get(wall_key, 0.0))
            + wall
        )

        self._data[cpu_key] = (
            float(self._data.get(cpu_key, 0.0))
            + cpu
        )

        return wall, cpu

    def snapshot(self):
        return dict(self._data)
