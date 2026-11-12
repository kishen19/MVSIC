"""Synthetic CPU accounting and fail-closed guard tests; no /proc polling."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

spec = importlib.util.spec_from_file_location("host_guard_compare", Path(__file__).parent / "refactor/compare.py")
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)


def process(ticks=0, *, parent=0, start=1, name="worker"):
    return {"ticks": ticks, "ppid": parent, "start": start, "name": name}


class HostGuardTests(unittest.TestCase):
    def test_cpu_delta_excludes_nested_and_reparented_owned_processes(self):
        state = compare.HostLoadState(owner=1, ticks_per_second=100)
        state.observe({1: process(), 2: process(parent=1), 3: process(parent=2),
                       4: process(5000)}, 10)
        row = state.observe({1: process(100), 2: process(2000, parent=1),
                             3: process(3000, parent=0), 4: process(5200)}, 12)
        self.assertEqual(row["unrelated_cpu_cores"], 1)
        self.assertEqual([p["pid"] for p in row["top_processes"]], [4])
        self.assertFalse(row["triggered"])

    def test_pid_reuse_and_new_processes_need_a_matching_counter_baseline(self):
        state = compare.HostLoadState(owner=1, ticks_per_second=100)
        state.observe({1: process(), 2: process(9000, parent=1), 4: process(8000)}, 0)
        row = state.observe({1: process(), 2: process(100, start=2),
                             4: process(200, start=2), 5: process(300)}, 2)
        self.assertFalse(row["busy"])
        row = state.observe({1: process(), 2: process(300, start=2),
                             4: process(400, start=2), 5: process(500)}, 4)
        self.assertEqual(row["unrelated_cpu_cores"], 3)
        self.assertEqual({p["pid"] for p in row["top_processes"]}, {2, 4, 5})

    def test_two_sustained_samples_trigger_and_a_quiet_sample_resets(self):
        state = compare.HostLoadState(owner=1, ticks_per_second=100)
        state.observe({1: process(), 2: process()}, 0)
        self.assertFalse(state.observe({1: process(), 2: process(150)}, 2)["triggered"])
        self.assertEqual(state.observe({1: process(), 2: process(160)}, 4)["busy_samples"], 0)
        self.assertFalse(state.observe({1: process(), 2: process(310)}, 6)["triggered"])
        self.assertTrue(state.observe({1: process(), 2: process(460)}, 8)["triggered"])

    def test_aggregate_threshold_includes_small_services_but_tolerates_normal_background(self):
        state = compare.HostLoadState(owner=1, ticks_per_second=100)
        state.observe({1: process(), **{p: process() for p in range(2, 22)}}, 0)
        quiet = state.observe({1: process(), **{p: process(4) for p in range(2, 22)}}, 2)
        self.assertAlmostEqual(quiet["unrelated_cpu_cores"], .4)
        self.assertFalse(quiet["busy"])
        busy = state.observe({1: process(), **{p: process(24) for p in range(2, 22)}}, 4)
        self.assertAlmostEqual(busy["unrelated_cpu_cores"], 2)
        self.assertTrue(busy["busy"])

    def test_monitor_error_and_sustained_activity_interrupt_only_the_controller(self):
        for failure in (OSError("cannot read counters"), {"triggered": True}):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temporary:
                guard = compare.HostGuard(temporary)
                guard.stopped = Mock()
                guard.stopped.wait.return_value = False
                if isinstance(failure, Exception):
                    guard.sample = Mock(side_effect=failure)
                else:
                    guard.sample = Mock(return_value=failure)
                with patch.object(compare.os, "kill") as interrupt:
                    guard.watch()
                interrupt.assert_called_once_with(guard.owner, compare.signal.SIGINT)
                record = json.loads((Path(temporary) / "host_guard.json").read_text())
                self.assertFalse(record["timings_accepted"])
                self.assertIn("unsuitable", record["reason"])

    def test_preflight_read_error_blocks_without_starting_a_thread(self):
        with tempfile.TemporaryDirectory() as temporary:
            guard = compare.HostGuard(temporary)
            with patch.object(compare, "process_snapshot", side_effect=PermissionError("denied")):
                with self.assertRaises(PermissionError):
                    guard.start()
            self.assertIsNone(guard.thread)
            self.assertEqual(json.loads((Path(temporary) / "host_guard.json").read_text())["status"], "blocked")

    def test_shared_host_records_disjoint_load_but_keeps_it_out_of_overlap_pool(self):
        masks = {2: {2: set(range(64)), 22: {40, 41}}, 3: {3: set(range(192))}}
        state = compare.HostLoadState(1, 100, reserved_cpus={80, 176}, affinity_reader=masks.__getitem__)
        state.observe({1: process(), 2: process(), 3: process()}, 0)
        row = state.observe({1: process(), 2: process(12800), 3: process(20)}, 2)
        self.assertAlmostEqual(row["unrelated_cpu_cores"], 64.1)
        self.assertAlmostEqual(row["overlapping_cpu_cores"], .1)
        self.assertFalse(row["busy"])
        self.assertEqual(len(row["processes"]), 2)
        self.assertEqual(row["processes"][0]["thread_affinities"]["22"], [40, 41])

    def test_one_secondary_thread_on_an_smt_sibling_blocks_the_whole_heavy_process(self):
        reserved = compare.expand_smt_cpus([80, 84], {80: {80, 176}, 84: {84, 180}})
        self.assertEqual(reserved, {80, 84, 176, 180})
        reader = lambda pid: {pid: set(range(64)), pid + 100: {176}}
        state = compare.HostLoadState(1, 100, reserved_cpus=reserved, affinity_reader=reader)
        state.observe({1: process(), 2: process()}, 0)
        self.assertFalse(state.observe({1: process(), 2: process(200)}, 2)["triggered"])
        row = state.observe({1: process(), 2: process(400)}, 4)
        self.assertTrue(row["triggered"])
        self.assertTrue(row["processes"][0]["overlaps_reserved"])
        self.assertEqual(row["overlapping_cpu_cores"], 1)

    def test_shared_host_aggregate_threshold_uses_only_overlapping_processes(self):
        masks = {2: {2: {80}}, 3: {3: {176}}, 4: {4: set(range(192))}, 5: {5: {0}}}
        state = compare.HostLoadState(1, 100, reserved_cpus={80, 176}, affinity_reader=masks.__getitem__)
        state.observe({p: process() for p in range(1, 6)}, 0)
        row = state.observe({1: process(), 2: process(120), 3: process(120), 4: process(120), 5: process(4000)}, 2)
        self.assertAlmostEqual(row["overlapping_cpu_cores"], 1.8)
        self.assertAlmostEqual(row["unrelated_cpu_cores"], 21.8)
        self.assertTrue(row["busy"])

    def test_shared_host_affinity_errors_and_missing_masks_fail_closed(self):
        for reader in (Mock(side_effect=PermissionError("denied thread mask")), lambda pid: {}, lambda pid: {pid: set()}):
            state = compare.HostLoadState(1, 100, reserved_cpus={80, 176}, affinity_reader=reader)
            state.observe({1: process(), 2: process()}, 0)
            with self.assertRaises((PermissionError, RuntimeError)):
                state.observe({1: process(), 2: process(200)}, 2)

    def test_every_thread_affinity_is_read_and_task_set_changes_fail_closed(self):
        first = [Path("2"), Path("3")]
        with patch.object(compare.Path, "iterdir", side_effect=[first, first]), \
             patch.object(compare.os, "sched_getaffinity", side_effect=[{0}, {176}]) as affinity:
            self.assertEqual(compare.process_thread_affinities(2), {2: {0}, 3: {176}})
        self.assertEqual([call.args[0] for call in affinity.call_args_list], [2, 3])
        with patch.object(compare.Path, "iterdir", side_effect=[first, first + [Path("4")]]), \
             patch.object(compare.os, "sched_getaffinity", return_value={0}):
            with self.assertRaisesRegex(RuntimeError, "changed"):
                compare.process_thread_affinities(2)


if __name__ == "__main__":
    unittest.main()
