import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path

import scenario_toolkit as tk

TRUTH = tk.ROOT / "ground_truth.json"
TRADE_BASES = ("S01", "S02", "S05", "S07", "S08")


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


class FileSafety(unittest.TestCase):
    def test_approved_files_are_never_writable(self):
        for name in ("T01-v001.json", "T08-v004_ground_truth.json", "T05-v002_report.json"):
            for overwrite in (False, True):
                with self.assertRaises(PermissionError):
                    tk.write_json(tk.DATA_DIR / name, {}, overwrite=overwrite)

    def test_other_files_are_not_silently_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "x.json"
            tk.write_json(target, {"a": 1})
            with self.assertRaises(FileExistsError):
                tk.write_json(target, {"a": 2})
            tk.write_json(target, {"a": 3}, overwrite=True)
            self.assertEqual(tk.read_json(target), {"a": 3})
            self.assertEqual([p.name for p in Path(tmp).iterdir()], ["x.json"])  # no temp leftovers

    def test_generate_refuses_the_approved_folder(self):
        with self.assertRaises(FileExistsError):
            quiet(tk.generate, tk.DATA_DIR, {"T01-v001"})


class Finance(unittest.TestCase):
    def test_fund_comparison_known_values(self):
        r = tk.fund_comparison(tk.load_base("S04"))
        self.assertAlmostEqual(r["values"]["Fund A"], 137688.62493758881, places=6)
        self.assertAlmostEqual(r["values"]["Fund B"], 145469.34517263254, places=6)
        self.assertEqual(r["winner"], "Fund B")

    def test_retirement_known_values(self):
        r = tk.retirement_truth(tk.load_base("S06"))
        self.assertAlmostEqual(r["projected_balance"], 1000045.735359339, places=6)
        self.assertEqual(r["extra_monthly_rounded_up"], 432.77)

    def test_one_cent_less_misses_the_target(self):
        initial, rate, months, contributions, withdrawals, target = tk.retirement_plan(tk.load_base("S06"))
        extra = tk.retirement_truth(tk.load_base("S06"))["extra_monthly_rounded_up"]
        self.assertLess(tk.project_balance(initial, rate, months, contributions, withdrawals, extra - 0.01), target)

    def test_s03_base_violations(self):
        found = {(v["kind"], v["scope"]) for v in tk.policy_violations(tk.load_base("S03"))}
        self.assertEqual(found, {("CLASS_MAX", "STOCK"), ("CLASS_MIN", "BOND"), ("INDIVIDUAL_MAX", "AAPL"),
                                 ("INDIVIDUAL_MAX", "MSFT"), ("PROHIBITED_CLASS", "CRYPTO")})

    def test_violations_are_reported_once_per_rule_and_scope(self):
        scenario = tk.load_base("S03")
        scenario["investment_policy"]["restrictions"].append(
            {"restriction_type": "PROHIBITED_CLASS", "parameters_json": {"asset_class": "CRYPTO", "max_pct": 0}})
        found = [(v["kind"], v["scope"]) for v in tk.policy_violations(scenario)]
        self.assertEqual(len(found), len(set(found)))

    def test_solver_plans_replay_as_valid(self):
        for sid in TRADE_BASES:
            scenario = tk.load_base(sid)
            solution = tk.solve_trade_plan(scenario, whole_shares=True)
            self.assertTrue(solution["found"], sid)
            self.assertTrue(tk.verify_trade_plan(scenario, solution["trades"])["valid"], sid)

    def test_verifier_rejects_rule_breaking_plans(self):
        s01 = tk.load_base("S01")
        cases = {
            "sell protected AAPL": [{"symbol": "AAPL", "quantity_delta": -1}],
            "fractional shares": [{"symbol": "BND", "quantity_delta": 0.5}],
            "sell more than owned": [{"symbol": "TSLA", "quantity_delta": -999}],
            "unknown asset": [{"symbol": "ZZZ", "quantity_delta": 1}],
            "overspend cash": [{"symbol": "BND", "quantity_delta": 5000}],
            "no trades leaves 20% individual cap breached": [],
        }
        for name, trades in cases.items():
            self.assertFalse(tk.verify_trade_plan(s01, trades)["valid"], name)

    def test_unknown_restriction_is_an_error_not_ignored(self):
        scenario = tk.load_base("S01")
        scenario["investment_policy"]["restrictions"].append({"restriction_type": "NEW_RULE", "parameters_json": {}})
        self.assertIn("Unsupported restriction: NEW_RULE", tk.verify_trade_plan(scenario, [])["errors"])
        with self.assertRaises(ValueError):
            tk.solve_trade_plan(scenario)

    @unittest.skipUnless(TRUTH.exists(), "ground_truth.json not available")
    def test_reference_plans_from_ground_truth(self):
        truth = tk.read_json(TRUTH)
        for sid in TRADE_BASES:
            result = tk.verify_trade_plan(tk.load_base(sid), truth[sid]["trades"])
            self.assertTrue(result["valid"], f"{sid}: {result['errors']}")
            self.assertAlmostEqual(result["final_cash"], truth[sid]["metrics"]["final_cash"], places=2)


class Dataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.tmp.name)
        quiet(tk.generate, cls.folder)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def audit(self, folder):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            ok = tk.audit(folder)
        return ok, out.getvalue()

    def test_generates_all_32_and_they_pass_the_audit(self):
        self.assertEqual(len(list(self.folder.glob("T??-v???.json"))), 32)
        ok, output = self.audit(self.folder)
        self.assertTrue(ok, output)

    def test_generation_is_deterministic(self):
        for sid, rows in tk.load_specs().items():
            for spec in rows:
                self.assertEqual(tk.build_scenario(sid, spec), tk.read_json(self.folder / f"{spec['id']}.json"))

    def test_scenario_id_is_the_variant_name_and_there_is_no_variant_id(self):
        for path in self.folder.glob("T??-v???.json"):
            scenario = tk.read_json(path)
            self.assertEqual(scenario["scenario_id"], path.stem)
            self.assertNotIn("variant_id", scenario)
            self.assertEqual(scenario["output_requirements"]["response_schema"]["properties"]["scenario_id"]["enum"], [path.stem])
            truth = tk.read_json(self.folder / f"{path.stem}_ground_truth.json")
            self.assertEqual(truth["scenario_id"], path.stem)
            self.assertNotIn("variant_id", truth)

    def test_old_two_field_identity_is_flagged_by_the_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            for path in self.folder.glob("*.json"):
                tk.write_json(Path(tmp) / path.name, tk.read_json(path))
            scenario = tk.read_json(Path(tmp) / "T02-v001.json")
            scenario["scenario_id"], scenario["variant_id"] = "S02", "T02-v001"      # the legacy form
            tk.write_json(Path(tmp) / "T02-v001.json", scenario, overwrite=True)
            ok, output = self.audit(Path(tmp))
            self.assertFalse(ok)
            self.assertIn("scenario_id must be 'T02-v001' and there must be no variant_id", output)

    def test_audit_catches_corruption(self):
        corruptions = {
            "T01-v001.json": lambda j: j["portfolio"].__setitem__("total_value", 1),
            "T02-v003.json": lambda j: j.__setitem__("scenario_id", "S02"),
            "T01-v002_ground_truth.json": lambda j: j["trades"].append({"symbol": "MSFT", "quantity_delta": -5}),
            "T03-v001_ground_truth.json": lambda j: j["violations"].pop(),
            "T04-v001_ground_truth.json": lambda j: j.__setitem__("winner", "Fund A"),
            "T06-v002_ground_truth.json": lambda j: j.__setitem__("extra_monthly_rounded_up", 1.0),
        }
        for name, corrupt in corruptions.items():
            with tempfile.TemporaryDirectory() as tmp:
                copy_dir = Path(tmp)
                for path in self.folder.glob("*.json"):
                    tk.write_json(copy_dir / path.name, tk.read_json(path))
                data = tk.read_json(copy_dir / name)
                corrupt(data)
                tk.write_json(copy_dir / name, data, overwrite=True)
                self.assertFalse(self.audit(copy_dir)[0], name)

    def test_compare_flags_edited_scenarios(self):
        with tempfile.TemporaryDirectory() as tmp:
            for path in self.folder.glob("*.json"):
                tk.write_json(Path(tmp) / path.name, tk.read_json(path))
            self.assertTrue(quiet(tk.compare, tmp))
            edited = tk.read_json(Path(tmp) / "T05-v002.json")
            edited["client_request"] += " edited"
            tk.write_json(Path(tmp) / "T05-v002.json", edited, overwrite=True)
            self.assertFalse(quiet(tk.compare, tmp))

    def test_reference_response_matches_each_answer_type(self):
        for vid in ("T01-v001", "T03-v001", "T03-v002", "T04-v001", "T06-v001"):
            scenario = tk.read_json(self.folder / f"{vid}.json")
            truth = tk.read_json(self.folder / f"{vid}_ground_truth.json")
            response = tk.reference_response(scenario, copy.deepcopy(truth))
            self.assertEqual(response["answer_type"], scenario["answer_type"])
            self.assertEqual(response["scenario_id"], scenario["scenario_id"])
        no_violations = tk.reference_response(tk.read_json(self.folder / "T03-v002.json"),
                                              tk.read_json(self.folder / "T03-v002_ground_truth.json"))
        self.assertEqual((no_violations["status"], no_violations["violations"]), ("FEASIBLE", []))

@unittest.skipUnless((tk.ROOT / "checks.json").exists() and (tk.ROOT / "metric_mapping.json").exists(),
                     "checks.json / metric_mapping.json templates not available")
class Consolidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.variants = Path(cls.tmp.name) / "variants"
        quiet(tk.generate, cls.variants)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def consolidate(self, out, **kwargs):
        return quiet(tk.consolidate, self.variants, out, **kwargs)

    def test_creates_three_files_keyed_by_variant_id(self):
        out = Path(self.tmp.name) / "fresh"
        self.assertTrue(self.consolidate(out))
        keys = {n: tk.read_json(out / n) for n in (tk.GROUND_TRUTH_NAME, tk.CHECKS_NAME, tk.MAPPING_NAME)}
        keys[tk.MAPPING_NAME] = keys[tk.MAPPING_NAME]["scenarios"]
        expected = {f"T{i:02d}-v{j:03d}" for i in range(1, 9) for j in range(1, 5)}
        for name, data in keys.items():
            self.assertEqual(set(data), expected, name)
        self.assertTrue(quiet(tk.check_consistency, out, self.variants))

    def test_entries_use_the_variant_name_as_scenario_id(self):
        out = Path(self.tmp.name) / "identity"
        self.consolidate(out)
        for key, entry in tk.read_json(out / tk.GROUND_TRUTH_NAME).items():
            self.assertEqual(entry["scenario_id"], key)
            self.assertNotIn("variant_id", entry)

    def test_legacy_input_files_give_identical_output(self):
        legacy = Path(self.tmp.name) / "legacy_inputs"
        for path in self.variants.glob("*.json"):
            data = tk.read_json(path)
            if "variant_id" not in data and data.get("scenario_id", "").startswith("T"):
                data["variant_id"], data["scenario_id"] = data["scenario_id"], tk.base_scenario_id(data["scenario_id"])
            tk.write_json(legacy / path.name, data)
        new_out, old_out = Path(self.tmp.name) / "from_new", Path(self.tmp.name) / "from_legacy"
        self.assertTrue(quiet(tk.consolidate, self.variants, new_out))
        self.assertTrue(quiet(tk.consolidate, legacy, old_out))
        for name in (tk.GROUND_TRUTH_NAME, tk.CHECKS_NAME, tk.MAPPING_NAME):
            self.assertEqual(tk.read_json(new_out / name), tk.read_json(old_out / name), name)

    def test_values_taken_from_each_variant(self):
        out = Path(self.tmp.name) / "values"
        self.consolidate(out)
        checks = tk.read_json(out / tk.CHECKS_NAME)
        param = lambda vid, kind: next(c["params_json"] for c in checks[vid] if c["check_type"] == kind)
        self.assertEqual(param("T03-v002", "STATUS")["expected"], "FEASIBLE")
        self.assertEqual(param("T03-v002", "EXACT_SCOPED_VIOLATIONS")["crypto_count"], 0)
        self.assertEqual(param("T03-v004", "EXACT_SCOPED_VIOLATIONS")["crypto_count"], 0)
        self.assertEqual(param("T03-v001", "EXACT_SCOPED_VIOLATIONS")["crypto_count"], 1)
        self.assertIsNone(param("T02-v003", "REPLAY_TRADES")["target_allocation_tolerance"])
        self.assertEqual(param("T01-v003", "REPLAY_TRADES")["target_allocation_tolerance"], 0.005)

    def test_second_run_changes_nothing(self):
        out = Path(self.tmp.name) / "again"
        self.consolidate(out)
        before = {p.name: p.read_bytes() for p in out.iterdir()}
        self.assertTrue(self.consolidate(out))
        self.assertEqual({p.name: p.read_bytes() for p in out.iterdir()}, before)

    def test_existing_entries_are_extended_not_overwritten(self):
        out = Path(self.tmp.name) / "extend"
        self.consolidate(out)
        path = out / tk.GROUND_TRUTH_NAME
        data = tk.read_json(path)
        data["T03-v001"]["metrics"]["violation_count"] = 4  # a stale entry conflicts
        tk.write_json(path, data, overwrite=True)
        stale = path.read_bytes()
        self.assertFalse(self.consolidate(out))
        self.assertEqual(path.read_bytes(), stale)          # refused: nothing written
        self.assertTrue(self.consolidate(out, replace=True))
        fixed = tk.read_json(path)
        self.assertEqual(fixed["T03-v001"]["metrics"]["violation_count"], 5)
        self.assertEqual(tk.read_json(out / "ground_truth.json.bak")["T03-v001"]["metrics"]["violation_count"], 4)

    def test_output_order_is_scenarios_then_variants(self):
        out = Path(self.tmp.name) / "ordered"
        self.consolidate(out)
        path = out / tk.GROUND_TRUTH_NAME
        data = tk.read_json(path)
        tk.write_json(path, {"T03-v001": data.pop("T03-v001"), "S01": {"x": 1}, **data}, overwrite=True)  # scrambled
        data["T01-v001"]["explanation"] = "changed"                                                       # forces a rewrite
        tk.write_json(path, {**tk.read_json(path), "T01-v001": data["T01-v001"]}, overwrite=True)
        self.consolidate(out, replace=True)
        keys = list(tk.read_json(path))
        self.assertEqual(keys[0], "S01")
        self.assertEqual(keys[1:], sorted(keys[1:]))

    def test_merge_keeps_unrelated_entries_and_order(self):
        merged, result = tk._merge({"S01": {"a": 1}, "T01-v001": {"x": 1}}, {"T01-v001": {"x": 1}, "T01-v002": {"x": 2}}, False)
        self.assertEqual(list(merged), ["S01", "T01-v001", "T01-v002"])
        self.assertEqual((result["added"], result["same"], result["conflicts"]), (["T01-v002"], ["T01-v001"], []))
        _, clash = tk._merge({"K": {"x": 1}}, {"K": {"x": 2}}, False)
        self.assertEqual(clash["conflicts"], ["K"])

    def test_consistency_catches_missing_and_wrong_entries(self):
        out = Path(self.tmp.name) / "broken"
        self.consolidate(out)
        checks = tk.read_json(out / tk.CHECKS_NAME)
        checks.pop("T05-v002")
        for check in checks["T03-v002"]:
            if check["check_type"] == "STATUS":
                check["params_json"]["expected"] = "VIOLATIONS_FOUND"
        tk.write_json(out / tk.CHECKS_NAME, checks, overwrite=True)
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            self.assertFalse(tk.check_consistency(out))
        self.assertIn("T05-v002: no checks entry", printed.getvalue())
        self.assertIn("T03-v002: STATUS expects", printed.getvalue())

    def test_consistency_rejects_a_variant_id_field_or_wrong_scenario_id(self):
        out = Path(self.tmp.name) / "ident_broken"
        self.consolidate(out)
        data = tk.read_json(out / tk.GROUND_TRUTH_NAME)
        data["T01-v001"]["variant_id"] = "T01-v001"
        data["T04-v002"]["scenario_id"] = "S04"
        tk.write_json(out / tk.GROUND_TRUTH_NAME, data, overwrite=True)
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            self.assertFalse(tk.check_consistency(out))
        self.assertIn("T01-v001: scenario_id must equal the key", printed.getvalue())
        self.assertIn("T04-v002: scenario_id must equal the key", printed.getvalue())


if __name__ == "__main__":
    unittest.main()
