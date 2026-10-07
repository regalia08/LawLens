"""fetch_law 버전 선택 테스트. 행은 실제 lawSearch(eflaw) 응답에서 발췌 (네트워크 호출 없음).

실행: python -m unittest -v
"""
import unittest
from datetime import date

from fetch_law import pick_version


def row(name, mst, efyd, prom, code="연혁"):
    return {
        "법령명한글": name,
        "법령ID": "001872",
        "법령일련번호": mst,
        "시행일자": efyd,
        "공포일자": prom,
        "현행연혁코드": code,
    }


ROWS = [
    row("근로기준법", "286771", "20270610", "20260609", "시행예정"),
    row("근로기준법", "285279", "20261008", "20260407", "시행예정"),
    row("근로기준법", "290781", "20261002", "20260804", "현행"),
    row("근로기준법", "283457", "20260820", "20260219"),
    row("근로기준법", "77179", "20070701", "20070126"),
    row("근로기준법", "76363", "20070701", "20061221"),
    row("근로기준법 시행규칙", "269393", "20250223", "20250221", "현행"),
    row("근로기준법시행규칙", "59941", "20031215", "20031215"),
]


class PickVersionTests(unittest.TestCase):
    def test_picks_version_in_force_not_scheduled_ones(self):
        r = pick_version(ROWS, "근로기준법", date(2026, 10, 7))
        self.assertEqual(r["법령일련번호"], "290781")

    def test_scheduled_version_becomes_current_on_its_date(self):
        r = pick_version(ROWS, "근로기준법", date(2026, 10, 8))
        self.assertEqual(r["법령일련번호"], "285279")

    def test_same_effective_date_prefers_later_promulgation(self):
        r = pick_version(ROWS, "근로기준법", date(2007, 7, 1))
        self.assertEqual(r["법령일련번호"], "77179")

    def test_does_not_match_other_laws_with_same_prefix(self):
        r = pick_version(ROWS, "근로기준법", date(2026, 10, 7))
        self.assertEqual(r["법령명한글"], "근로기준법")

    def test_name_match_ignores_spaces_for_old_names(self):
        r = pick_version(ROWS, "근로기준법 시행규칙", date(2004, 1, 1))
        self.assertEqual(r["법령일련번호"], "59941")

    def test_none_before_first_version(self):
        self.assertIsNone(pick_version(ROWS, "근로기준법 시행규칙", date(2000, 1, 1)))


if __name__ == "__main__":
    unittest.main()
