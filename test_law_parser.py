"""law_parser 테스트. 실제 응답(근로기준법, 시행 2026-10-02)에서 발췌한 구조를 픽스처로 사용.

실행: python -m unittest -v
"""
import copy
import unittest
from datetime import date

from law_parser import CIRCLED, chunk_article, parse_law, summarize

TODAY = date(2026, 10, 7)


def unit(no, key, content, title="", **extra):
    base = {
        "조문번호": str(no),
        "조문제개정유형": "타법개정",
        "조문시행일자": "20261002",
        "조문변경여부": "N",
        "조문키": key,
        "조문내용": content,
        "조문제목": title,
        "조문여부": "조문",
    }
    base.update(extra)
    return base


SAMPLE = {
    "법령": {
        "법령키": "0018722026080421857",
        "기본정보": {
            "법령명_한글": "근로기준법",
            "법령ID": "001872",
            "시행일자": "20261002",
            "공포일자": "20260804",
            "제개정구분": "타법개정",
            "소관부처": {"content": "고용노동부", "소관부처코드": "1492000"},
        },
        "조문": {
            "조문단위": [
                unit(1, "0001000", "                    제1장 총칙", 조문여부="전문"),
                unit(
                    1,
                    "0001001",
                    "제1조(목적) 이 법은 헌법에 따라 근로조건의 기준을 정함으로써 근로자의 기본적 생활을 보장, 향상시키며 균형 있는 국민경제의 발전을 꾀하는 것을 목적으로 한다.",
                    "목적",
                ),
                unit(
                    2,
                    "0002001",
                    "제2조(정의)",
                    "정의",
                    항=[
                        {
                            "항번호": "①",
                            "항내용": "① 이 법에서 사용하는 용어의 뜻은 다음과 같다. <개정 2018.3.20, 2019.1.15>",
                            "호": [
                                {"호번호": "1.", "호내용": '1. "근로자"란 직업의 종류와 관계없이 임금을 목적으로 사업이나 사업장에 근로를 제공하는 사람을 말한다.'},
                                {"호번호": "9의2.", "호내용": "9의2. 가상의 호 번호 테스트"},
                            ],
                        },
                        {
                            "항번호": "②",
                            "항내용": "② 제1항제6호에 따라 산출된 금액이 그 근로자의 통상임금보다 적으면 그 통상임금액을 평균임금으로 한다.",
                        },
                    ],
                ),
                unit(2, "0015000", "                    제2장 근로계약", 조문여부="전문"),
                unit(
                    16,
                    "0016001",
                    "제16조(계약기간) 근로계약은 기간을 정하지 아니한 것과 일정한 사업의 완료에 필요한 기간을 정한 것 외에는 그 기간은 1년을 초과하지 못한다.",
                    "계약기간",
                    조문참고자료="[법률 제8372호(2007.4.11) 제16조의 개정규정은 같은 법 부칙 제3조의 규정에 의하여 2007년 6월 30일까지 유효함]",
                ),
                unit(
                    26,
                    "0026001",
                    "제26조(해고의 예고) 사용자는 근로자를 해고하려면 적어도 30일 전에 예고를 하여야 한다. 다만, 다음 각 호의 어느 하나에 해당하는 경우에는 그러하지 아니하다. <개정 2010.6.4, 2019.1.15>",
                    "해고의 예고",
                    항={
                        "호": [
                            {"호번호": "1.", "호내용": "1. 근로자가 계속 근로한 기간이 3개월 미만인 경우"},
                            {"호번호": "2.", "호내용": "2. 천재ㆍ사변, 그 밖의 부득이한 사유로 사업을 계속하는 것이 불가능한 경우"},
                        ]
                    },
                ),
                unit(
                    35,
                    "0035001",
                    "제35조 삭제 <2019.1.15>",
                    "",
                    조문참고자료="[2019.1.15 법률 제16270호에 의하여 위헌 결정된 이 조를 삭제함.]",
                ),
                unit(
                    43,
                    "0043021",
                    "제43조의2(체불사업주 명단 공개)",
                    "체불사업주 명단 공개",
                    조문가지번호="2",
                    항=[
                        {"항번호": "①", "항내용": "① 고용노동부장관은 임금등을 지급하지 아니한 사업주의 인적사항 등을 공개할 수 있다. <개정 2020.5.26>"},
                        {"항번호": "②", "항내용": "② 고용노동부장관은 명단 공개를 할 경우에 체불사업주에게 3개월 이상의 기간을 정하여 소명 기회를 주어야 한다."},
                    ],
                ),
                unit(
                    53,
                    "0053001",
                    "제53조(연장 근로의 제한)",
                    "연장 근로의 제한",
                    조문참고자료="[법률 제15513호(2018.3.20) 제53조제3항, 제53조제6항의 개정규정은 같은 법 부칙 제2조의 규정에 의하여 2022년 12월 31일까지 유효함]",
                    항=[
                        {"항번호": "①", "항내용": "① 당사자 간에 합의하면 1주 간에 12시간을 한도로 제50조의 근로시간을 연장할 수 있다."},
                        {"항번호": "③", "항내용": "③ 상시 30명 미만의 근로자를 사용하는 사용자는 근로자대표와 서면으로 합의한 경우 1주 간에 8시간을 초과하지 아니하는 범위에서 근로시간을 연장할 수 있다. <신설 2018.3.20>"},
                        {"항번호": "⑥", "항내용": "⑥ 제3항은 15세 이상 18세 미만의 근로자에 대하여는 적용하지 아니한다. <신설 2018.3.20>"},
                    ],
                ),
                unit(
                    60,
                    "0060001",
                    "제60조(연차 유급휴가)",
                    "연차 유급휴가",
                    항=[
                        {"항번호": "①", "항내용": "① 사용자는 1년간 80퍼센트 이상 출근한 근로자에게 15일의 유급휴가를 주어야 한다. <개정 2012.2.1>"},
                        {"항번호": "③", "항내용": "③ 삭제 <2017.11.28>"},
                        {"항번호": "④", "항내용": "④ 사용자는 3년 이상 계속하여 근로한 근로자에게는 가산휴가를 주어야 한다."},
                    ],
                ),
            ]
        },
    }
}


class ParseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.law, arts = parse_law(SAMPLE, today=TODAY)
        cls.list = arts
        cls.arts = {a.label: a for a in arts}

    def test_heading_rows_are_not_articles(self):
        self.assertEqual(
            [a.label for a in self.list],
            ["제1조", "제2조", "제16조", "제26조", "제35조", "제43조의2", "제53조", "제60조"],
        )

    def test_chapter_assigned_from_heading_rows(self):
        self.assertEqual(self.arts["제1조"].chapter, "제1장 총칙")
        self.assertEqual(self.arts["제16조"].chapter, "제2장 근로계약")
        self.assertEqual(self.arts["제60조"].chapter, "제2장 근로계약")

    def test_law_metadata(self):
        self.assertEqual(self.law["name"], "근로기준법")
        self.assertEqual(self.law["law_id"], "001872")
        self.assertEqual(self.law["ministry"], "고용노동부")

    def test_plain_article_text(self):
        a = self.arts["제1조"]
        self.assertTrue(a.text.startswith("이 법은"))
        self.assertEqual(a.title, "목적")
        self.assertEqual(a.paragraphs, [])

    def test_paragraph_article_and_tag_stripping(self):
        a = self.arts["제2조"]
        self.assertEqual(len(a.paragraphs), 2)
        self.assertEqual(len(a.paragraphs[0].items), 2)
        self.assertNotIn("<", a.text)
        self.assertIn("개정 2018.3.20, 2019.1.15", a.amendments)
        self.assertEqual(a.paragraphs[0].items[1].no, "9의2.")

    def test_ho_only_paragraph(self):
        a = self.arts["제26조"]
        self.assertEqual(len(a.paragraphs), 1)
        self.assertIsNone(a.paragraphs[0].no)
        self.assertEqual(len(a.paragraphs[0].items), 2)
        self.assertIn("해고하려면", a.head)
        self.assertEqual(a.title, "해고의 예고")
        self.assertIn("3개월 미만", a.text)
        self.assertNotIn("<", a.text)

    def test_branch_article(self):
        a = self.arts["제43조의2"]
        self.assertEqual((a.number, a.branch, a.key), (43, 2, "0043021"))

    def test_deleted_article(self):
        a = self.arts["제35조"]
        self.assertEqual(a.status, "deleted")
        self.assertEqual(chunk_article(self.law, a), [])

    def test_deleted_paragraph_removed_from_text(self):
        a = self.arts["제60조"]
        self.assertTrue(a.paragraphs[1].deleted)
        self.assertNotIn("삭제", a.text)
        self.assertEqual(a.status, "valid")

    def test_whole_article_expiry(self):
        a = self.arts["제16조"]
        self.assertEqual(a.status, "expired")
        self.assertEqual(a.expiry["scope"], "article")
        self.assertEqual(a.expiry["paragraphs"], [])
        self.assertEqual(chunk_article(self.law, a), [])

    def test_partial_expiry_marks_only_referenced_paragraphs(self):
        a = self.arts["제53조"]
        self.assertEqual(a.status, "partially_expired")
        self.assertEqual(sorted(p.number for p in a.paragraphs if p.expired), [3, 6])
        text = " ".join(c["text"] for c in chunk_article(self.law, a))
        self.assertIn("합의하면 1주 간에 12시간", text)
        self.assertNotIn("상시 30명 미만", text)
        self.assertNotIn("15세 이상 18세 미만", text)

    def test_expiry_not_reached_yet(self):
        _, arts = parse_law(SAMPLE, today=date(2022, 1, 1))
        a = {x.label: x for x in arts}["제53조"]
        self.assertEqual(a.status, "valid")
        self.assertFalse(a.expiry["expired"])

    def test_summary_counts(self):
        s = summarize(self.list)
        self.assertEqual(s["articles"], 8)
        self.assertEqual((s["deleted"], s["expired"], s["partially_expired"]), (1, 1, 1))
        self.assertEqual(s["chapters"], 2)
        self.assertEqual(s["expiry_notes"], 2)


def decree_with(*units):
    doc = copy.deepcopy(SAMPLE)
    doc["법령"]["기본정보"]["법령명_한글"] = "근로기준법 시행령"
    doc["법령"]["조문"]["조문단위"] = list(units)
    return doc


def note(target):
    return f"[대통령령 제31584호(2021.3.30) 부칙 제2조의 규정에 의하여 {target} 2022년 12월 31일까지 유효함]"


def paras(*nos):
    return [{"항번호": CIRCLED[n - 1], "항내용": f"{CIRCLED[n - 1]} 제{n}항 본문"} for n in nos]


class ExpiryScopeTests(unittest.TestCase):
    """조문참고자료의 만료 범위 판정. 문구는 실제 근로기준법 시행령(시행 2025-10-23)에서 가져옴."""

    def parse_one(self, u):
        law, (a,) = parse_law(decree_with(u), today=TODAY)
        return law, a

    def test_item_level_expiry_keeps_article(self):
        # 시행령 제8조의2: 호의 일부만 만료 -> 조를 지우면 안 된다
        _, a = self.parse_one(
            unit(
                8,
                "0008021",
                "제8조의2(근로자의 요구에 따른 서면 교부) 법 제17조제2항 단서에서 ... 말한다.",
                "근로자의 요구에 따른 서면 교부",
                조문가지번호="2",
                조문참고자료="[본조신설 2011.9.22]"
                + note("이 조 제1호의 개정규정 중 법 제53조제3항에 관한 부분은"),
                항={"호": [{"호번호": "1.", "호내용": "1. 법 제51조제2항, 제51조의2제1항, 제52조제1항, 제53조제3항에 따른 근로시간 변경"}]},
            )
        )
        self.assertEqual(a.status, "valid")
        self.assertEqual(a.expiry["scope"], "partial")
        self.assertTrue(a.expiry["expired"])

    def test_paragraph_item_expiry_keeps_paragraph(self):
        # 시행령 제22조: 제1항제8호의 일부만 만료 -> 제1항도 살아 있어야 한다
        law, a = self.parse_one(
            unit(
                22,
                "0022001",
                "제22조(보존 대상 서류 등)",
                "보존 대상 서류 등",
                조문참고자료=note("이 조 제1항제8호의 개정규정 중 법 제53조제3항에 관한 부분은"),
                항=paras(1, 2),
            )
        )
        self.assertEqual(a.status, "valid")
        self.assertEqual(a.expiry["scope"], "partial")
        self.assertFalse(any(p.expired for p in a.paragraphs))
        self.assertIn("제1항 본문", " ".join(c["text"] for c in chunk_article(law, a)))

    def test_branch_article_paragraph_expiry(self):
        _, a = self.parse_one(
            unit(43, "0043021", "제43조의2(가상)", "가상", 조문가지번호="2",
                 조문참고자료=note("제43조의2제2항의 개정규정은"), 항=paras(1, 2))
        )
        self.assertEqual(a.status, "partially_expired")
        self.assertEqual([p.number for p in a.paragraphs if p.expired], [2])

    def test_branch_reference_does_not_hit_main_article(self):
        _, a = self.parse_one(
            unit(43, "0043001", "제43조(가상)", "가상",
                 조문참고자료=note("제43조의2제1항의 개정규정은"), 항=paras(1, 2))
        )
        self.assertEqual(a.status, "valid")
        self.assertFalse(any(p.expired for p in a.paragraphs))

    def test_parent_law_reference_is_not_this_article(self):
        # 시행령 제53조의 문구 속 "법 제53조제3항"은 모법 조문이다
        _, a = self.parse_one(
            unit(53, "0053001", "제53조(가상)", "가상",
                 조문참고자료=note("이 조 제2항 중 법 제53조제3항에 관한 개정규정은"), 항=paras(1, 2, 3))
        )
        self.assertEqual(a.expiry["paragraphs"], [2])
        self.assertEqual([p.number for p in a.paragraphs if p.expired], [2])

    def test_note_without_reference_to_article_is_not_applied(self):
        _, a = self.parse_one(
            unit(30, "0030001", "제30조(가상) 본문", "가상", 조문참고자료=note("제29조의 개정규정은"))
        )
        self.assertEqual(a.status, "valid")
        self.assertEqual(a.expiry["scope"], "unknown")


class ChunkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.law, arts = parse_law(SAMPLE, today=TODAY)
        cls.arts = {a.label: a for a in arts}

    def test_chunk_starts_with_citation_header(self):
        (c,) = chunk_article(self.law, self.arts["제1조"])
        self.assertTrue(c["text"].startswith("근로기준법 제1조(목적)\n이 법은"))
        self.assertEqual(c["chunk_id"], "001872:0001001")
        self.assertEqual(c["chapter"], "제1장 총칙")

    def test_long_article_splits_by_paragraph(self):
        chunks = chunk_article(self.law, self.arts["제43조의2"], max_chars=60)
        self.assertEqual([c["paragraph_no"] for c in chunks], [1, 2])
        self.assertTrue(chunks[0]["citation"].endswith("제1항"))
        self.assertTrue(chunks[1]["text"].startswith("근로기준법 제43조의2(체불사업주 명단 공개) 제2항"))

    def test_short_article_stays_whole(self):
        chunks = chunk_article(self.law, self.arts["제43조의2"], max_chars=1000)
        self.assertEqual(len(chunks), 1)
        self.assertIsNone(chunks[0]["paragraph_no"])

    def test_deleted_paragraph_not_in_chunk(self):
        (c,) = chunk_article(self.law, self.arts["제60조"])
        self.assertNotIn("삭제", c["text"])
        self.assertIn("15일의 유급휴가", c["text"])


class RobustnessTests(unittest.TestCase):
    def test_single_dict_instead_of_list(self):
        doc = copy.deepcopy(SAMPLE)
        doc["법령"]["조문"]["조문단위"] = doc["법령"]["조문"]["조문단위"][1]  # 단일 객체
        _, arts = parse_law(doc, today=TODAY)
        self.assertEqual([a.label for a in arts], ["제1조"])

    def test_api_error_response_raises_with_message(self):
        err = {
            "result": "사용자 정보 검증에 실패하였습니다.",
            "msg": "OPEN API 호출 시 사용자 검증을 위하여 정확한 서버장비의 IP주소 및 도메인주소를 등록해 주세요.",
        }
        with self.assertRaises(RuntimeError) as ctx:
            parse_law(err)
        self.assertIn("IP주소", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
