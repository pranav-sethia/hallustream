import json

import numpy as np
import pandas as pd

from hallustream.research.factcheck import judge, topics
from hallustream.research.factcheck.combine import LAYERS


def test_category_takes_the_first_matching_group():
    assert topics.category(["village", "human settlement"]) == "place"
    assert topics.category(["taxon"]) == "living_thing"
    assert topics.category(["association football club"]) == "organization"
    assert topics.category(["sports season", "football club"]) == "event"
    assert topics.category(["studio album"]) == "creative_work"
    assert topics.category(["chemical compound"]) is None
    assert topics.category(["display technology"]) is None  # "play" only as a whole word


def test_pick_is_even_fixed_and_respects_exclusions():
    items = {}
    for c, words in topics.CATEGORIES.items():
        for b, links in enumerate((2, 5, 20, 50)):
            for i in range(4):
                items[f"{c}{b}{i} (x)"] = {"sitelinks": links, "classes": [words[0]]}
    picked = topics.pick(items, exclude_text="about place00 here", per_cell=3)
    assert len(picked) == len(topics.CATEGORIES) * 4 * 3
    assert "place00 (x)" not in {p["name"] for p in picked}
    assert picked == topics.pick(items, exclude_text="about place00 here", per_cell=3)


def test_topic_prompts_fill_and_mention_no_person():
    req = judge.bio_request("k", "Lake Tahoe", ["Lake Tahoe is a lake."], "It is big.",
                            "t1_minimal", "topic_judge")  # fmt: skip
    text = req["request"]["contents"][0]["parts"][0]["text"]
    assert "write about Lake Tahoe" in text and "<text>\nIt is big.\n</text>" in text
    assert "biography of Lake Tahoe" not in text
    assert not any(slot in text for slot in ("{name}", "{passages}", "{biography}"))
    audit = judge.audit_request("k", "Lake Tahoe", ["p"], "It is big.", [("big", "It is big.")],
                                "t1_minimal", "topic_audit")  # fmt: skip
    assert "aspect of the subject" in audit["request"]["contents"][0]["parts"][0]["text"]


def test_evaluate_scores_every_detector(tmp_path):
    rng = np.random.default_rng(0)
    n = 80
    kind = np.where(rng.random(n) < 0.5, "not_supported", "supported")
    bad = kind == "not_supported"
    df = pd.DataFrame({"name": [f"s{i % 8}" for i in range(n)], "kind": kind,
                       "entropy_max": rng.random(n) + bad, "nll_max": rng.random(n),
                       **{f"p_seed{k}": np.clip(0.3 + 0.4 * bad + 0.1 * rng.random(n), 0, 1)
                          for k in range(5)}})  # fmt: skip
    df.to_parquet(tmp_path / "spans.parquet")
    np.save(tmp_path / "hidden.npy", rng.normal(size=(n, len(LAYERS), 2560)).astype(np.float16))
    subjects = [{"name": f"s{i}", "category": "place" if i < 4 else "event"} for i in range(8)]
    out = topics.evaluate(tmp_path, subjects, n_boot=50)
    assert out["all"]["entities"] == n and out["all"]["v1"] > 0.9
    assert out["by_category"]["place"]["subjects"] == 4
    assert {"v3-max_entropy", "v1-max_entropy", "v3-v1"} <= out.keys()
    json.dumps(out)
