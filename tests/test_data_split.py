from hallustream.research.data import Record, make_splits, normalize_question, question_key


def rec(i, source, split, question):
    return Record(
        answer_id=f"x/{source}/{split}/{i}",
        dataset="x",
        source=source,
        orig_split=split,
        subset="s",
        messages=[{"role": "user", "content": question}],
        answer="a",
    )


def test_normalize_question():
    assert normalize_question("  Tell me   about\nAda ") == "tell me about ada"
    assert question_key("Tell me about Ada") == question_key("tell  me about ada")


def test_no_test_question_in_train_or_val():
    records = [
        rec(0, "A", "train", "Q1"),
        rec(1, "A", "train", "q1 "),  # same question as a test item, different spacing
        rec(2, "A", "train", "Q2"),
        rec(3, "A", "validation", "Q3"),
        rec(4, "A", "validation", "Q4"),  # Q4 also in test
        rec(5, "B", "test", "Q1"),
        rec(6, "B", "test", "Q4"),
        rec(7, "C", "train", "Q5"),  # source not used for training
    ]
    res = make_splits(records, train_sources=["A"])
    by_id = {r.answer_id: r for r in records}
    test_keys = {question_key(by_id[i].question) for i in res.test}
    for i in res.train + res.val:
        assert question_key(by_id[i].question) not in test_keys
    assert res.train == ["x/A/train/2"]
    assert res.val == ["x/A/validation/3"]
    assert len(res.test) == 2
    assert res.stats["train_removed_frac"] == round(2 / 3, 4)


def test_question_in_train_and_val_goes_to_val():
    records = [
        rec(0, "A", "train", "Q1"),
        rec(1, "A", "validation", "Q1"),
        rec(2, "A", "test", "Z"),
    ]
    res = make_splits(records, train_sources=["A"])
    assert res.train == [] and res.val == ["x/A/validation/1"]
