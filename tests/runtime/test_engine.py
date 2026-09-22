"""批量推理正确性: 整批 decode 的结果必须与逐条单发一致."""

import numpy as np
import pytest

from minicpm_jev import BatchEngine, EngineConfig
from minicpm_jev.labels import BOOL_LABEL_SPECS, LabelSet, resolve_labels
from tests.conftest import MODEL_PATH

# 批形状会带来全词表 logits 的整体偏移, 但决策相关的受限概率漂移只有 ~0.025,
# 这里量的是决策量, 容差取实测上限的两倍.
PROB_TOL = 0.05

SYSTEM = "你是猫猫决策引擎, 只回答 yes 或 no。"
QUESTIONS = ("天空是蓝色的吗?", "1+1=2 吗?", "猫会飞吗?", "水是湿的吗?")


def _sequence(engine: BatchEngine, question: str) -> list[int]:
    """渲染一条问答序列, 决策位落在 assistant 头收尾处."""
    return engine.render_tokens(
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"问题: {question}"},
        ],
        assistant_prefix="答案:",
    )


@pytest.fixture(scope="module")
def labels(engine: BatchEngine) -> LabelSet:
    return resolve_labels(
        lambda text: engine.tokenize(text, add_bos=False), BOOL_LABEL_SPECS
    )


def test_batch_argmax_matches_single(engine: BatchEngine) -> None:
    """整批 decode 与逐条单发的 argmax 必须一致."""
    sequences = [_sequence(engine, question) for question in QUESTIONS]
    batched = engine.decision_logits(sequences)
    alone = [engine.decision_logits([sequence])[0] for sequence in sequences]
    for index, (batch_vector, single_vector) in enumerate(zip(batched, alone)):
        assert int(np.argmax(batch_vector)) == int(np.argmax(single_vector)), (
            f"第 {index} 条序列批量与单发 argmax 不一致"
        )




def test_duplicate_sequences_share_one_computation(engine: BatchEngine) -> None:
    """完全相同的序列按哈希去重, 返回逐位相同的结果."""
    sequence = _sequence(engine, "1+1=2 吗?")
    logits = engine.decision_logits([sequence, sequence, sequence])
    assert len(logits) == 3
    assert np.array_equal(logits[0], logits[1])
    assert np.array_equal(logits[1], logits[2])


def test_chunked_decode_matches_large_batch(engine: BatchEngine, labels: LabelSet) -> None:
    """把 n_batch 压到 16 强制分块, 决策量仍要与逐条单发一致."""
    sequences = [_sequence(engine, question) for question in QUESTIONS[:3]]
    reference = [engine.score([sequence], labels)[0] for sequence in sequences]
    small_config = EngineConfig(
        model_path=MODEL_PATH, n_ctx=1024, n_batch=16, n_ubatch=16, n_seq_max=8
    )
    with BatchEngine(small_config) as chunked_engine:
        chunked = chunked_engine.score(sequences, labels)
    for index, (chunked_scores, single_scores) in enumerate(zip(chunked, reference)):
        for name, value in chunked_scores.items():
            assert abs(value - single_scores[name]) <= PROB_TOL, (
                f"第 {index} 条序列分块后 {name} 概率 {value} 与单发 {single_scores[name]} 差超限"
            )


def test_tokenize_label_uses_label_contract(engine: BatchEngine) -> None:
    """标签口径 = 不带 BOS、不解析特殊标记; 标签必须落在单个 token 上."""
    assert engine.tokenize_label("12") == engine.tokenize(
        "12", add_bos=False, special=False
    )
    assert engine.tokenize_label("None") == engine.tokenize(
        "None", add_bos=False, special=False
    )
    for text in ("0", "9", "12", "127", "None"):
        assert len(engine.tokenize_label(text)) == 1, text




