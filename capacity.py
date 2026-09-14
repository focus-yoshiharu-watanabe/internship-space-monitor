"""
定員カウント担当のファイル

ゾーンの中の人数が定員を超えたら、警告を出す。
"""

CAPACITY = 1  # 定員。2人で決めた値にする


def judge_capacity(count, now, state):
    """
    count: ゾーンの中の人数（common.py の count_people が数えた値）
    now:   今の時刻（秒）。1秒たつと 1 増える
    state: 前のフレームから覚えておきたいことを入れておける辞書（最初は空の {}）

    警告を出すときは、2人で約束した形の辞書を返す。例：
        return {"rule": "capacity", "level": "warn", "message": "定員オーバー"}
    警告を出さないときは None を返す。
    """
    # TODO: ここに自分のルールを書く
    return None
