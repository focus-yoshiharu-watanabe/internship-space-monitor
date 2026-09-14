"""
離席検知担当のファイル

ゾーンの中に誰もいない状態が一定時間続いたら、通知を出す。
"""

ABSENCE_SECONDS = 10  # 何秒いなかったら離席とするか。2人で決めた値にする


def judge_absence(count, now, state):
    """
    count: ゾーンの中の人数（common.py の count_people が数えた値）
    now:   今の時刻（秒）。1秒たつと 1 増える
    state: 前のフレームから覚えておきたいことを入れておける辞書（最初は空の {}）

    通知を出すときは、2人で約束した形の辞書を返す。例：
        return {"rule": "absence", "level": "info", "message": "離席中"}
    通知を出さないときは None を返す。

    ヒント：「0人になった時刻」を state に覚えておくと、
          now との差で「何秒いないか」が分かる。
    """
    # TODO: ここに自分のルールを書く
    return None
