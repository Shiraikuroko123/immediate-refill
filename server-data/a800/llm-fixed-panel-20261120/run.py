"""Measure live finite-pool scheduling against a local OpenAI-compatible server."""
import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import hashlib
import http.client
import json
from pathlib import Path
import platform
import random
import statistics
import sys
import time
import traceback
from urllib.parse import urlsplit

from workload import parse_answer, path_seed, questions


def sha(text):
    return hashlib.sha256(text.encode("utf8")).hexdigest()


def dump(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf8")


def log(path, data):
    with path.open("a", encoding="utf8") as file:
        file.write(json.dumps(data, ensure_ascii=False) + "\n")


def record_failed_panel(args, key, start, paths, active, errors, events):
    """Retain observed and outstanding calls from an invalid trial before raising."""
    for task, index in active.items():
        try:
            paths[index] = {**task.result(), "used_at_decision": False}
        except Exception as exc:
            errors.append(dict(index=index, error=repr(exc)))
    for index, result in sorted(paths.items()):
        log(args.out/"partial_paths.jsonl", {**key, "index":index,
            "arrival_s":result["arrival"]-start, **result})
    log(args.out/"failed_panels.jsonl", {**key, "errors":errors, "events":events})


def counts(labels):
    ranked = sorted(Counter(x for x in labels if x is not None).items(),
                    key=lambda item: (-item[1], item[0]))
    if not ranked:
        return None, 0
    return ranked[0][0], ranked[0][1] - (ranked[1][1] if len(ranked) > 1 else 0)


def stopping_prefix(votes):
    for k in range(1, len(votes) + 1):
        if counts(votes[:k])[1] > len(votes) - k:
            return k
    return len(votes)


def request(args, prompt, seed, token_limit):
    payload = {"model": args.model, "seed": seed, "max_tokens": token_limit,
               "temperature": args.temperature, "top_p": 1.0, "n": 1, "stream": False}
    if args.api_mode == "chat":
        payload["messages"] = [{"role": "user", "content": prompt}]
        if args.no_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        route = "/chat/completions"
    else:
        payload["prompt"] = prompt
        route = "/completions"
    address = urlsplit(args.base_url)
    conn = http.client.HTTPConnection(address.hostname, address.port, timeout=args.timeout)
    stamp = time.monotonic()
    try:
        conn.request("POST", address.path.rstrip("/") + route, body=json.dumps(payload),
                     headers={"Content-Type": "application/json"})
        response = conn.getresponse()
        raw = response.read()
        arrival = time.monotonic()
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {raw[:1000]!r}")
        result = json.loads(raw)
        choice = result["choices"][0]
        text = ((choice["message"].get("content") or "") if args.api_mode == "chat"
                else choice["text"])
        message = choice.get("message", {})
        reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
        usage = result.get("usage") or {}
        tokens = usage.get("completion_tokens")
        if type(tokens) is not int or tokens < 0:
            raise ValueError("Server omitted a valid usage.completion_tokens")
        reason = choice.get("finish_reason")
        return dict(seed=seed, request=payload, raw_response=result, http_start=stamp,
                    arrival=arrival, request_s=arrival-stamp, completion_tokens=tokens,
                    prompt_tokens=usage.get("prompt_tokens"), text_hash=sha(text),
                    generated_text_hash=sha(json.dumps([reasoning, text], ensure_ascii=False)),
                    answer=parse_answer(text, reason), finish_reason=reason)
    finally:
        conn.close()


def run_panel(args, question, rep, policy, pool):
    key = dict(qid=question["qid"], repeat=rep, policy=policy)
    warm = request(args, question["prompt"], path_seed(key["qid"], rep, -1), 1)
    log(args.out / "warmups.jsonl", {**key, **warm})
    start = time.monotonic()
    active, paths, answers, events = {}, {}, [], []
    committed = launched = 0
    while True:
        candidate, gap = counts(answers)
        remaining = args.n - len(answers)
        if not remaining or (policy != "full" and gap > remaining):
            decision = time.monotonic()
            break
        distance = min(remaining, max(0, (remaining - gap) // 2 + 1))
        if policy == "rolling":
            if len(answers) == committed:
                committed += distance
            width = min(args.capacity, committed-len(answers))
        elif policy == "refill":
            width = min(args.capacity, distance)
        else:
            width = args.capacity
        before = launched
        while len(active) < width and launched < args.n:
            seed = path_seed(key["qid"], rep, launched)
            task = pool.submit(request, args, question["prompt"], seed, args.max_tokens)
            active[task] = launched
            launched += 1
        events.append(dict(at_s=time.monotonic()-start, completed=len(answers), gap=gap,
                           width=width, active=len(active), launched_ids=list(range(before,launched))))
        ready, _ = wait(active, return_when=FIRST_COMPLETED)
        finished, errors = [], []
        for task in ready:
            index = active.pop(task)
            try:
                finished.append((index, task.result()))
            except Exception as exc:
                errors.append(dict(index=index, error=repr(exc)))
        for idx, result in sorted(finished, key=lambda pair: (pair[1]["arrival"], pair[0])):
            paths[idx] = {**result, "used_at_decision": True}
            answers.append(result["answer"])
        if errors:
            record_failed_panel(args, key, start, paths, active, errors, events)
            raise RuntimeError(f"Failed live panel {key}: see failed_panels.jsonl")
    at_lock = len(active)
    if policy in ("refill","rolling") and at_lock:
        raise AssertionError("Safe admission left paths outstanding at the lock")
    # Leave outstanding fixed-width calls on the server to obtain actual usage.
    for task, idx in active.items():
        paths[idx] = {**task.result(), "used_at_decision": False}
    drained = time.monotonic()
    for idx, row in sorted(paths.items()):
        log(args.out / "paths.jsonl", {**key, "index": idx, "arrival_s":row["arrival"]-start,
                                       "launch_s":row["http_start"]-start, **row})
    for row in events:
        log(args.out / "admission.jsonl", {**key, **row})
    summary = dict(**key, decision_s=decision-start, drained_s=drained-start,
                   winner=candidate, truth=question["truth"], correct=candidate==question["truth"],
                   completed_at_decision=len(answers), outstanding_at_decision=at_lock,
                   launched=launched, abstentions_at_decision=answers.count(None),
                   generation_tokens=sum(r["completion_tokens"] for r in paths.values()),
                   tokens_of_completed_requests=sum(r["completion_tokens"] for r in paths.values()
                                                    if r["used_at_decision"]),
                   generation_tokens_at_decision=None if at_lock else
                   sum(r["completion_tokens"] for r in paths.values()))
    log(args.out / "panels.jsonl", summary)
    print(f"q={key['qid']} rep={rep} policy={policy} s={summary['decision_s']:.3f} "
          f"paths={launched} winner={candidate}", flush=True)
    return summary, paths


def compare(args, group):
    full, baseline = group["full"]
    seq = [baseline[i]["answer"] for i in range(args.n)]
    prefix = stopping_prefix(seq)
    records = []
    for policy, (panel, paths) in group.items():
        by_index = []
        for idx, result in sorted(paths.items()):
            ref = baseline[idx]
            by_index.append(dict(index=idx,
                text_matches=result["text_hash"]==ref["text_hash"],
                generation_matches=result["generated_text_hash"]==ref["generated_text_hash"],
                vote_matches=result["answer"]==ref["answer"],
                tokens_match=result["completion_tokens"]==ref["completion_tokens"],
                finish_matches=result["finish_reason"]==ref["finish_reason"]))
        row = dict(qid=panel["qid"], repeat=panel["repeat"], policy=policy,
                   winner_agrees=panel["winner"]==full["winner"], full_k=prefix,
                   launched_matches_full_prefix=list(sorted(paths))==list(range(prefix)),
                   texts_mismatched=sum(not x["text_matches"] for x in by_index),
                   generated_texts_mismatched=sum(not x["generation_matches"] for x in by_index),
                   votes_mismatched=sum(not x["vote_matches"] for x in by_index),
                   lengths_mismatched=sum(not x["tokens_match"] for x in by_index),
                   checked_paths=len(paths), per_seed=by_index)
        records.append(row)
        log(args.out / "coupling.jsonl", row)
    return records


def main(args):
    modes = ["rolling", "refill", "full"] + (["fixed"] if args.fixed else [])
    prompts = questions(args.questions)
    random.Random(20261120).shuffle(prompts)
    args.out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parent
    metadata = dict(arguments={**vars(args), "out": str(args.out)}, python=sys.version,
                    platform=platform.platform(), started_unix=time.time(),
                    code={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [root/"run.py", root/"workload.py", root/"PROTOCOL.md"]})
    dump(args.out / "metadata.json", metadata)
    dump(args.out / "questions.json", prompts)
    trials = []
    try:
        with ThreadPoolExecutor(args.capacity) as pool:
            for rep in range(args.repeats):
                for number, question in enumerate(prompts):
                    cycle = rep * len(prompts) + number
                    offset = cycle % len(modes)
                    base = modes if (cycle//len(modes)) % 2 == 0 else modes[::-1]
                    order = base[offset:] + base[:offset]
                    log(args.out/"order.jsonl", dict(qid=question["qid"],repeat=rep,order=order))
                    group = {policy: run_panel(args, question, rep, policy, pool) for policy in order}
                    compare(args, group)
                    roll = group["rolling"][0]
                    refill = group["refill"][0]
                    trials.append(dict(qid=question["qid"],repeat=rep,
                        refill_s=refill["decision_s"], rolling_s=roll["decision_s"],
                        refill_tokens=refill["generation_tokens"],rolling_tokens=roll["generation_tokens"]))
        per_question = []
        for q in prompts:
            block = [r for r in trials if r["qid"] == q["qid"]]
            time_ratio = sum(r["refill_s"] for r in block)/sum(r["rolling_s"] for r in block)
            reference_tokens = sum(r["rolling_tokens"] for r in block)
            per_question.append(dict(qid=q["qid"],time_savings=1-time_ratio,
                token_savings=(1-sum(r["refill_tokens"] for r in block)/reference_tokens
                               if reference_tokens else None)))
        dump(args.out/"summary.json", dict(paired=trials, per_question=per_question,
            question_macro_time_savings=statistics.mean(r["time_savings"] for r in per_question)))
    except Exception:
        (args.out/"failure.txt").write_text(traceback.format_exc(),encoding="utf8")
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--api-mode", choices=["chat","completion"], default="chat")
    p.add_argument("--questions", type=int, choices=range(1,9), default=6)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--n", type=int,default=8)
    p.add_argument("--capacity", type=int,default=4)
    p.add_argument("--max-tokens", type=int,default=160)
    p.add_argument("--temperature", type=float,default=0.7)
    p.add_argument("--timeout", type=int,default=300)
    p.add_argument("--fixed", action="store_true")
    p.add_argument("--no-thinking", action="store_true")
    p.add_argument("--server-config", default="unspecified",
                   help="Describe backend version, model revision, GPU, precision and cache configuration")
    args = p.parse_args()
    address = urlsplit(args.base_url)
    if address.scheme != "http" or address.hostname not in {"127.0.0.1","localhost","::1"}:
        p.error("Use a local HTTP inference server or an SSH-forwarded loopback endpoint")
    if min(args.n,args.capacity,args.max_tokens,args.repeats,args.timeout)<1 or args.capacity>args.n:
        p.error("Choose positive limits and capacity <= n")
    if args.no_thinking and args.api_mode != "chat":
        p.error("--no-thinking uses chat_template_kwargs on chat requests")
    main(args)
