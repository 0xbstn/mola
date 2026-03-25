#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


DETACHED_BATCH_BLOCK = '''

@dataclass
class DetachedBatch:
    batch: Batch

    @property
    def uids(self):
        return self.batch.uids
'''.rstrip()


STEP_HELPERS_BLOCK = '''

    def _allocate_uids(self, count, existing_uids=None):
        existing_uids = set(existing_uids or [])
        if self.active_batch is not None:
            existing_uids.update(self.active_batch.uids)
        next_uid = max(
            self.uid_count,
            (max(existing_uids) + 1) if existing_uids else 0,
        )
        uids = list(range(next_uid, next_uid + count))
        self.uid_count = next_uid + count
        return uids

    def _step_live_batch(self, batch):
        tic = time.perf_counter()

        y, logprobs = batch.y, batch.logprobs
        for i, toks in enumerate(batch.tokens):
            batch.tokens[i] = mx.concatenate((toks, y[i : i + 1]))
        batch.y, batch.logprobs = self._step(
            y[:, None],
            batch.cache,
            batch.samplers,
            batch.logits_processors,
            batch.tokens,
        )

        mx.async_eval(batch.y, batch.logprobs, batch.tokens)

        y = y.tolist()
        toc = time.perf_counter()
        elapsed = toc - tic
        self._stats.generation_time += elapsed
        keep_idx = []
        end_idx = []
        responses = []

        for e, (t, uid, num_tok, max_tok) in enumerate(
            zip(y, batch.uids, batch.num_tokens, batch.max_tokens)
        ):
            cache = None
            num_tok += 1
            batch.num_tokens[e] = num_tok
            if t in self.stop_tokens:
                finish_reason = "stop"
                end_idx.append(e)
            elif num_tok >= max_tok:
                finish_reason = "length"
                end_idx.append(e)
            else:
                finish_reason = None
                keep_idx.append(e)
            if finish_reason is not None:
                cache = batch.extract_cache(e)
            responses.append(self.Response(uid, t, logprobs[e], finish_reason, cache))

        next_batch = batch
        if len(end_idx):
            if len(keep_idx) > 0:
                batch.filter(keep_idx)
            else:
                next_batch = None

        self._next_count += 1
        if self._next_count % 512 == 0:
            mx.clear_cache()
        self._stats.generation_tokens += len(responses)
        return next_batch, responses, elapsed
'''.rstrip()


NEXT_AND_DETACHED_BLOCK = '''

    def _next(self):
        tic = time.perf_counter()

        prompt_processing = False
        batch = self.active_batch
        num_active = len(batch) if batch else 0
        num_to_add = self.completion_batch_size - num_active
        while num_to_add >= self.prefill_batch_size:
            prompts = self.unprocessed_prompts[: self.prefill_batch_size]
            # Finish processing the last examples of the last batch
            if len(prompts) == 0 and num_active > 0:
                break
            # No more prompts and no more completions, all done
            elif len(prompts) == 0:
                self.active_batch = None
                return []
            # Process prompts
            if batch is not None and not prompt_processing:
                # Finish any active completion tokens
                mx.eval(batch.y, batch.logprobs)
                self._stats.generation_time += time.perf_counter() - tic
                tic = time.perf_counter()

            batch = self._process_prompts(prompts)
            self.unprocessed_prompts = self.unprocessed_prompts[
                self.prefill_batch_size :
            ]
            prompt_processing = True
            # If there was no active batch, set it
            if self.active_batch is None:
                self.active_batch = batch
            else:
                self.active_batch.extend(batch)

            num_active = len(self.active_batch)
            num_to_add -= len(batch)

        batch = self.active_batch
        generation_time_before = self._stats.generation_time
        batch, responses, elapsed = self._step_live_batch(batch)
        if prompt_processing:
            self._stats.prompt_time += elapsed
            self._stats.generation_time = generation_time_before
        self.active_batch = batch
        return responses

    def detach_active_batch(self):
        if self.active_batch is None:
            return None
        batch = self.active_batch
        self.active_batch = None
        return DetachedBatch(batch)

    def restore_detached_batch(self, batch):
        if self.active_batch is not None:
            raise RuntimeError("cannot restore detached batch while active_batch is populated")
        self.active_batch = batch.batch

    def snapshot_detached_batch(self, batch):
        return batch.batch

    def step_detached_batch(self, batch):
        with mx.stream(generation_stream):
            next_batch, responses, _elapsed = self._step_live_batch(batch.batch)
        return (DetachedBatch(next_batch) if next_batch is not None else None), responses

    def promote_detached_batch(self, dst, src):
        src_batch = src.batch
        remapped_uids = self._allocate_uids(
            len(src_batch.uids),
            existing_uids=dst.uids if dst is not None else (),
        )
        src_batch.uids = remapped_uids
        if dst is None:
            merged = src_batch
        else:
            dst.batch.extend(src_batch)
            merged = dst.batch
        return DetachedBatch(merged), remapped_uids

    def next(self):
        with mx.stream(generation_stream):
            return self._next()
'''.rstrip()


ORIGINAL_NEXT_BLOCK = '''
    def _next(self):
        tic = time.perf_counter()

        prompt_processing = False
        batch = self.active_batch
        num_active = len(batch) if batch else 0
        num_to_add = self.completion_batch_size - num_active
        while num_to_add >= self.prefill_batch_size:
            prompts = self.unprocessed_prompts[: self.prefill_batch_size]
            # Finish processing the last examples of the last batch
            if len(prompts) == 0 and num_active > 0:
                break
            # No more prompts and no more completions, all done
            elif len(prompts) == 0:
                self.active_batch = None
                return []
            # Process prompts
            if batch is not None and not prompt_processing:
                # Finish any active completion tokens
                mx.eval(batch.y, batch.logprobs)
                self._stats.generation_time += time.perf_counter() - tic
                tic = time.perf_counter()

            batch = self._process_prompts(prompts)
            self.unprocessed_prompts = self.unprocessed_prompts[
                self.prefill_batch_size :
            ]
            prompt_processing = True
            # If there was no active batch, set it
            if self.active_batch is None:
                self.active_batch = batch
            else:
                self.active_batch.extend(batch)

            num_active = len(self.active_batch)
            num_to_add -= len(batch)

        batch = self.active_batch
        y, logprobs = batch.y, batch.logprobs
        for i, toks in enumerate(batch.tokens):
            batch.tokens[i] = mx.concatenate((toks, y[i : i + 1]))
        batch.y, batch.logprobs = self._step(
            y[:, None],
            batch.cache,
            batch.samplers,
            batch.logits_processors,
            batch.tokens,
        )

        mx.async_eval(batch.y, batch.logprobs, batch.tokens)

        y = y.tolist()
        toc = time.perf_counter()
        if prompt_processing:
            self._stats.prompt_time += toc - tic
        else:
            self._stats.generation_time += toc - tic
        keep_idx = []
        end_idx = []
        responses = []

        for e, (t, uid, num_tok, max_tok) in enumerate(
            zip(y, batch.uids, batch.num_tokens, batch.max_tokens)
        ):
            cache = None
            num_tok += 1
            batch.num_tokens[e] = num_tok
            if t in self.stop_tokens:
                finish_reason = "stop"
                end_idx.append(e)
            elif num_tok >= max_tok:
                finish_reason = "length"
                end_idx.append(e)
            else:
                finish_reason = None
                keep_idx.append(e)
            if finish_reason is not None:
                cache = batch.extract_cache(e)
            responses.append(self.Response(uid, t, logprobs[e], finish_reason, cache))

        # Remove any finished completions
        if len(end_idx):
            if len(keep_idx) > 0:
                batch.filter(keep_idx)
            else:
                self.active_batch = None

        self._next_count += 1
        if self._next_count % 512 == 0:
            mx.clear_cache()
        self._stats.generation_tokens += len(responses)
        return responses

    def next(self):
        with mx.stream(generation_stream):
            return self._next()
'''.rstrip()


def patch_generate_py(path: Path) -> bool:
    text = path.read_text()
    if "class DetachedBatch:" in text and "def detach_active_batch(self):" in text:
        return False

    batch_anchor = """    def extract_cache(self, idx):\n        return [c.extract(idx) for c in self.cache]\n"""
    if batch_anchor not in text:
        raise RuntimeError("Batch.extract_cache anchor not found")
    text = text.replace(batch_anchor, batch_anchor + DETACHED_BATCH_BLOCK + "\n", 1)

    step_anchor = """        return sampled, list(logprobs)\n"""
    if step_anchor not in text:
        raise RuntimeError("BatchGenerator._step anchor not found")
    text = text.replace(step_anchor, step_anchor + STEP_HELPERS_BLOCK + "\n", 1)

    if ORIGINAL_NEXT_BLOCK not in text:
        raise RuntimeError("BatchGenerator._next block anchor not found")
    text = text.replace(ORIGINAL_NEXT_BLOCK, NEXT_AND_DETACHED_BLOCK, 1)

    backup = path.with_suffix(path.suffix + ".bak_detached_batch")
    if not backup.exists():
        backup.write_text(path.read_text())
    path.write_text(text)
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        type=Path,
        default=Path(".venv/lib/python3.13/site-packages/mlx_lm/generate.py"),
    )
    args = parser.parse_args()
    changed = patch_generate_py(args.target)
    if changed:
        print(f"Patched mlx_lm detached-batch API in {args.target}")
    else:
        print(f"mlx_lm detached-batch API already present in {args.target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
