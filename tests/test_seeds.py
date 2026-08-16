"""Pure logic: seed progression, prompt expansion, filename tokens."""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from krea2_batchfix.seeds import (  # noqa: E402
    PromptShapeError,
    expand_prompts,
    iter_logical_positions,
    logical_index,
    rewrite_filename_pattern,
    seed_sequence,
    subseed_sequence,
    total_outputs,
)


class SeedMathTests(unittest.TestCase):
    def test_total_outputs(self):
        self.assertEqual(total_outputs(3, 2), 6)
        self.assertEqual(total_outputs(1, 1), 1)

    def test_logical_index_is_count_major(self):
        self.assertEqual(logical_index(0, 0, 2), 0)
        self.assertEqual(logical_index(0, 1, 2), 1)
        self.assertEqual(logical_index(1, 0, 2), 2)
        self.assertEqual(logical_index(2, 1, 2), 5)

    def test_iteration_order_matches_forge(self):
        positions = list(iter_logical_positions(3, 2))
        self.assertEqual([p[0] for p in positions], [0, 1, 2, 3, 4, 5])
        self.assertEqual([(p[1], p[2]) for p in positions], [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)])

    def test_seed_progression(self):
        """Forge: all_seeds = [seed + x for x in range(batch_size * n_iter)]"""
        self.assertEqual(seed_sequence(1000, 3, 2, 0), [1000, 1001, 1002, 1003, 1004, 1005])

    def test_seed_progression_with_variation_seed(self):
        """Forge holds the seed constant when subseed_strength is non-zero."""
        self.assertEqual(seed_sequence(1000, 2, 2, 0.5), [1000, 1000, 1000, 1000])

    def test_subseed_progression_always_increments(self):
        self.assertEqual(subseed_sequence(7, 2, 2), [7, 8, 9, 10])


class PromptExpansionTests(unittest.TestCase):
    def test_scalar_prompts(self):
        prompts, negatives = expand_prompts("a cat", "blurry", 4)
        self.assertEqual(prompts, ["a cat"] * 4)
        self.assertEqual(negatives, ["blurry"] * 4)

    def test_prompt_list_is_preserved_per_output(self):
        prompts, negatives = expand_prompts(["a", "b", "c"], "neg", 3)
        self.assertEqual(prompts, ["a", "b", "c"])
        self.assertEqual(negatives, ["neg"] * 3)

    def test_negative_list_drives_length(self):
        prompts, negatives = expand_prompts("a", ["n1", "n2"], 2)
        self.assertEqual(prompts, ["a", "a"])
        self.assertEqual(negatives, ["n1", "n2"])

    def test_length_mismatch_is_rejected(self):
        with self.assertRaises(PromptShapeError):
            expand_prompts(["a", "b"], "neg", 6)

    def test_prompt_and_negative_mismatch_is_rejected(self):
        with self.assertRaises(PromptShapeError):
            expand_prompts(["a", "b", "c"], ["n1", "n2"], 3)


class FilenameTokenTests(unittest.TestCase):
    def test_generation_number_is_frozen_per_logical_output(self):
        pattern = "source-[generation_number]"
        self.assertEqual(rewrite_filename_pattern(pattern, 3, 2, 0, 0), "source-1")
        self.assertEqual(rewrite_filename_pattern(pattern, 3, 2, 0, 1), "source-2")
        self.assertEqual(rewrite_filename_pattern(pattern, 3, 2, 2, 1), "source-6")

    def test_batch_number_is_frozen_only_when_batch_size_exceeds_one(self):
        self.assertEqual(rewrite_filename_pattern("x-[batch_number]", 2, 2, 1, 1), "x-2")
        self.assertEqual(rewrite_filename_pattern("x-[batch_number]", 4, 1, 0, 0), "x-[batch_number]")

    def test_single_output_request_is_left_to_forge(self):
        self.assertEqual(rewrite_filename_pattern("x-[generation_number]", 1, 1, 0, 0), "x-[generation_number]")

    def test_empty_pattern(self):
        self.assertEqual(rewrite_filename_pattern("", 3, 2, 0, 0), "")


if __name__ == "__main__":
    unittest.main()
