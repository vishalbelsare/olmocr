import random
import re
import string
import time
import unittest
from typing import Literal

class RepeatDetector:
    def __init__(self, max_ngram_size: int = 10):
        self.max_ngram_size = max_ngram_size
        self.data = ""

    def add_letters(self, new_str: str):
        self.data += new_str

    def ngram_repeats(self) -> list[int]:
        result = [0] * self.max_ngram_size

        if not self.data:
            return result

        # Normalize all whitespace to single spaces
        text = re.sub(r"\s+", " ", self.data)

        # For each n-gram size
        for size in range(1, self.max_ngram_size + 1):
            if len(text) < size:
                continue

            # Get the last n-gram
            target = text[-size:]

            # Count backwards from the end to find repeats
            count = 0
            pos = len(text) - size  # Start position for previous n-gram

            while pos >= 0:
                if text[pos : pos + size] == target:
                    count += 1
                    pos -= size  # Move back by the size of the n-gram
                else:
                    break

            result[size - 1] = count

        return result


class RepeatDetectorTest(unittest.TestCase):
    def test_basicTest1(self):
        d = RepeatDetector(max_ngram_size=3)
        d.add_letters("a")
        self.assertEqual(d.ngram_repeats(), [1, 0, 0])

    def test_basicTest2(self):
        d = RepeatDetector(max_ngram_size=3)
        d.add_letters("abab")
        self.assertEqual(d.ngram_repeats(), [1, 2, 1])

    def test_longer_sequence(self):
        d = RepeatDetector(max_ngram_size=3)
        d.add_letters("aabaabaa")
        self.assertEqual(d.ngram_repeats(), [2, 1, 2])

    def test_no_repeats(self):
        d = RepeatDetector(max_ngram_size=3)
        d.add_letters("abc")
        self.assertEqual(d.ngram_repeats(), [1, 1, 1])

    def test_empty_data(self):
        d = RepeatDetector(max_ngram_size=3)
        self.assertEqual(d.ngram_repeats(), [0, 0, 0])

    def test_max_ngram_greater_than_data_length(self):
        d = RepeatDetector(max_ngram_size=5)
        d.add_letters("abc")
        self.assertEqual(d.ngram_repeats(), [1, 1, 1, 0, 0])

    def test_large_single_char(self):
        d = RepeatDetector(max_ngram_size=5)
        d.add_letters("a" * 10000)
        self.assertEqual(d.ngram_repeats(), [10000, 5000, 3333, 2500, 2000])

    def test_repeating_pattern(self):
        d = RepeatDetector(max_ngram_size=5)
        d.add_letters("abcabcabcabc")
        self.assertEqual(d.ngram_repeats(), [1, 1, 4, 1, 1])

    def test_mixed_characters(self):
        d = RepeatDetector(max_ngram_size=4)
        d.add_letters("abcdabcabcdabc")
        self.assertEqual(d.ngram_repeats(), [1, 1, 1, 1])

    def test_palindrome(self):
        d = RepeatDetector(max_ngram_size=5)
        d.add_letters("racecar")
        self.assertEqual(d.ngram_repeats(), [1, 1, 1, 1, 1])

    def test_repeats_not_at_end(self):
        d = RepeatDetector(max_ngram_size=3)
        d.add_letters("abcabcxyz")
        self.assertEqual(d.ngram_repeats(), [1, 1, 1])

    def test_long_repeat_at_end(self):
        d = RepeatDetector(max_ngram_size=5)
        d.add_letters("abcabcabcabcabcabcabcabcabcabc")
        self.assertEqual(d.ngram_repeats(), [1, 1, 10, 1, 1])

    def test_large_repeating_pattern(self):
        d = RepeatDetector(max_ngram_size=4)
        pattern = "abcd"
        repeat_count = 1000
        d.add_letters(pattern * repeat_count)
        self.assertEqual(d.ngram_repeats(), [1, 1, 1, repeat_count])

    def test_unicode_characters(self):
        d = RepeatDetector(max_ngram_size=3)
        d.add_letters("αβγαβγ")
        self.assertEqual(d.ngram_repeats(), [1, 1, 2])

    def test_random_data(self):
        random.seed(42)
        d = RepeatDetector(max_ngram_size=5)
        data = "".join(random.choices(string.ascii_letters, k=10000))
        d.add_letters(data)
        counts = d.ngram_repeats()
        for count in counts:
            self.assertTrue(0 <= count <= len(data))

    def test_special_characters(self):
        d = RepeatDetector(max_ngram_size=4)
        d.add_letters("@@##@@##")
        self.assertEqual(d.ngram_repeats(), [2, 1, 1, 2])

    def test_incremental_addition(self):
        d = RepeatDetector(max_ngram_size=3)
        d.add_letters("abc")
        self.assertEqual(d.ngram_repeats(), [1, 1, 1])
        d.add_letters("abc")
        self.assertEqual(d.ngram_repeats(), [1, 1, 2])
        d.add_letters("abc")
        self.assertEqual(d.ngram_repeats(), [1, 1, 3])

    def test_long_non_repeating_sequence(self):
        d = RepeatDetector(max_ngram_size=5)
        d.add_letters("abcdefghijklmnopqrstuvwxyz")
        self.assertEqual(d.ngram_repeats(), [1, 1, 1, 1, 1])

    def test_alternating_characters(self):
        d = RepeatDetector(max_ngram_size=4)
        d.add_letters("ababababab")
        self.assertEqual(d.ngram_repeats(), [1, 5, 1, 2])


class BenchmarkRepeatDetect(unittest.TestCase):
    def testLargeRandom(self):
        all_data = []

        for iter in range(1000):
            all_data.append("".join(random.choices("a", k=10000)))

        start = time.perf_counter()

        for data in all_data:
            d = RepeatDetector(max_ngram_size=20)
            d.add_letters(data)
            print(d.ngram_repeats())

        end = time.perf_counter()

        print(f"testLargeRandom took {end-start:0.0001f} seconds")

# From Hynek Kydlicek @ HF
class HynekRepetitionChecker:
      """
      Check for repeated lines and sentences in a text.
      """
      def __init__(self, min_line_repetitions=3, min_sentence_repetitions=5, min_char_repetition=350, check_lines=True, check_sentences=True, check_chars=True, min_sentence_length=50, min_line_length=8):
            """
            Initializes the RepetitionChecker.

            Args:
                min_repetitions (int): The number of times a line or sentence (after normalization)
                                       must be seen to be flagged as a repetition.
                check_lines (bool): Whether to check for repeated lines.
                check_sentences (bool): Whether to check for repeated sentences.
                min_sentence_length (int): Minimum character length for a normalized sentence string
                                           (excluding the terminator) to be considered for repetition checking.
            """

            self.min_line_repetitions = min_line_repetitions
            self.min_sentence_repetitions = min_sentence_repetitions
            self.min_char_repetition = min_char_repetition
            self.min_sentence_length = min_sentence_length
            self.min_line_length = min_line_length

            self.do_check_lines = check_lines
            self.do_check_sentences = check_sentences
            self.do_check_chars = check_chars

            self._current_line_buffer = []
            self._current_sentence_buffer = []
            self._current_char_buffer = []
            self._last_sentence_content = ""
            self._last_line_content = ""
            self._last_char_content = ""

            self._line_reps = 0
            self._sentence_reps = 0
            self._char_reps = 0

            self._sentence_terminators = {'.', '?', '!'}

      def add_char(self, unigram: str) -> Literal["sentence", "line", "char"] | None:
            """
            Processes the next character and checks for repetitions.

            Checks if the characters processed so far contain:
              1. Repeated lines (normalized by stripping whitespace; newline characters themselves are delimiters, not content).
              2. Repeated sentences (normalized by stripping whitespace; identified by '.', '?', or '!').

            Args:
                unigram (str): The next single character to process.

            Returns:
                "sentence": If a sentence repetition meeting the criteria is detected.
                "line": If a line repetition meeting the criteria is detected (and no sentence repetition took precedence).
                None: If no new repetition is detected with this character.
            """
            detected_repetition_type = None

            # Add character to both buffers
            self._current_line_buffer.append(unigram)
            self._current_sentence_buffer.append(unigram)

            # --- Sentence Check ---
            if self.do_check_sentences and unigram in self._sentence_terminators:
                # Sentence content is everything in the buffer *before* the current terminator, then stripped.
                sentence_content = "".join(self._current_sentence_buffer)

                if len(sentence_content) >= self.min_sentence_length and sentence_content == self._last_sentence_content:
                    self._sentence_reps += 1
                    if self._sentence_reps >= self.min_sentence_repetitions:
                        detected_repetition_type = "sentence"
                else:
                    self._sentence_reps = 0
                
                # Reset sentence buffer after processing its end
                self._current_sentence_buffer = []
                self._last_sentence_content = sentence_content

            # --- Unigram Check ---
            if self.do_check_chars:
                # Char content is everything in the buffer *before* the newline char, then stripped.
                if unigram == self._last_char_content:
                    self._char_reps += 1
                    if self._char_reps >= self.min_char_repetition:
                        detected_repetition_type = "char"
                else:
                    self._char_reps = 0
                
                # Reset char buffer after processing its end
                self._last_char_content = unigram
            

            # --- Line Check ---
            if self.do_check_lines and unigram == '\n' and len(self._current_line_buffer) > 2 and self._current_line_buffer[-2] != '\n':
                # Line content is everything in the buffer *before* the newline char, then stripped.
                line_content = "".join(self._current_line_buffer)

                if line_content and line_content == self._last_line_content:
                    self._line_reps += 1
                    if self._line_reps >= self.min_line_repetitions and (not "|" in line_content or self._line_reps >= 3*self.min_line_repetitions) and (len(line_content) >= self.min_line_length or self._line_reps >= 4*self.min_line_repetitions):
                        detected_repetition_type = "line"
                else:
                    self._line_reps = 0
                
                # Reset line buffer after processing its end
                self._current_line_buffer = []
                self._last_line_content = line_content
            
            return detected_repetition_type

if __name__ == "__main__":
    unittest.main()
