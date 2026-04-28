#!/usr/bin/env python3
"""
Paper Quality Checker

A helper script to run basic quality checks on academic papers.
Checks for common issues like passive voice, hedging, construct consistency, etc.

Usage:
    python check_paper.py your_paper.md
"""

import re
import sys
from collections import Counter
from pathlib import Path


class PaperChecker:
    """Checks academic papers for common quality issues."""

    # Hedging words to watch for
    HEDGES = [
        'might', 'may', 'could', 'possibly', 'perhaps', 'maybe',
        'somewhat', 'relatively', 'fairly', 'rather', 'quite',
        'appears', 'seems', 'suggests', 'indicates', 'tends'
    ]

    # Passive voice indicators
    PASSIVE_INDICATORS = [
        r'\bis\s+\w+ed\b',
        r'\bare\s+\w+ed\b',
        r'\bwas\s+\w+ed\b',
        r'\bwere\s+\w+ed\b',
        r'\bbeen\s+\w+ed\b',
        r'\bcan be found\b'
    ]

    # Nominalizations to avoid
    NOMINALIZATIONS = {
        'utilization': 'use',
        'implementation': 'implement',
        'analysis': 'analyze',
        'investigation': 'investigate',
        'examination': 'examine',
        'consideration': 'consider',
        'determination': 'determine'
    }

    # Wordy phrases
    WORDY_PHRASES = {
        'due to the fact that': 'because',
        'in order to': 'to',
        'at this point in time': 'now',
        'with regard to': 'about',
        'for the purpose of': 'for',
        'make a decision': 'decide',
        'conduct an investigation': 'investigate'
    }

    def __init__(self, filepath):
        self.filepath = Path(filepath)
        self.content = self.filepath.read_text(encoding='utf-8')
        self.sentences = self._split_sentences()
        self.issues = []

    def _split_sentences(self):
        """Split content into sentences."""
        # Simple sentence splitting
        return [s.strip() for s in re.split(r'[.!?]+', self.content) if s.strip()]

    def check_hedging(self):
        """Check for excessive hedging."""
        hedge_count = 0
        hedge_sentences = []

        for sentence in self.sentences:
            words = sentence.lower().split()
            hedges_in_sentence = [word for word in words if word in self.HEDGES]
            if hedges_in_sentence:
                hedge_count += len(hedges_in_sentence)
                hedge_sentences.append((sentence[:100], hedges_in_sentence))

        total_words = len(self.content.split())
        hedge_ratio = (hedge_count / total_words * 100) if total_words > 0 else 0

        if hedge_ratio > 2.0:  # More than 2% hedging words
            self.issues.append({
                'type': 'Excessive Hedging',
                'severity': 'warning',
                'message': f'Found {hedge_count} hedge words ({hedge_ratio:.1f}% of total). Consider reducing hedging.',
                'examples': hedge_sentences[:3]
            })

        return hedge_ratio

    def check_passive_voice(self):
        """Check for passive voice usage."""
        passive_count = 0
        passive_sentences = []

        for sentence in self.sentences:
            for pattern in self.PASSIVE_INDICATORS:
                if re.search(pattern, sentence, re.IGNORECASE):
                    passive_count += 1
                    passive_sentences.append(sentence[:100])
                    break

        total_sentences = len(self.sentences)
        passive_ratio = (passive_count / total_sentences * 100) if total_sentences > 0 else 0

        if passive_ratio > 15.0:  # More than 15% passive
            self.issues.append({
                'type': 'Excessive Passive Voice',
                'severity': 'warning',
                'message': f'{passive_count} sentences use passive voice ({passive_ratio:.1f}%). Target: <10%',
                'examples': passive_sentences[:3]
            })

        return passive_ratio

    def check_sentence_length(self):
        """Check average sentence length."""
        lengths = [len(s.split()) for s in self.sentences]
        avg_length = sum(lengths) / len(lengths) if lengths else 0
        long_sentences = [s[:100] for s in self.sentences if len(s.split()) > 30]

        if avg_length > 25:
            self.issues.append({
                'type': 'Long Sentences',
                'severity': 'info',
                'message': f'Average sentence length: {avg_length:.1f} words. Target: 14-20 words.',
                'examples': long_sentences[:3]
            })

        return avg_length

    def check_nominalizations(self):
        """Check for nominalizations."""
        found_nominalizations = []

        for nominalization, verb in self.NOMINALIZATIONS.items():
            pattern = r'\b' + nominalization + r'\b'
            matches = re.finditer(pattern, self.content, re.IGNORECASE)
            for match in matches:
                context_start = max(0, match.start() - 40)
                context_end = min(len(self.content), match.end() + 40)
                context = self.content[context_start:context_end]
                found_nominalizations.append((nominalization, verb, context))

        if found_nominalizations:
            self.issues.append({
                'type': 'Nominalizations Found',
                'severity': 'info',
                'message': f'Found {len(found_nominalizations)} nominalizations. Consider using verb forms.',
                'examples': [(f'{nom} → {verb}: {ctx}') for nom, verb, ctx in found_nominalizations[:3]]
            })

        return len(found_nominalizations)

    def check_wordy_phrases(self):
        """Check for wordy phrases."""
        found_phrases = []

        for wordy, concise in self.WORDY_PHRASES.items():
            pattern = r'\b' + wordy + r'\b'
            matches = re.finditer(pattern, self.content, re.IGNORECASE)
            for match in matches:
                context_start = max(0, match.start() - 30)
                context_end = min(len(self.content), match.end() + 30)
                context = self.content[context_start:context_end]
                found_phrases.append((wordy, concise, context))

        if found_phrases:
            self.issues.append({
                'type': 'Wordy Phrases',
                'severity': 'info',
                'message': f'Found {len(found_phrases)} wordy phrases. Consider more concise alternatives.',
                'examples': [(f'"{wordy}" → "{concise}": {ctx}') for wordy, concise, ctx in found_phrases[:3]]
            })

        return len(found_phrases)

    def generate_report(self):
        """Generate a comprehensive report."""
        print("=" * 70)
        print(f"Paper Quality Check: {self.filepath.name}")
        print("=" * 70)
        print()

        # Run all checks
        hedge_ratio = self.check_hedging()
        passive_ratio = self.check_passive_voice()
        avg_sentence_length = self.check_sentence_length()
        nominalization_count = self.check_nominalizations()
        wordy_phrase_count = self.check_wordy_phrases()

        # Summary stats
        print("SUMMARY STATISTICS")
        print("-" * 70)
        print(f"Total words: {len(self.content.split())}")
        print(f"Total sentences: {len(self.sentences)}")
        print(f"Average sentence length: {avg_sentence_length:.1f} words (target: 14-20)")
        print(f"Hedging ratio: {hedge_ratio:.1f}% (target: <2%)")
        print(f"Passive voice: {passive_ratio:.1f}% (target: <10%)")
        print(f"Nominalizations found: {nominalization_count}")
        print(f"Wordy phrases found: {wordy_phrase_count}")
        print()

        # Issues
        if self.issues:
            print("ISSUES FOUND")
            print("-" * 70)
            for issue in self.issues:
                print(f"\n[{issue['severity'].upper()}] {issue['type']}")
                print(f"  {issue['message']}")
                if issue.get('examples'):
                    print("  Examples:")
                    for i, example in enumerate(issue['examples'], 1):
                        if isinstance(example, tuple):
                            print(f"    {i}. {example[0][:80]}...")
                        else:
                            print(f"    {i}. {example[:80]}...")
            print()
        else:
            print("No major issues found! Good job!")
            print()

        # Scoring
        score = 100
        if hedge_ratio > 2.0:
            score -= min(20, (hedge_ratio - 2.0) * 5)
        if passive_ratio > 10.0:
            score -= min(15, (passive_ratio - 10.0) * 2)
        if avg_sentence_length > 20:
            score -= min(10, (avg_sentence_length - 20) * 2)
        score -= min(10, nominalization_count)
        score -= min(5, wordy_phrase_count)

        print(f"OVERALL SCORE: {max(0, score):.0f}/100")
        print()

        if score >= 90:
            print("Excellent! Your paper is well-written and clear.")
        elif score >= 75:
            print("Good work! Consider addressing the issues above for improvement.")
        elif score >= 60:
            print("Fair. Review the issues above and revise accordingly.")
        else:
            print("Needs work. Please review and revise based on the issues identified.")

        print("=" * 70)


def main():
    if len(sys.argv) < 2:
        print("Usage: python check_paper.py <paper_file>")
        sys.exit(1)

    filepath = sys.argv[1]

    if not Path(filepath).exists():
        print(f"Error: File '{filepath}' not found.")
        sys.exit(1)

    checker = PaperChecker(filepath)
    checker.generate_report()


if __name__ == '__main__':
    main()
