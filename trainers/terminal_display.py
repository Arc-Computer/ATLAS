#!/usr/bin/env python
"""
Terminal Display Manager for ATLAS Teaching Optimization
Inspired by CrewAI's clean terminal output formatting
"""

import sys
import time
import json
import re
from typing import Dict, List, Optional, Any
from datetime import datetime
from colorama import init, Fore, Back, Style
import threading
from queue import Queue

init(autoreset=True)


class TerminalDisplay:
    """Clean terminal display manager with CrewAI-style formatting"""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.current_eval = 0
        self.total_evaluations = 0
        self.scores = {'baseline': [], 'teaching': []}
        self.start_time = datetime.now()
        self.update_queue = Queue()
        self.running = False

    def _print_header(self):
        """Print the application header"""
        print("\n" + "="*80)
        print(f"ATLAS Teaching Optimization System")
        print(f"Real-time Training Monitor")
        print("="*80 + "\n")

    def _convert_markdown_to_plain(self, text: str) -> str:
        """Convert markdown-formatted text to plain readable format"""
        # Remove bold markdown (** or __)
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        text = re.sub(r'__(.*?)__', r'\1', text)

        # Remove italic markdown (* or _)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        text = re.sub(r'_([^_]+)_', r'\1', text)

        # Remove code blocks
        text = re.sub(r'```[a-z]*\n(.*?)```', r'\1', text, flags=re.DOTALL)
        text = re.sub(r'`([^`]+)`', r'\1', text)

        # Convert headers to uppercase
        text = re.sub(r'^#{1,6}\s+(.*)$', lambda m: m.group(1).upper(), text, flags=re.MULTILINE)

        # Convert bullet points
        text = re.sub(r'^\s*[-*+]\s+', '• ', text, flags=re.MULTILINE)

        # Convert numbered lists
        text = re.sub(r'^\s*(\d+)\.\s+', r'\1. ', text, flags=re.MULTILINE)

        # Remove horizontal rules
        text = re.sub(r'^-{3,}$', '', text, flags=re.MULTILINE)

        # Clean up excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text.strip()

    def _print_box(self, title: str, content: str, width: int = 120):
        """Print content in a formatted box with white text only"""
        border = "─" * (width - 2)

        # Convert markdown to plain text
        content = self._convert_markdown_to_plain(content)

        print(f"\n┌─{border}┐")

        # Center the title
        title_padding = (width - len(title) - 4) // 2
        print(f"│ {' ' * title_padding}{title}{' ' * (width - len(title) - title_padding - 3)}│")
        print(f"├─{border}┤")

        # Split content into lines and wrap if needed
        lines = content.split('\n')
        for line in lines:
            if len(line) > width - 4:
                # Wrap long lines
                wrapped_lines = [line[i:i+width-4] for i in range(0, len(line), width-4)]
                for wrapped_line in wrapped_lines:
                    print(f"│ {wrapped_line:<{width-3}}│")
            else:
                print(f"│ {line:<{width-3}}│")

        print(f"└─{border}┘")

    def _print_progress_bar(self, current: int, total: int, label: str = "Progress"):
        """Print a progress bar"""
        if total == 0:
            percentage = 0
        else:
            percentage = (current / total) * 100

        bar_length = 40
        filled = int(bar_length * current // total) if total > 0 else 0
        bar = "█" * filled + "░" * (bar_length - filled)

        print(f"\n{label}: [{bar}] {percentage:.1f}% ({current}/{total})")

    def _print_section_header(self, title: str, icon: str = "▶"):
        """Print a section header"""
        print(f"\n{icon} {title}")
        print("─" * 60)

    def _format_json_output(self, data: Any, indent: int = 2) -> str:
        """Format JSON data for display - handle various input types"""
        if isinstance(data, str):
            # Try to extract JSON from the string
            json_match = re.search(r'\{.*\}', data, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                except:
                    # If JSON extraction fails, try to parse the whole string
                    try:
                        data = json.loads(data)
                    except:
                        # If all parsing fails, return cleaned string
                        return self._convert_markdown_to_plain(data)
            else:
                # No JSON found, return as plain text
                return self._convert_markdown_to_plain(data)

        # Format as JSON with proper indentation
        try:
            return json.dumps(data, indent=indent, ensure_ascii=False)
        except:
            return str(data)

    def _print_comparison_table(self, eval_num: int, baseline_score: float, teaching_score: float):
        """Print a comparison table row"""
        delta = teaching_score - baseline_score
        delta_str = f"+{delta:.3f}" if delta >= 0 else f"{delta:.3f}"

        if delta > 0:
            status = "Improved"
        elif delta < 0:
            status = "Declined"
        else:
            status = "Same"

        print(f"│ {eval_num:^8} │ {baseline_score:^12.3f} │ {teaching_score:^12.3f} │ "
              f"{delta_str:^10} │ {status:^15} │")

    def _print_metrics(self, metrics: Dict[str, Any]):
        """Print metrics panel with 4 separate RIM rewards"""
        print(f"\nPerformance Metrics")
        print("─" * 40)

        if metrics.get('accuracy') is not None:
            print(f"Accuracy:     {metrics['accuracy']:.3f}")

        if metrics.get('helpfulness') is not None:
            print(f"Helpfulness:  {metrics['helpfulness']:.3f}")

        if metrics.get('process') is not None:
            print(f"Process:      {metrics['process']:.3f}")

        if metrics.get('diagnostic') is not None:
            print(f"Diagnostic:   {metrics['diagnostic']:.3f}")

        if metrics.get('avg_reward') is not None:
            print(f"Combined:     {metrics['avg_reward']:.3f}")

        if metrics.get('token_savings') is not None:
            print(f"Token Savings: {metrics['token_savings']:.1f}%")

        print("─" * 40)

    def update_student_diagnostic(self, text: str, sample_num: int = 1, total: int = 1):
        """Display student diagnostic output"""
        if not self.verbose:
            return

        if text:
            self._print_box(
                "STUDENT APPROACH",
                text
            )
        else:
            print(f"\nStudent generating diagnostic approach...")

    def update_teacher_guidance(self, text: str, sample_num: int = 1, total: int = 1):
        """Display teacher guidance"""
        if not self.verbose:
            return

        if text:
            self._print_box(
                "TEACHING STRATEGY",
                text
            )
        else:
            print(f"\nTeacher reviewing student's approach...")

    def update_student_with_teaching(self, text: str, sample_num: int = 1, total: int = 1):
        """Display student with teaching output"""
        if not self.verbose:
            return

        if text:
            self._print_box(
                "ENHANCED APPROACH",
                self._format_json_output(text)
            )
        else:
            print(f"\nStudent executing task with teacher's guidance...")

    def update_baseline(self, text: str):
        """Display baseline output"""
        if not self.verbose:
            return

        if text:
            self._print_box(
                "BASELINE OUTPUT",
                self._format_json_output(text)
            )
        else:
            print(f"\nStudent executing task without guidance (baseline)...")

    def update_reflection_prompts(self, prompts: Dict[str, str]):
        """Display reflection model prompts in boxes"""
        if not self.verbose:
            return

        self._print_section_header("Reflection Model Prompts", ">")

        # Display each prompt in its own box
        for key, value in prompts.items():
            prompt_title = key.upper().replace('_', ' ')
            self._print_box(
                prompt_title,
                value
            )

    def show_comparison_results(self, baseline_score: float, teaching_score: float):
        """Show comparison between baseline and teaching results"""
        if not self.verbose:
            return

        self.scores['baseline'].append(baseline_score)
        self.scores['teaching'].append(teaching_score)

    def update_progress(self, current: int, total: int = None):
        """Update progress indicator"""
        if total:
            self.total_evaluations = total
        self.current_eval = current

        if self.verbose:
            self._print_progress_bar(current, self.total_evaluations, "Overall Progress")

    def show_iteration_complete(self, iteration: int, score: float, metrics: Optional[Dict] = None):
        """Show completion of an iteration"""
        if not self.verbose:
            return

        print(f"\nIteration {iteration} Complete")

        if metrics:
            self._print_metrics(metrics)

        elapsed = (datetime.now() - self.start_time).total_seconds()
        print(f"Time Elapsed: {elapsed:.1f}s")
        print("="*80)

    def show_final_summary(self):
        """Show final summary statistics"""
        if not self.verbose or not self.scores['baseline'] or not self.scores['teaching']:
            return

        avg_baseline = sum(self.scores['baseline']) / len(self.scores['baseline'])
        avg_teaching = sum(self.scores['teaching']) / len(self.scores['teaching'])
        avg_improvement = avg_teaching - avg_baseline

        print(f"\nFINAL SUMMARY")
        print("="*80)
        print(f"Total Evaluations: {len(self.scores['baseline'])}")
        print(f"Average Baseline Score: {avg_baseline:.3f}")
        print(f"Average Teaching Score: {avg_teaching:.3f}")

        if avg_improvement > 0:
            print(f"Average Improvement: +{avg_improvement:.3f}")
        else:
            print(f"Average Improvement: {avg_improvement:.3f}")

        print(f"\nTotal Runtime: {(datetime.now() - self.start_time).total_seconds():.1f}s")
        print("="*80)

    def start(self, total_evaluations: int = 50):
        """Initialize the display"""
        self.total_evaluations = total_evaluations
        self.running = True
        self._print_header()
        return self

    def stop(self):
        """Stop the display and show summary"""
        self.running = False
        self.show_final_summary()


class DisplayManager:
    """Manager for terminal display operations"""

    def __init__(self, verbose: bool = True):
        self.display = TerminalDisplay(verbose=verbose)

    def start(self, total_evaluations: int = 50):
        """Start the display manager"""
        return self.display.start(total_evaluations)

    def update(self, update_type: str, **kwargs):
        """Update display with new information"""
        if not self.display:
            return

        
        if update_type == "student":
            self.display.update_student_diagnostic(
                kwargs.get('text', ''),
                kwargs.get('sample', 1),
                kwargs.get('total', 1)
            )
        elif update_type == "teacher":
            self.display.update_teacher_guidance(
                kwargs.get('text', ''),
                kwargs.get('sample', 1),
                kwargs.get('total', 1)
            )
        elif update_type == "student_with_teaching":
            
            self.display.update_student_with_teaching(
                kwargs.get('text', ''),
                kwargs.get('sample', 1),
                kwargs.get('total', 1)
            )
        elif update_type == "baseline":
            self.display.update_baseline(kwargs.get('text', ''))
        elif update_type == "reflection_prompts":
            self.display.update_reflection_prompts(kwargs.get('prompts', {}))
        elif update_type == "comparison":
            self.display.show_comparison_results(
                kwargs.get('baseline_score', 0.0),
                kwargs.get('teaching_score', 0.0)
            )
        elif update_type == "progress":
            self.display.update_progress(
                kwargs.get('current', 0),
                kwargs.get('total')
            )
        elif update_type == "iteration_complete":
            self.display.show_iteration_complete(
                kwargs.get('iteration', 0),
                kwargs.get('score', 0.0),
                kwargs.get('metrics')
            )

    def stop(self):
        """Stop the display manager"""
        if self.display:
            self.display.stop()