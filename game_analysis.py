import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import pandas as pd
from collections import defaultdict
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress
import re
from statistics import mean, median, stdev

class GameAnalytics:
    """Analyzes game logs and provides insights into gameplay patterns and performance."""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.console = Console()
        self.cached_stats = None
        self.cached_logs = None
        
    def read_game_stats(self) -> List[Dict]:
        """Read all game statistics from the JSONL file with caching."""
        if self.cached_stats is not None:
            return self.cached_stats
            
        stats_file = self.log_dir / "game_stats.jsonl"
        stats = []
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                for line in f:
                    try:
                        stats.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        self.console.print("[red]Warning: Corrupted game stat entry found[/red]")
                        continue
        self.cached_stats = stats
        return stats
    
    def read_game_logs(self) -> Dict[str, List[str]]:
        """Read all individual game log files with caching."""
        if self.cached_logs is not None:
            return self.cached_logs
            
        logs = {}
        log_files = self.log_dir.glob("game_*.log")
        
        with Progress() as progress:
            task = progress.add_task("[cyan]Reading log files...", total=len(list(self.log_dir.glob("game_*.log"))))
            for log_file in self.log_dir.glob("game_*.log"):
                try:
                    with open(log_file, 'r') as f:
                        logs[log_file.name] = f.readlines()
                    progress.advance(task)
                except Exception as e:
                    self.console.print(f"[red]Error reading {log_file}: {str(e)}[/red]")
        
        self.cached_logs = logs
        return logs
    
    def analyze_game_patterns(self, stats: List[Dict]) -> Dict:
        """Analyze patterns in gameplay data with enhanced metrics."""
        patterns = {
            "total_games": len(stats),
            "ai_wins": sum(1 for game in stats if game.get("ai_won", False)),
            "avg_questions": sum(game["questions_asked"] for game in stats) / len(stats) if stats else 0,
            "mode_distribution": defaultdict(int),
            "common_subjects": defaultdict(int),
            "avg_time": timedelta(),
            "success_rate_by_mode": defaultdict(lambda: {"wins": 0, "total": 0}),
            "question_efficiency": [],  # Questions needed vs success
            "time_periods": {
                "morning": 0,
                "afternoon": 0,
                "evening": 0,
                "night": 0
            }
        }
        
        total_time = timedelta()
        question_counts = []
        
        for game in stats:
            # Existing analysis
            patterns["mode_distribution"][game["mode"]] += 1
            if game.get("secret_subject"):
                patterns["common_subjects"][game["secret_subject"].lower()] += 1
            
            # Track success rates by mode
            mode = game["mode"]
            patterns["success_rate_by_mode"][mode]["total"] += 1
            if game.get("ai_won", False):
                patterns["success_rate_by_mode"][mode]["wins"] += 1
            
            # Question efficiency
            question_counts.append(game["questions_asked"])
            
            # Time analysis
            if game.get("conversation_history"):
                start_time = datetime.fromisoformat(game["conversation_history"][0]["timestamp"])
                end_time = datetime.fromisoformat(game["conversation_history"][-1]["timestamp"])
                game_duration = end_time - start_time
                total_time += game_duration
                
                # Categorize by time of day
                hour = start_time.hour
                if 5 <= hour < 12:
                    patterns["time_periods"]["morning"] += 1
                elif 12 <= hour < 17:
                    patterns["time_periods"]["afternoon"] += 1
                elif 17 <= hour < 22:
                    patterns["time_periods"]["evening"] += 1
                else:
                    patterns["time_periods"]["night"] += 1
        
        if stats:
            patterns["avg_time"] = total_time / len(stats)
            patterns["question_stats"] = {
                "mean": mean(question_counts),
                "median": median(question_counts),
                "std_dev": stdev(question_counts) if len(question_counts) > 1 else 0
            }
        
        return patterns
    
    def analyze_question_patterns(self, stats: List[Dict]) -> Dict:
        """Analyze patterns in questions asked during games with enhanced metrics."""
        question_patterns = {
            "common_questions": defaultdict(int),
            "yes_no_distribution": {"yes": 0, "no": 0, "unsure": 0},
            "avg_turns_to_win": 0,
            "question_timing": [],
            "question_categories": defaultdict(int),
            "winning_questions": defaultdict(int),
            "question_sequence_patterns": defaultdict(int)
        }

        winning_turns = []
        last_three_questions = []

        # Regex patterns for more accurate matching
        direct_guess_pattern = re.compile(r'\bis\s+it\b', re.IGNORECASE)
        capability_pattern = re.compile(r'\b(can|does|do)\b', re.IGNORECASE)
        function_pattern = re.compile(r'\b(used|purpose|function)\b', re.IGNORECASE)
        
        for game in stats:
            history = game.get("conversation_history", [])
            game_questions = []

            for entry in history:
                question = entry["question"].lower()
                answer = entry["answer"].lower()

                # Existing analysis
                question_patterns["common_questions"][question] += 1

                # Track answer distribution
                if "yes" in answer:
                    question_patterns["yes_no_distribution"]["yes"] += 1
                elif "no" in answer:
                    question_patterns["yes_no_distribution"]["no"] += 1
                else:
                    question_patterns["yes_no_distribution"]["unsure"] += 1

                # Categorize questions using regex
                if direct_guess_pattern.search(question):
                    question_patterns["question_categories"]["direct_guess"] += 1
                elif capability_pattern.search(question):
                    question_patterns["question_categories"]["capability"] += 1
                elif function_pattern.search(question):
                    question_patterns["question_categories"]["function"] += 1

                # Track timing between questions
                if len(game_questions) > 0:
                    current_time = datetime.fromisoformat(entry["timestamp"])
                    prev_time = datetime.fromisoformat(history[len(game_questions) - 1]["timestamp"])
                    question_patterns["question_timing"].append((current_time - prev_time).total_seconds())

                game_questions.append(question)

            # Analyze question sequences
            for i in range(len(game_questions) - 2):
                sequence = tuple(game_questions[i:i+3])
                question_patterns["question_sequence_patterns"][sequence] += 1

            if game.get("ai_won", False):
                winning_turns.append(len(history))
                # Track questions that led to wins
                if game_questions:
                    question_patterns["winning_questions"][game_questions[-1]] += 1

        if winning_turns:
            question_patterns["avg_turns_to_win"] = sum(winning_turns) / len(winning_turns)
            question_patterns["win_turn_distribution"] = {
                "min": min(winning_turns),
                "max": max(winning_turns),
                "median": median(winning_turns)
            }

        return question_patterns
    
    def analyze_performance_trends(self, stats: List[Dict]) -> pd.DataFrame:
        """Analyze performance trends over time with enhanced metrics."""
        if not stats:
            return pd.DataFrame()
            
        data = []
        for game in stats:
            game_data = {
                "timestamp": datetime.fromisoformat(game["timestamp"]),
                "questions_asked": game["questions_asked"],
                "ai_won": game.get("ai_won", False),
                "mode": game["mode"],
                "duration": None,
                "avg_time_per_question": None
            }
            
            # Calculate duration and time per question
            if game.get("conversation_history"):
                start_time = datetime.fromisoformat(game["conversation_history"][0]["timestamp"])
                end_time = datetime.fromisoformat(game["conversation_history"][-1]["timestamp"])
                duration = end_time - start_time
                game_data["duration"] = duration.total_seconds()
                game_data["avg_time_per_question"] = duration.total_seconds() / game["questions_asked"]
            
            data.append(game_data)
        
        df = pd.DataFrame(data)
        
        # Add derived metrics
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.day_name()
        df['success_rate'] = df['ai_won'].rolling(window=10).mean()
        
        return df
    
    def analyze_error_patterns(self, logs: Dict[str, List[str]]) -> Dict:
        """Analyze error patterns and system performance from logs."""
        error_patterns = {
            "error_counts": defaultdict(int),
            "warning_counts": defaultdict(int),
            "token_usage": [],
            "response_times": [],
            "initialization_issues": []
        }
        
        error_regex = r"ERROR.*?](.+)"
        warning_regex = r"WARNING.*?](.+)"
        token_regex = r"Token usage.*?Input: (\d+), Output: (\d+), Total: (\d+)"
        
        for log_file, lines in logs.items():
            for line in lines:
                # Track errors
                if "ERROR" in line:
                    if match := re.search(error_regex, line):
                        error_patterns["error_counts"][match.group(1).strip()] += 1
                
                # Track warnings
                if "WARNING" in line:
                    if match := re.search(warning_regex, line):
                        error_patterns["warning_counts"][match.group(1).strip()] += 1
                
                # Track token usage
                if match := re.search(token_regex, line):
                    error_patterns["token_usage"].append({
                        "input": int(match.group(1)),
                        "output": int(match.group(2)),
                        "total": int(match.group(3))
                    })
                
                # Track initialization issues
                if "initialization" in line.lower() and ("fail" in line.lower() or "error" in line.lower()):
                    error_patterns["initialization_issues"].append(line.strip())
        
        return error_patterns
    
    def display_game_summary(self, patterns: Dict):
        """Display enhanced game summary statistics."""
        # Existing table implementation enhanced with new metrics...
        table = Table(title="Game Summary Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Total Games", str(patterns["total_games"]))
        win_rate = (patterns['ai_wins'] / patterns['total_games'] * 100) if patterns['total_games'] > 0 else 0
        table.add_row("AI Win Rate", f"{win_rate:.1f}%")
        table.add_row("Average Questions", f"{patterns['avg_questions']:.1f}")
        table.add_row("Average Game Time", str(patterns["avg_time"]))
        
        # Add new success rate by mode
        for mode, stats in patterns["success_rate_by_mode"].items():
            rate = (stats["wins"] / stats["total"] * 100) if stats["total"] > 0 else 0
            table.add_row(f"{mode} Success Rate", f"{rate:.1f}%")
        
        self.console.print(table)
        
        # Display time period distribution
        time_table = Table(title="Game Time Distribution")
        time_table.add_column("Time Period", style="cyan")
        time_table.add_column("Games", style="magenta")
        
        for period, count in patterns["time_periods"].items():
            time_table.add_row(period.title(), str(count))
        
        self.console.print(time_table)
    
    def generate_insights_report(self) -> str:
        """Generate a comprehensive insights report from all analysis."""
        stats = self.read_game_stats()
        logs = self.read_game_logs()
        
        if not stats:
            return "No game statistics found."
        
        patterns = self.analyze_game_patterns(stats)
        question_patterns = self.analyze_question_patterns(stats)
        performance_df = self.analyze_performance_trends(stats)
        error_patterns = self.analyze_error_patterns(logs)
        
        # Display enhanced analysis
        self.display_game_summary(patterns)
        self.display_question_analysis(question_patterns)
        self.display_performance_trends(performance_df)
        self.display_error_analysis(error_patterns)
        
        return self.generate_recommendations(patterns, question_patterns, error_patterns)
    
    def display_error_analysis(self, error_patterns: Dict):
        """Display error analysis results."""
        table = Table(title="System Performance Analysis")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        # Token usage statistics
        if error_patterns["token_usage"]:
            avg_total = mean(t["total"] for t in error_patterns["token_usage"])
            table.add_row("Average Token Usage", f"{avg_total:.1f}")
        
        # Error counts
        total_errors = sum(error_patterns["error_counts"].values())
        table.add_row("Total Errors", str(total_errors))
        
        self.console.print(table)
        
        if error_patterns["error_counts"]:
            error_table = Table(title="Common Errors")
            error_table.add_column("Error Type", style="red")
            error_table.add_column("Count", style="magenta")
            
            for error, count in sorted(error_patterns["error_counts"].items(), 
                                    key=lambda x: x[1], reverse=True)[:5]:
                error_table.add_row(error, str(count))
            
            self.console.print(error_table)
    
    def generate_recommendations(self, patterns: Dict, question_patterns: Dict, 
                           error_patterns: Dict) -> str:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        # Performance recommendations
        win_rate = (patterns['ai_wins'] / patterns['total_games'] * 100) if patterns['total_games'] > 0 else 0
        if win_rate < 70:
            recommendations.append("Consider improving question strategy - win rate below 70%")
        
        # Question pattern recommendations
        if question_patterns["avg_turns_to_win"] > 15:
            recommendations.append("Questions could be more efficient - averaging high turn count")
        
        # Mode-specific recommendations
        for mode, stats in patterns["success_rate_by_mode"].items():
            rate = (stats["wins"] / stats["total"] * 100) if stats["total"] > 0 else 0
            if rate < 60:
                recommendations.append(f"Performance in {mode} mode needs improvement - success rate: {rate:.1f}%")
        
        # Time-based recommendations
        time_distribution = patterns["time_periods"]
        total_games = sum(time_distribution.values())
        for period, count in time_distribution.items():
            percentage = (count / total_games * 100) if total_games > 0 else 0
            if percentage > 40:
                recommendations.append(f"Consider diversifying game times - {percentage:.1f}% of games played during {period}")
        
        # Question type recommendations
        yes_no = question_patterns["yes_no_distribution"]
        total_answers = sum(yes_no.values())
        if total_answers > 0:
            yes_ratio = yes_no["yes"] / total_answers * 100
            if yes_ratio > 70:
                recommendations.append("Questions may be too leading - high proportion of 'yes' answers")
        
        # Error-based recommendations
        total_errors = sum(error_patterns["error_counts"].values())
        if total_errors > patterns["total_games"] * 0.1:  # More than 10% of games have errors
            recommendations.append("System stability needs attention - high error rate detected")
        
        # Token usage recommendations
        if error_patterns["token_usage"]:
            avg_total = mean(t["total"] for t in error_patterns["token_usage"])
            if avg_total > 1000:  # Arbitrary threshold, adjust as needed
                recommendations.append("Consider optimizing token usage - high average consumption")
        
        # Format recommendations
        if not recommendations:
            return "No specific recommendations at this time. System is performing well."
        
        return "\n".join([f"- {rec}" for rec in recommendations])

    def display_question_analysis(self, question_patterns: Dict):
        """Display detailed question pattern analysis."""
        # Question Statistics Table
        stats_table = Table(title="Question Pattern Analysis")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="magenta")
        
        # Basic statistics
        stats_table.add_row("Average Turns to Win", f"{question_patterns['avg_turns_to_win']:.1f}")
        if 'win_turn_distribution' in question_patterns:
            stats_table.add_row("Min Turns to Win", str(question_patterns['win_turn_distribution']['min']))
            stats_table.add_row("Max Turns to Win", str(question_patterns['win_turn_distribution']['max']))
            stats_table.add_row("Median Turns to Win", str(question_patterns['win_turn_distribution']['median']))
        
        self.console.print(stats_table)
        
        # Question Categories Table
        categories_table = Table(title="Question Categories")
        categories_table.add_column("Category", style="cyan")
        categories_table.add_column("Count", style="magenta")
        
        for category, count in sorted(question_patterns["question_categories"].items(), 
                                    key=lambda x: x[1], reverse=True):
            categories_table.add_row(category.replace('_', ' ').title(), str(count))
        
        self.console.print(categories_table)
        
        # Answer Distribution Table
        answer_table = Table(title="Answer Distribution")
        answer_table.add_column("Response", style="cyan")
        answer_table.add_column("Count", style="magenta")
        
        for response, count in question_patterns["yes_no_distribution"].items():
            answer_table.add_row(response.title(), str(count))
        
        self.console.print(answer_table)

    def display_performance_trends(self, df: pd.DataFrame):
        """Display performance trends analysis."""
        if df.empty:
            self.console.print("[yellow]No performance data available[/yellow]")
            return
        
        # Performance Trends Table
        trends_table = Table(title="Performance Trends")
        trends_table.add_column("Metric", style="cyan")
        trends_table.add_column("Value", style="magenta")
        
        # Overall statistics
        trends_table.add_row("Average Questions per Game", f"{df['questions_asked'].mean():.1f}")
        if 'duration' in df.columns:
            avg_duration = df['duration'].mean()
            trends_table.add_row("Average Game Duration", f"{avg_duration:.1f} seconds")
        
        # Success rate over time
        recent_success = df['success_rate'].iloc[-1] if not df['success_rate'].empty else 0
        trends_table.add_row("Recent Success Rate", f"{recent_success*100:.1f}%")
        
        self.console.print(trends_table)
        
        # Time of Day Analysis
        hour_success = df.groupby('hour')['ai_won'].mean()
        
        time_table = Table(title="Success Rate by Hour")
        time_table.add_column("Hour", style="cyan")
        time_table.add_column("Success Rate", style="magenta")
        
        for hour, rate in hour_success.items():
            time_table.add_row(f"{hour:02d}:00", f"{rate*100:.1f}%")
        
        self.console.print(time_table)