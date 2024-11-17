import boto3
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime,timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any
import time
from botocore.exceptions import BotoCoreError, ClientError
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
import random
import re
from game_analysis import  GameAnalytics
@dataclass
class GameConfig:
    """Configuration settings for the game."""
    max_questions: int = 20
    max_retries: int = 3
    timeout: int = 30
    model_id: str = "us.meta.llama3-2-90b-instruct-v1:0"
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 200
    max_tokens: int = 256
    log_dir: str = "logs"

class GameLogger:
    """Enhanced logger with rich console output and file logging."""
    
    def __init__(self, config: GameConfig):
        self.logger = logging.getLogger('20Questions')
        self.logger.setLevel(logging.DEBUG)
        
        # Set up file logging
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = Path(config.log_dir) / f"game_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
        )
        file_handler.setFormatter(file_format)
        self.logger.addHandler(file_handler)
        
        # Set up rich console logging
        self.console = Console()
        rich_handler = RichHandler(console=self.console, show_path=False)
        rich_handler.setLevel(logging.INFO)
        self.logger.addHandler(rich_handler)

    def debug(self, msg: str):
        self.logger.debug(msg)
    
    def info(self, msg: str, style: Optional[str] = None):
        if style:
            self.console.print(msg, style=style)
        else:
            self.logger.info(msg)
    
    def error(self, msg: str):
        self.logger.error(msg)
        self.console.print(f"Error: {msg}", style="bold red")
    
    def warning(self, msg: str):
        self.logger.warning(msg)
        self.console.print(f"Warning: {msg}", style="bold yellow")
    
    def display_game_header(self):
        self.console.print(
            Panel.fit(
                "[bold cyan]Welcome to 20 Questions![/bold cyan]\n"
                "[italic]A classic game of deduction and wit[/italic]",
                border_style="cyan"
            )
        )

    def display_game_status(self, questions_asked: int, max_questions: int):
        table = Table(show_header=False, box=None)
        table.add_row(
            f"Questions remaining: [bold green]{max_questions - questions_asked}[/bold green]",
            f"Questions asked: [bold yellow]{questions_asked}[/bold yellow]"
        )
        self.console.print(table)

class GameError(Exception):
    """Base class for game-related exceptions."""
    pass

class InputError(GameError):
    """Raised when there's an issue with user input."""
    pass

class AIError(GameError):
    """Raised when there's an issue with AI interaction."""
    pass

class TwentyQuestions:
    """Main game class implementing both AI guesser and AI thinker modes."""
    
    def __init__(self, config: Optional[GameConfig] = None):
        self.config = config or GameConfig()
        self.logger = GameLogger(self.config)
        self.questions_asked = 0
        self.conversation_history: List[Dict[str, Any]] = []
        self.is_ai_guesser: bool = False
        self.secret_subject: Optional[str] = None
        
        try:
            self.bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')
            self.logger.debug("Successfully initialized Bedrock client")
        except Exception as e:
            self.logger.error(f"Failed to initialize Bedrock client: {e}")
            raise GameError("Failed to initialize AI service") from e

    def get_system_prompt(self) -> List[Dict[str, str]]:
        """Get the appropriate system prompt based on game mode."""
        base_prompt = f"""You are playing a game of 20 Questions. Follow these rules strictly:
1. Keep all responses concise and natural
2. Stay in character as a player
3. Never reveal the secret when you're the thinker
4. Respond directly to questions without unnecessary commentary
Examples 

{self.read_game_flow('ai_guesser_example.txt')}
********************************************************************************
{self.read_game_flow('human_guesser_example.txt')}"""
        
        if not self.is_ai_guesser:
            role_prompt = """
            You are trying to guess what the human is thinking of by asking yes/no questions.
            - Ask clear, specific yes/no questions
            - Keep track of previous answers
            - Make logical deductions,example is it a person yes, is it a historic figure yes etc. trim down the possiblities
            -The answer could be:

                A person: A historical figure (e.g., Albert Einstein), a fictional character (e.g., Sherlock Holmes), or a contemporary celebrity (e.g., Taylor Swift).It will only be well known people.
                A place: A city (e.g., Paris), a natural landmark (e.g., Mount Everest), or a fictional location (e.g., Hogwarts).
                A thing: A tangible object (e.g., a smartphone), an abstract concept (e.g., freedom), or a famous artifact (e.g., the Mona Lisa)
            - Only make a final guess when confident or when reaching max questions"""
        else:
            role_prompt = """
You have thought of a secret subject and are answering the human's yes/no questions.
- Answer honestly and consistently
- Only respond with 'yes', 'no', or 'I'm not sure' plus minimal clarification if needed
- Do not reveal the secret unless the human guesses correctly
- Remember and maintain the same secret throughout the game"""
        
        return [{"text": base_prompt + role_prompt}]

    def format_messages(self, new_prompt: str) -> List[Dict[str, Any]]:
        """Format the conversation history and new prompt for the API."""
        messages = []
        
        # Add conversation history
        for entry in self.conversation_history:
            messages.append({
                "role": "user",
                "content": [{"text": entry["question"]}]
            })
            messages.append({
                "role": "assistant",
                "content": [{"text": entry["answer"]}]
            })
        
        # Add new prompt
        messages.append({
            "role": "user",
            "content": [{"text": new_prompt}]
        })
        
        return messages
    def read_game_flow(self,file_path):
        """
        Reads a 20 Questions game flow from a text file and returns the conversation in a structured format.
        
        Args:
        - file_path (str): Path to the .txt file containing the game flow.
        
        Returns:
        - List of tuples: Each tuple contains (speaker, question/answer).
        """
        game_flow = []

        try:
            with open(file_path, 'r') as file:
                lines = file.readlines()
                game_flow = lines
                
            
        
        except FileNotFoundError:
            self.logger.error(f"Error: The file at {file_path} was not found.")
        except Exception as e:
            self.logger.error(f"An error occurred: {e}")
        
        return game_flow

    def call_llm(self, prompt: str,append:bool = True) -> str:
        """Call the LLM using the Bedrock Conversation API with error handling and retries."""
        for attempt in range(self.config.max_retries):
            try:
                messages = self.format_messages(prompt)
                system_prompt = self.get_system_prompt()

                self.logger.debug(f"Sending request to LLM: {prompt}")
                
                

                response = self.bedrock.converse(
                    modelId=self.config.model_id,
                    messages=messages,
                    inferenceConfig={
                        "maxTokens": self.config.max_tokens,
                        "temperature": self.config.temperature,
                        "topP": self.config.top_p,
                    },
                    system=system_prompt
                )

                # Extract and process response
                response_text = response["output"]["message"]["content"][0]["text"].strip()
                self.logger.debug(f"Received response from LLM: {response_text}")

                # Log token usage
                if "usage" in response:
                    usage = response["usage"]
                    self.logger.debug(
                        f"Token usage - Input: {usage.get('inputTokens', 'N/A')}, "
                        f"Output: {usage.get('outputTokens', 'N/A')}, "
                        f"Total: {usage.get('totalTokens', 'N/A')}"
                    )

                # Update conversation history
                if append:
                    self.conversation_history.append({
                        "question": prompt,
                        "answer": response_text,
                        "turn": self.questions_asked,
                        "timestamp": datetime.now().isoformat()
                    })

                return response_text

            except (ClientError, BotoCoreError) as e:
                self.logger.error(f"AWS error on attempt {attempt + 1}: {e}")
                if attempt == self.config.max_retries - 1:
                    raise AIError(f"Failed to get AI response after {self.config.max_retries} attempts")
                time.sleep(min(2 ** attempt, self.config.timeout))  # Exponential backoff
            except Exception as e:
                self.logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
                if attempt == self.config.max_retries - 1:
                    raise AIError("Unexpected error during AI interaction")
                time.sleep(1)

    def get_valid_input(self, prompt: str, valid_responses: Optional[List[str]] = None) -> str:
        """Get and validate user input."""
        for _ in range(self.config.max_retries):
            try:
                self.logger.info(prompt, style="bold white")
                user_input = input().strip().lower()
                
                if not valid_responses or user_input in valid_responses:
                    return user_input
                
                self.logger.warning(f"Please choose from: {', '.join(valid_responses)}")
            except EOFError:
                raise InputError("Failed to read input")
        raise InputError(f"Failed to get valid input after {self.config.max_retries} attempts")

    def play_as_ai_guesser(self) -> bool:
        """Handle one round where AI is guessing. Returns True to continue game."""
        try:
            # AI asks a question
            prompt = "Based on previous responses, ask your next yes/no question to guess what the human is thinking of.Remember it could be a person,place or thing"
            question = self.call_llm(prompt)
            self.logger.info(f"\nAI Question {self.questions_asked + 1}: {question}", style="bold cyan")
            
            # Get human's answer
            answer = self.get_valid_input(
                "Your answer (yes/no/not sure): ",
                valid_responses=['yes', 'no', 'not sure']
            )
            self.questions_asked += 1
            
            self.logger.display_game_status(self.questions_asked, self.config.max_questions)
            
            # Handle final guess if max questions reached
            if self.questions_asked >= self.config.max_questions:
                return self.handle_ai_final_guess()
            
            # Let AI decide if it wants to make a guess
            prompt = (f"Based on all previous answers including '{answer}' to your last question, "
                     f"do you want to make a guess now or continue asking questions? "
                     f"Remember only do this if you are confident enough of the answer based on the conversation till now, you have 20 questions and you are upto {self.questions_asked} "

                     f"Respond with 'guess' or 'continue'.")
            decision = self.call_llm(prompt)
            
            if "guess" in decision.lower():
                return self.handle_ai_final_guess()
            
            return True
            
        except (InputError, AIError) as e:
            self.logger.error(f"Error during AI guesser turn: {e}")
            return self.handle_error()

    def play_as_ai_thinker(self) -> bool:
        """Handle one round where AI is thinking of something. Returns True to continue game."""
        try:
            # Get human's question
            self.logger.info("\nAsk a yes/no question or make a guess!", style="bold green")
            question = self.get_valid_input("Your question/guess: ")
            self.questions_asked += 1
            
            # Determine if it's a guess or question
            prompt = f"""Based on our conversation history, the human said: "{question}"
Is this a direct guess at what you're thinking of, or a yes/no question?
If it's a guess, evaluate if they're correct.if yes, respond with correct else incorrect
If it's a question, answer with yes/no/not sure."""
            
            response = self.call_llm(prompt)
            self.logger.info(f"\nAI: {response}", style="bold blue")
            
            self.logger.display_game_status(self.questions_asked, self.config.max_questions)
            
            if any(re.search(rf"\b{re.escape(phrase)}\b", response, re.IGNORECASE) for phrase in ["correct", "you won", "you got it"]):
                self.logger.info("\n🎉 Congratulations! You've won! 🎉", style="bold green")
                return False
            elif any(re.search(rf"\b{re.escape(phrase)}\b", response, re.IGNORECASE) for phrase in ["incorrect", "you lost", "sorry"]):
                self.logger.info("\nSorry, wrong guess. You lost.", style="bold red")
        
                return False
            
            # Check if max questions reached
            if self.questions_asked >= self.config.max_questions:
                self.logger.info("\nGame Over! You've run out of questions!", style="bold red")
                final_reveal = self.call_llm("Reveal what you were thinking of.")
                self.logger.info(f"\nI was thinking of: {final_reveal}", style="bold yellow")
                return False
            
            return True
            
        except (InputError, AIError) as e:
            self.logger.error(f"Error during AI thinker turn: {e}")
            return self.handle_error()

    def handle_ai_final_guess(self) -> bool:
        """Handle the AI's final guess. Returns False to end game."""
        try:
            prompt = "Based on all previous answers, make your final guess about what the human is thinking of."
            guess = self.call_llm(prompt)
            self.logger.info(f"\nAI Final Guess: {guess}", style="bold magenta")
            
            correct = self.get_valid_input(
                "Was this correct? (yes/no): ",
                valid_responses=['yes', 'no']
            )
            
            if correct == 'yes':
                self.logger.info("\n🎮 AI wins! Well played! 🎮", style="bold green")
            else:
                self.logger.info(
                    f"\n🏆 You win! The AI couldn't guess '{self.secret_subject}'! 🏆",
                    style="bold green"
                )
            
            # Save game statistics
            self.save_game_stats(ai_won=(correct == 'yes'))
            
            return False
            
        except (InputError, AIError) as e:
            self.logger.error(f"Error during AI final guess: {e}")
            return False

    def handle_error(self) -> bool:
        """Handle errors during gameplay. Returns True to continue game."""
        try:
            continue_game = self.get_valid_input(
                "\nWould you like to continue? (yes/no): ",
                valid_responses=['yes', 'no']
            )
            return continue_game == 'yes'
        except InputError:
            return False

    def save_game_stats(self, ai_won: bool):
        """Save game statistics to a log file."""
        stats = {
            "timestamp": datetime.now().isoformat(),
            "mode": "AI Guesser" if self.is_ai_guesser else "AI Thinker",
            "questions_asked": self.questions_asked,
            "ai_won": ai_won,
            "secret_subject": self.secret_subject if self.is_ai_guesser else None,
            "conversation_history": self.conversation_history
        }
        
        try:
            stats_file = Path(self.config.log_dir) / "game_stats.jsonl"
            with open(stats_file, 'a') as f:
                json.dump(stats, f)
                f.write('\n')
        except Exception as e:
            self.logger.error(f"Failed to save game statistics: {e}")
    def pick_type(self) -> str:
        """
        Randomly selects a detailed category for the game.
        Returns:
            str: The selected category type
        """
        
        categories = {
            "person": [
                "historical figure",
                "celebrity",
                "fictional character",
                "athlete",
                "musician"
            ],
            "place": [
                "country",
                "city",
                "landmark",
                "natural wonder",
                "building"
            ],
            "thing": [
                "household item",
                "food",
                "technology",
                "animal",
                "vehicle"
            ]
        }
        
        # First pick main category
        main_category = random.choice(list(categories.keys()))
        # Then pick specific subcategory
        specific_type = random.choice(categories[main_category])
        
        # Inform the player
        
        return specific_type
    

    def initialize_game(self):
        """Set up the game based on player's role choice."""
        self.logger.display_game_header()
        
        # Let player choose role
        self.logger.info("\nChoose your role:", style="bold white")
        self.logger.info("1. Thinker (you think of something, AI guesses)", style="blue")
        self.logger.info("2. Guesser (AI thinks of something, you guess)", style="green")
        
        choice = self.get_valid_input(
            "\nEnter your choice (1 or 2): ",
            valid_responses=['1', '2']
        )
        
        self.is_ai_guesser = choice == '1'
        
        if self.is_ai_guesser:
            self.secret_subject = self.get_valid_input(
                "\nThink of something (any object, person, or concept) and enter it here: "
            )
            self.logger.debug(f"Secret subject set: {self.secret_subject}")
            
            # Initialize game context for AI
            init_prompt = (
                "You are now the guesser in a game of 20 questions. "
                "The human has thought of something, and you need to guess it "
                "Remember it could be a person,place or thing"

                "by asking yes/no questions. Are you ready to begin?"
                "Dont ask the question now, just  sintroduce yourself,wish them luck"

            )
            response = self.call_llm(init_prompt)
            self.logger.info(f"\nAI: {response}", style="bold blue")
            
        else:
            # AI thinks of something
            category = self.pick_type()
            self.secret_subject = self.call_llm(f"Think of something and only respond with the item, person, or concept.This time pick on in category : {category}",append=False)
            print(self.secret_subject)
            init_prompt = (
                "You are now the thinker in a game of 20 questions. "
                f"The thing you are thinking is {self.secret_subject} "
                "Remember it and don't reveal it unless the player guesses correctly. "
                "Let the player know you're ready to begin."
            )
            response = self.call_llm(init_prompt)
            self.logger.info(f"\nAI: {response}", style="bold blue")

    def run(self):
        """Main game loop."""
        try:
            self.initialize_game()
            
            while self.questions_asked < self.config.max_questions:
                if self.is_ai_guesser:
                    if not self.play_as_ai_guesser():
                        break
                else:
                    if not self.play_as_ai_thinker():
                        break
            
            # Final game summary
            self.display_game_summary()
            
        except GameError as e:
            self.logger.error(f"Game ended with error: {e}")
            sys.exit(1)
        except KeyboardInterrupt:
            self.logger.info("\nGame terminated by user")
            self.save_game_stats(ai_won=False)  # Save stats on interruption
            sys.exit(0)

    
    def display_game_summary(self):
        """Display a summary of the game statistics."""
        self.logger.info("\n🎮 Game Summary 🎮", style="bold cyan")
        self.logger.info(f"Questions asked: {self.questions_asked}/{self.config.max_questions}")
        self.logger.info(f"Game mode: {'AI Guesser' if self.is_ai_guesser else 'AI Thinker'}")

        if len(self.conversation_history) > 0:
            # Calculate the total time taken for the game
            start_time = datetime.fromisoformat(self.conversation_history[0]['timestamp'])
            end_time = datetime.fromisoformat(self.conversation_history[-1]['timestamp'])
            total_time = end_time - start_time
            
      

            # Calculate the average time per question (assuming timestamps are present for each question)
            question_times = []
            for i in range(1, len(self.conversation_history)):
                prev_time = datetime.fromisoformat(self.conversation_history[i-1]['timestamp'])
                curr_time = datetime.fromisoformat(self.conversation_history[i]['timestamp'])
                question_times.append(curr_time - prev_time)

            if question_times:
                avg_time_per_question = sum(question_times, timedelta()) / len(question_times)
                self.logger.info(f"\nAverage time per question: {avg_time_per_question}")

            # Log total time taken
            self.logger.info(f"\nTotal time taken: {total_time}")

        self.logger.info("\nThanks for playing 20 Questions!", style="bold cyan")
    def analyze_game_history(self):
        """Add this method to your TwentyQuestions class"""
        analyzer = GameAnalytics(self.config.log_dir)
        self.logger.info("\n📊 Game History Analysis 📊")
        analysis = analyzer.generate_insights_report()
        self.logger.info(analysis)

def main():
    """Entry point for the game."""
    try:
        # Load configuration from environment variables if present
        config = GameConfig(
            max_questions=int(os.getenv('GAME_MAX_QUESTIONS', 20)),
            max_retries=int(os.getenv('GAME_MAX_RETRIES', 3)),
            timeout=int(os.getenv('GAME_TIMEOUT', 30)),
            model_id=os.getenv('GAME_MODEL_ID', 'us.meta.llama3-2-90b-instruct-v1:0'),
            temperature=float(os.getenv('GAME_TEMPERATURE', 0.7)),
            top_p=float(os.getenv('GAME_TOP_P', 0.9)),
            top_k=int(os.getenv('GAME_TOP_K', 200)),
            max_tokens=int(os.getenv('GAME_MAX_TOKENS', 256)),
            log_dir=os.getenv('GAME_LOG_DIR', 'logs')
        )
        
        game = TwentyQuestions(config)
        game.run()
        
        # Create console for styled output
        console = Console()
        
        # Ask if user wants to see analysis
        console.print("\nWould you like to see a detailed analysis of your game history?", style="bold cyan")
        console.print("1. Yes - Show full analysis", style="green")
        console.print("2. No - Exit game", style="yellow")
        
        try:
            choice = input("\nEnter your choice (1 or 2): ").strip()
            if choice == "1":
                console.print("\n📊 Generating Game Analysis Report...", style="bold blue")
                game.analyze_game_history()
            else:
                console.print("\nThanks for playing! Come back soon! 👋", style="bold green")
        except EOFError:
            console.print("\nInput error occurred. Exiting game.", style="bold red")
            
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()