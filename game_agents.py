from typing import List, Dict, Optional
import json
from botocore.exceptions import BotoCoreError, ClientError
from five_questions_game import GameConfig, CustomLogger, AIError

class BaseGameAgent:
    """Base class for game agents."""
    def __init__(self, config: GameConfig, logger: CustomLogger, bedrock_client):
        self.config = config
        self.logger = logger
        self.bedrock = bedrock_client
        self.game_history: List[Dict[str, str]] = []

    def call_llm(self, prompt: str) -> str:
        """Call the LLM with error handling and retries."""
        for attempt in range(self.config.max_retries):
            try:
                history_context = "\n".join(
                    f"Q: {h['question']}\nA: {h['answer']}"
                    for h in self.game_history
                )
                
                full_prompt = f"""You are playing a game of 20 questions.
Current game state:
Game history:
{history_context}

Current interaction:
{prompt}

Respond naturally as if you're playing the game. Keep responses concise and clear."""

                request_body = {
                    "prompt": full_prompt,
                    "temperature": self.config.temperature,
                    "top_p": self.config.top_p,
                    "max_gen_len": self.config.max_tokens
                }
                
                self.logger.debug(f"Sending request to LLM: {prompt}")
                
                response = self.bedrock.invoke_model(
                    modelId=self.config.model_id,
                    body=json.dumps(request_body)
                )
                
                response_body = json.loads(response['body'].read())
                generated_text = response_body['generation']
                
                self.logger.debug(f"Received response from LLM: {generated_text}")
                
                self.game_history.append({
                    "question": prompt,
                    "answer": generated_text
                })
                
                return generated_text

            except (BotoCoreError, ClientError) as e:
                self.logger.error(f"AWS error on attempt {attempt + 1}: {e}")
                if attempt == self.config.max_retries - 1:
                    raise AIError(f"Failed to get AI response after {self.config.max_retries} attempts") from e
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Unexpected error: {e}")
                raise AIError("Unexpected error in AI communication") from e

class ThinkerAgent(BaseGameAgent):
    """Agent for when player is thinking of something and AI is guessing."""
    def __init__(self, config: GameConfig, logger: CustomLogger, bedrock_client):
        super().__init__(config, logger, bedrock_client)
        self.secret_subject: Optional[str] = None

    def initialize_game(self, secret_subject: str) -> str:
        """Initialize the game with the player's secret subject."""
        self.secret_subject = secret_subject
        self.logger.debug(f"Secret subject set: {self.secret_subject}")
        
        response = self.call_llm(
            "I'm ready to start guessing. I'll ask yes/no questions to figure out what you're thinking of."
        )
        return response

    def get_next_question(self) -> str:
        """Get the AI's next question."""
        response = self.call_llm("Ask the next yes/no question to guess the secret subject.")
        return response

    def handle_answer(self, answer: str) -> Optional[str]:
        """Handle the player's answer and potentially make a final guess."""
        # The AI can use the answer in the game history for the next question
        return None

class GuesserAgent(BaseGameAgent):
    """Agent for when AI is thinking of something and player is guessing."""
    def __init__(self, config: GameConfig, logger: CustomLogger, bedrock_client):
        super().__init__(config, logger, bedrock_client)

    def initialize_game(self) -> str:
        """Initialize the game by having the AI think of something."""
        response = self.call_llm(
            "Think of something for the player to guess. Respond with: 'I've thought of something. Ready for your yes/no questions!'"
        )
        return response

    def answer_question(self, question: str) -> str:
        """Answer the player's question."""
        return self.call_llm(question)