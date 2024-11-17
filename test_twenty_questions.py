import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import boto3
import json
from pathlib import Path
from rich.console import Console

from twenty_questions import TwentyQuestions, GameConfig, GameError, InputError, AIError, GameLogger

# Fixtures
@pytest.fixture
def mock_bedrock():
    """Mock the AWS Bedrock client"""
    with patch('boto3.client') as mock_client:
        mock_bedrock = Mock()
        mock_client.return_value = mock_bedrock
        yield mock_bedrock

@pytest.fixture
def game_config():
    """Create a test game configuration"""
    return GameConfig(
        max_questions=5,
        max_retries=2,
        timeout=1,
        model_id="us.meta.llama3-2-90b-instruct-v1:0",
        temperature=0.5,
        top_p=0.9,
        top_k=100,
        max_tokens=128,
        log_dir="test_logs"
    )

@pytest.fixture
def game(mock_bedrock, game_config):
    """Create a test game instance"""
    return TwentyQuestions(game_config)

@pytest.fixture
def mock_console():
    """Mock the Rich console"""
    with patch('rich.console.Console') as mock_console:
        yield mock_console

class TestGameConfig:
    """Test the GameConfig class"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = GameConfig()
        assert config.max_questions == 20
        assert config.max_retries == 3
        assert config.timeout == 30
        assert config.model_id == "us.meta.llama3-2-90b-instruct-v1:0"
        assert config.temperature == 0.7
        assert config.top_p == 0.9
        assert config.max_tokens == 256
        assert config.log_dir == "logs"

    def test_custom_config(self, game_config):
        """Test custom configuration values"""
        assert game_config.max_questions == 5
        assert game_config.max_retries == 2
        assert game_config.timeout == 1
        assert game_config.temperature == 0.5

class TestGameLogger:
    """Test the GameLogger class"""

    @pytest.fixture
    def logger(self, game_config):
        return GameLogger(game_config)

    def test_logger_initialization(self, logger):
        """Test logger initialization"""
        assert logger.logger.name == '20Questions'
        assert logger.logger.level == 10  # DEBUG level

    def test_debug_logging(self, logger, caplog):
        """Test debug level logging"""
        test_message = "Debug test message"
        logger.debug(test_message)
        assert test_message in caplog.text

    @patch('rich.console.Console.print')
    def test_info_logging_with_style(self, mock_print, logger):
        """Test styled info logging"""
        test_message = "Info test message"
        logger.info(test_message, style="bold")
        mock_print.assert_called_once_with(test_message, style="bold")

    def test_error_logging(self, logger, caplog):
        """Test error logging"""
        test_message = "Error test message"
        logger.error(test_message)
        assert "ERROR    20Questions:" in caplog.text
        assert "Error test message" in caplog.text

    def test_warning_logging(self, logger, caplog):
        """Test warning logging"""
        test_message = "Warning test message"
        logger.warning(test_message)
        assert "WARNING  20Questions:" in caplog.text
        assert "Warning test message" in caplog.text


class TestTwentyQuestions:
    """Test the main TwentyQuestions game class"""

    def test_initialization(self, game):
        """Test game initialization"""
        assert game.questions_asked == 0
        assert game.conversation_history == []
        assert game.is_ai_guesser is False
        assert game.secret_subject is None

    def test_system_prompts(self, game):
        """Test system prompts for both game modes"""
        # Test AI guesser mode
        game.is_ai_guesser = True
        ai_prompt = game.get_system_prompt()
        assert isinstance(ai_prompt, list)
        assert "answering the human's yes/no questions" in ai_prompt[0]["text"]

        # Test human guesser mode
        game.is_ai_guesser = False
        human_prompt = game.get_system_prompt()
        assert isinstance(human_prompt, list)
        assert "trying to guess what the human is thinking" in human_prompt[0]["text"]

    def test_message_formatting(self, game):
        """Test conversation message formatting"""
        # Add test conversation history
        game.conversation_history = [
            {
                "question": "Is it alive?",
                "answer": "Yes",
                "turn": 1,
                "timestamp": datetime.now().isoformat()
            }
        ]
        messages = game.format_messages("Is it an animal?")
        
        assert len(messages) == 3  # History message + new question + initial guess
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"][0]["text"] == "Is it an animal?"

    @patch('builtins.input')
    def test_valid_input_handling(self, mock_input, game):
        """Test input validation"""
        # Test valid input
        mock_input.return_value = "yes"
        result = game.get_valid_input("Test prompt", ["yes", "no"])
        assert result == "yes"

        # Test invalid input followed by valid input
        mock_input.side_effect = ["invalid", "yes"]
        result = game.get_valid_input("Test prompt", ["yes", "no"])
        assert result == "yes"

        # Test EOF error
        mock_input.side_effect = EOFError()
        with pytest.raises(InputError):
            game.get_valid_input("Test prompt")

    def test_llm_interaction(self, game, mock_bedrock):
        """Test LLM API interaction"""
        # Test successful API call
        mock_response = {
            "output": {
                "message": {
                    "content": [{"text": "Test response"}]
                }
            }
        }
        mock_bedrock.converse.return_value = mock_response
        response = game.call_llm("Test prompt")
        assert response == "Test response"
        assert len(game.conversation_history) == 1

        # Test API error
        mock_bedrock.converse.side_effect = Exception("API Error")
        with pytest.raises(AIError):
            game.call_llm("Test prompt")

    def test_game_modes(self, game):
        """Test AI guesser and thinker modes"""
        # Test AI guesser mode
        game.is_ai_guesser = True
        game.call_llm = Mock(side_effect=["Is it alive?", "continue"])
        with patch('builtins.input', return_value="yes"):
            assert game.play_as_ai_guesser() is True
            assert game.questions_asked == 1

        # Test AI thinker mode
        game.is_ai_guesser = False
        game.questions_asked = 0
        game.call_llm = Mock(return_value="Yes")
        with patch('builtins.input', return_value="Is it red?"):
            assert game.play_as_ai_thinker() is True
            assert game.questions_asked == 1

    def test_game_ending_conditions(self, game):
        """Test various game-ending scenarios"""
        # Test correct guess
        game.call_llm = Mock(return_value="Is it a cat?")
        with patch('builtins.input', side_effect=["yes"]):
            assert game.handle_ai_final_guess() is False

        # Test incorrect guess
        with patch('builtins.input', side_effect=["no"]):
            assert game.handle_ai_final_guess() is False

        # Test max questions reached
        game.questions_asked = game.config.max_questions
        game.call_llm = Mock(return_value="continue")
        with patch('builtins.input', side_effect=["some input"]):  # Mock to avoid stdin read issues
            assert game.play_as_ai_thinker() is False


    def test_game_statistics(self, game, tmp_path):
        """Test game statistics saving"""
        game.config.log_dir = str(tmp_path)
        game.secret_subject = "cat"
        game.is_ai_guesser = True
        game.conversation_history = [{
            "question": "Is it alive?",
            "answer": "Yes",
            "turn": 1,
            "timestamp": datetime.now().isoformat()
        }]
        
        game.save_game_stats(ai_won=True)
        stats_file = tmp_path / "game_stats.jsonl"
        
        assert stats_file.exists()
        with open(stats_file) as f:
            stats = json.loads(f.readline())
            assert stats["ai_won"] is True
            assert stats["secret_subject"] == "cat"
            assert len(stats["conversation_history"]) == 1

    def test_category_selection(self, game):
        """Test random category selection"""
        category = game.pick_type()
        valid_categories = {
            "historical figure", "celebrity", "fictional character", "athlete", "musician",
            "country", "city", "landmark", "natural wonder", "building",
            "household item", "food", "technology", "animal", "vehicle"
        }
        assert category in valid_categories

@pytest.mark.integration
class TestGameIntegration:
    """Integration tests for the game"""

    @pytest.fixture
    def integrated_game(self, tmp_path):
        config = GameConfig(log_dir=str(tmp_path))
        return TwentyQuestions(config)

    



    def test_error_handling(self, integrated_game):
        """Test game error handling"""
        with patch('builtins.input', side_effect=["1", "cat"]):
            integrated_game.call_llm = Mock(side_effect=GameError("Test error"))
            
            with pytest.raises(SystemExit) as exit_info:
                integrated_game.run()
            assert exit_info.value.code == 1

    def test_interruption_handling(self, integrated_game):
        """Test game interruption handling"""
        with patch('builtins.input', side_effect=["1", "cat", KeyboardInterrupt]):
            with patch.object(integrated_game, 'save_game_stats') as mock_save:
                with pytest.raises(SystemExit) as exit_info:
                    integrated_game.run()
                
                assert exit_info.value.code == 0
                mock_save.assert_called_once_with(ai_won=False)

if __name__ == '__main__':
    pytest.main(['-v'])