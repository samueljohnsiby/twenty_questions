# AI-Powered 20 Questions Game 🎮

An interactive implementation of the classic 20 Questions game using AWS Bedrock and Meta's Llama model. Play against an AI that can either guess what you're thinking or think of something for you to guess!

## Features ✨

- Two game modes:
  - AI Guesser: You think of something, AI tries to guess it
  - AI Thinker: AI thinks of something, you try to guess it
- Rich console interface with color-coded outputs
- Comprehensive logging system
- Game statistics tracking and analysis
- Error handling and retry mechanisms
- Configurable game parameters
- Post-game analysis and insights

## Prerequisites 📋

- Python 3.8+
- AWS account with Bedrock access
- AWS credentials configured locally
- Required Python packages (see Installation)

## Installation 🚀

1. Clone the repository:
```bash
git clone <repository-url>
cd twenty-questions
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Configure AWS credentials:
```bash
aws configure
```

## Configuration ⚙️

The game can be configured using environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| GAME_MAX_QUESTIONS | Maximum questions allowed | 20 |
| GAME_MAX_RETRIES | Max API retry attempts | 3 |
| GAME_TIMEOUT | API timeout in seconds | 30 |
| GAME_MODEL_ID | Bedrock model ID | us.meta.llama3-2-90b-instruct-v1:0 |
| GAME_TEMPERATURE | Model temperature | 0.7 |
| GAME_TOP_P | Top P sampling parameter | 0.9 |
| GAME_TOP_K | Top K sampling parameter | 200 |
| GAME_MAX_TOKENS | Maximum output tokens | 256 |
| GAME_LOG_DIR | Directory for logs | logs |

## Usage 🎯

1. Start the game:
```bash
python twenty_questions.py
```

2. Choose your role:
   - Thinker (1): You think of something, AI guesses
   - Guesser (2): AI thinks of something, you guess

3. Follow the prompts to play the game:
   - If you're the thinker, enter your secret subject when prompted
   - If you're the guesser, ask yes/no questions or make guesses
   - The game continues until either:
     - The correct answer is guessed
     - 20 questions are used
     - The player gives up

4. After the game, you can view detailed game statistics and analysis

## Game Analysis 📊

The game includes a comprehensive analysis system that tracks:
- Win/loss statistics
- Average questions per game
- Time taken per question
- Most common question types
- Success rates by category
- Historical performance trends

To view game analysis:
- Choose "Yes" when prompted after the game ends
- Or check the generated logs in the `logs` directory

## Logging 📝

The game maintains detailed logs in the specified log directory:
- Game events and interactions
- Error messages and warnings
- Game statistics in JSON format
- Performance metrics

## Error Handling 🛠️

The game includes robust error handling for:
- Network connectivity issues
- API timeouts
- Invalid user input
- AWS service errors


## Technologies Used 👏

- AWS Bedrock for AI capabilities
- Meta's Llama model for natural language processing
- Rich library for beautiful console output
- Amazon Q for development,writing test and this README.md file you are reading 

