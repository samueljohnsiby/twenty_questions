def read_game_flow(file_path):
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
        print(f"Error: The file at {file_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    
    return game_flow

# Example Usage:
game_flow = read_game_flow(r'/home/sam/Documents/knocknock/ai_guesser_example.txt')

# Display the game flow
# for speaker, statement in game_flow:
#     print(f"{speaker}: {statement}")

print(game_flow)
