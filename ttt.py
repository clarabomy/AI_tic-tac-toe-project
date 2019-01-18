# ----------------------------------------
#            Import librairies
# ----------------------------------------
import numpy as np
import os
import os.path
import copy
import keras
from keras.models import Model, load_model
from keras.layers import Flatten, Dense, Dropout


# ----------------------------------------
#            Global variables
# ----------------------------------------
model = None
callbacks = []
boardgames = []
whowon = []
current_game_boards = []
won  = lost = draw = 0
trained_model = "model_ttt.h5"


# ----------------------------------------
#           General functions
# ----------------------------------------
def make_model():
    """
    Creates a neural network
    """
    global model, callbacks
    if model != None:
        return
    
    inputs = keras.layers.Input(shape=(2,3,3))
    output = Flatten()(inputs)
    output = Dense(512, kernel_initializer='glorot_uniform', activation='relu')(output)
    output = Dense(512, kernel_initializer='glorot_uniform', activation='relu')(output)
    output = Dense(512, kernel_initializer='glorot_uniform', activation='relu')(output)
    output = Dense(512, kernel_initializer='glorot_uniform', activation='relu')(output)
    output = Dense(1, kernel_initializer='glorot_uniform', activation='sigmoid')(output)
    model = Model(inputs=inputs, outputs=output)
    model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=0.001))

def train():
    """
    Trains a neural network
    """
    global model, boardgames, whowon
    model.fit(np.array(boardgames), np.array(whowon), epochs=1, verbose=1)


def find_next_best_move(board, player):
    """
    Determines the best move to win
    """
    global model
    best_prob_to_win = -1
    if player == 1:
        best_prob_to_win = 2
    best_x = best_y = 0

    for x in range(3):
        for y in range(3):
            if not board[0, x, y] and not board[1, x, y]:
                # Nobody has played in this position.
                # Let's play and see how good the board looks for us
                board[player, x, y] = 1
                prob_to_win = model.predict(np.array([board]), batch_size=1, verbose=0)[0]
                board[player, x, y] = 0
                if ((player == 0 and prob_to_win > best_prob_to_win) or
                    (player == 1 and prob_to_win < best_prob_to_win)):
                    best_x = x
                    best_y = y
                    best_prob_to_win = prob_to_win
    #print("Best move is", best_x, best_y, "with probability to win: ", prob_to_win)
    return best_x, best_y

def remember_game_board(board):
    """
    To remember a game
    """
    global current_game_boards
    current_game_boards.append(np.array(board).tolist()) 

def notify_new_game(whowon_, training):
    """
    Notifies a new game
    """

    # whowon_  should be 1 if computer, 0 if person, 0.5 if tie
    global boardgames, whowon, current_game_boards
    
    boardgames = current_game_boards
    whowon = (np.ones(len(current_game_boards)) * whowon_).tolist()
    current_game_boards = []

    # if the IA trains after playing the game
    if training == True:
        train()

def get_valid_moves(board):
    """
    Gives the possible positions to play
    """
    valid_moves = []
    for x in range(3):
        for y in range(3):
            if not board[0, x, y] and not board[1, x, y]:
                valid_moves.append((x,y))
    return valid_moves

def get_random_move(board):
    valid_moves = get_valid_moves(board)
    return valid_moves[np.random.randint(len(valid_moves))]

def has_won(board, player):
    """
    Tests if the player has won
    """
    p = player
    if ((board[p,0,0] and board[p,1,1] and board[p,2,2]) or
        (board[p,2,0] and board[p,1,1] and board[p,0,2])):
        return True
    for x in range(3):
        if ((board[p,x,0] and board[p,x,1] and board[p,x,2]) or
            (board[p,0,x] and board[p,1,x] and board[p,2,x])):
            return True
    return False

def is_board_full(board):
    """
    Checks if the game board is filled or if there are still empty boxes
    """
    for x in range(3):
        for y in range(3):
            if not board[0, x, y] and not board[1, x, y]:
                return False
    return True

def play_game():
    if is_board_full(board):
        notify_new_game()


def print_board(board):
    """
    Prints the game board
    """
    matrix = []
    for x in range(3):
        for y in range(3):
            if board[0,x,y]:
                matrix.append("X")
            elif board[1,x,y]:
                matrix.append("O")
            else:
                matrix.append(".")

    os.system('cls')
    print("\n\n ========* TIC TAC TOE *========\n")
    print("            1   2   3")
    print("         * * * * * * * *")
    print("       1 *  {} | {} | {}  *".format(matrix[0],matrix[1],matrix[2]))
    print("         * ---+---+--- *")
    print("       2 *  {} | {} | {}  *".format(matrix[3],matrix[4],matrix[5]))
    print("         * ---+---+--- *")
    print("       3 *  {} | {} | {}  *".format(matrix[6],matrix[7],matrix[8]))
    print("         * * * * * * * *\n\n")

# ----------------------------------------
#     Functions of PLAY AGAINST SELF
# ----------------------------------------
def play_against_self_randomly(nb_matchs):
    """ 
    Plays n games (IA alone) and determines the score
    """
    for i in range(nb_matchs):
        global won, lost, draw, model
        player_who_won, board = play_against_self_randomly_()
        notify_new_game(player_who_won, True)
        #print_board(board)
        if player_who_won == 0 :
            lost+=1
        if player_who_won == 1 :
            won+=1
        if player_who_won == 0.5:
            draw+=1
        print("Résultats de l'IA - Parties gagnées: {}     Parties perdues: {}     Matchs nuls: {}".format(won, lost, draw))
        print()


def play_against_self_randomly_():
    """
    Plays one game (IA alone)
    """
    board = np.zeros((2, 3, 3))
    player = 0
    
   # Return 0.5 if tie, 1 if computer player won, 0 if we lost
    while True:
        if has_won(board, 0):
            return 1, board
        if has_won(board, 1):
            return 0, board
        if is_board_full(board):
            return 0.5, board
        
        # Determine the coordinates to play
        if np.random.randint(2) == 0:
            x,y = get_random_move(board)
        else:
            x, y = find_next_best_move(board, player)
       
        board[player, x, y] = 1 
        remember_game_board(board)

        # Player switch
        player = 1 if player == 0 else 0

        #print_board(board)

# ----------------------------------------
#     Functions of PLAY WITH PLAYER
# ----------------------------------------
def play_against_player():
    """ 
    Plays n games and determines the score
    """
    endGame = keyboard = None
    while endGame != "N":
        board = np.zeros((2, 3, 3))

        # Determines the first player to start
        while keyboard != "Y" and keyboard != "N":
            keyboard = str(input("\tSouhaitez vous commencer ? (Y/N) ")).upper()
            endGame = None
        computer = 1 if keyboard == "Y" else 0
        print_board(board)

        global won, lost, draw
        result, board = play_against_player_(computer)
        notify_new_game(result, False)
        print_board(board)
        
        if (result == 0 and computer == 0) or (result == 1 and computer == 1):
            lost+=1
            print(" Bravo, vous avez gagné !")
        elif (result == 0 and computer == 1) or (result == 1 and computer == 0) :
            won+=1
            print(" Vous avez perdu !")
        else:
            draw+=1
            print(" Match nul !")
        print(" Vos résultats - Parties gagnées: {}     Parties perdues: {}     Matchs nuls: {}".format(lost, won, draw))

        while endGame != "Y" and endGame != "N":
            keyboard = None
            endGame = str(input("\n\tSouhaitez vous rejouer ? (Y/N) ")).upper()
           

def play_against_player_(computer):
    """
    Plays one game (IA against player)
    """
    board = np.zeros((2, 3, 3))
    player = 0
   
    # Return 0.5 if tie, 1 if computer player won, 0 if we lost
    while True:
        print_board(board)
        if has_won(board, 0):
            return 1, board
        if has_won(board, 1):
            return 0, board
        if is_board_full(board):
            return 0.5, board

        # Determine the coordinates to play depending on the player
        x,y = find_next_best_move(board, player) if player == computer else next_player_move(board)


        board[player, x, y] = 1
        remember_game_board(board)
        
        # Player switch
        player = 1 if player == 0 else 0


def next_player_move(board):
    """
    Asks the player the positions to play
    """
    x = y = -1
    value = get_valid_moves(board)
    print(" A vous de jouer...")
    # While values are not valid, we ask the player
    while (x,y) not in value:
        x = input("\tNuméro de ligne : ")
        y = input("\tNuméro de colonne : ")
        if x.isdigit() and y.isdigit():
            x = int(x)-1
            y = int(y)-1
        else:
            print("\n Valeurs incorrectes. Veuillez recommencer : ")
            x = y = -1
    return x, y

# ----------------------------------------
#               Main entry
# ----------------------------------------
if __name__ == "__main__":
    # If the file of the model exists, we load the model
    if (os.path.isfile(trained_model)):
        model = load_model(trained_model)
    
    # Else, we create a new model
    else :
        make_model()

    play_against_self_randomly(15000) #Training received by the model
    play_against_player()
    
    model.save(trained_model)