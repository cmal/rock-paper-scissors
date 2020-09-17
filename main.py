from RPS_game import play, mrugesh, abbey, quincy, kris, human, random_player
from RPS import player
from unittest import main

games = 1000
play(player, quincy, games)
play(player, abbey, games)
play(player, kris, games)
play(player, mrugesh, games)

# play(player, human, games)

# play(human, abbey, 20, verbose=True)
# play(human, random_player, 1000)
# main(module='test_module', exit=False)
