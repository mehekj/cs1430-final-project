import unittest

from game import *
import numpy as np

class TestGameLogic(unittest.TestCase):

    def test_game1(self):
        start_board = Board()
                                    
        start_board.calculate_difference(np.array([[2, 2, 2, 2, 2, 2, 2, 2],
                                      [2, 2, 2, 2, 2, 2, 2, 2],
                                      [0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 1, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0],
                                      [1, 1, 1, 1, 0, 1, 1, 1],
                                      [1, 1, 1, 1, 1, 1, 1, 1]]))

        self.assertAlmostEqual(start_board.turn, Turn.BLACK)
        self.assertAlmostEqual(start_board.board[4, 4], Piece.WP)

        start_board.calculate_difference(np.array([[2, 2, 2, 2, 2, 2, 2, 2],
                                      [2, 2, 0, 2, 2, 2, 2, 2],
                                      [0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 2, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 1, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0],
                                      [1, 1, 1, 1, 0, 1, 1, 1],
                                      [1, 1, 1, 1, 1, 1, 1, 1]]))

        self.assertAlmostEqual(start_board.turn, Turn.WHITE) 
        self.assertAlmostEqual(start_board.board[1, 2], Piece.E)
        self.assertAlmostEqual(start_board.board[3, 2], Piece.BP)                             

        start_board.calculate_difference(np.array([[2, 2, 2, 2, 2, 2, 2, 2],
                                      [2, 2, 0, 2, 2, 2, 2, 2],
                                      [0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 2, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 1, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 1, 0, 0],
                                      [1, 1, 1, 1, 0, 0, 1, 1],
                                      [1, 1, 1, 1, 1, 1, 1, 1]]))

        self.assertAlmostEqual(start_board.turn, Turn.BLACK)
        self.assertAlmostEqual(start_board.board[6, 5], Piece.E)
        self.assertAlmostEqual(start_board.board[5, 5], Piece.WP)                              

        start_board.calculate_difference(np.array([[2, 2, 2, 2, 2, 2, 2, 2],
                                      [0, 2, 0, 2, 2, 2, 2, 2],
                                      [0, 0, 0, 0, 0, 0, 0, 0],
                                      [2, 0, 2, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 1, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 1, 0, 0],
                                      [1, 1, 1, 1, 0, 0, 1, 1],
                                      [1, 1, 1, 1, 1, 1, 1, 1]]))

        self.assertAlmostEqual(start_board.turn, Turn.WHITE)  
        self.assertAlmostEqual(start_board.board[3, 0], Piece.BP)                             

        start_board.calculate_difference(np.array([[2, 2, 2, 2, 2, 2, 2, 2],
                                      [0, 2, 0, 2, 2, 2, 2, 2],
                                      [0, 0, 0, 0, 0, 0, 0, 0],
                                      [2, 0, 2, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 1, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 1, 0, 0],
                                      [1, 1, 1, 1, 1, 0, 1, 1],
                                      [1, 1, 1, 1, 1, 1, 0, 1]]))

        self.assertAlmostEqual(start_board.turn, Turn.BLACK)
        self.assertAlmostEqual(start_board.board[6, 4], Piece.WN)
        self.assertAlmostEqual(start_board.board[7, 6], Piece.E)                              

        start_board.calculate_difference(np.array([[2, 2, 2, 2, 2, 2, 2, 2],
                                      [0, 2, 0, 0, 2, 2, 2, 2],
                                      [0, 0, 0, 0, 0, 0, 0, 0],
                                      [2, 0, 2, 2, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 1, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 1, 0, 0],
                                      [1, 1, 1, 1, 1, 0, 1, 1],
                                      [1, 1, 1, 1, 1, 1, 0, 1]]))

        self.assertAlmostEqual(start_board.turn, Turn.WHITE)
        self.assertAlmostEqual(start_board.board[1, 3], Piece.E)
        self.assertAlmostEqual(start_board.board[3, 3], Piece.BP)                              

        #capture
        start_board.calculate_difference(np.array([[2, 2, 2, 2, 2, 2, 2, 2],
                                      [0, 2, 0, 0, 2, 2, 2, 2],
                                      [0, 0, 0, 0, 0, 0, 0, 0],
                                      [2, 0, 2, 1, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 1, 0, 0],
                                      [1, 1, 1, 1, 1, 0, 1, 1],
                                      [1, 1, 1, 1, 1, 1, 0, 1]]))

        self.assertAlmostEqual(start_board.turn, Turn.BLACK)
        self.assertAlmostEqual(start_board.board[4, 4], Piece.E)
        self.assertAlmostEqual(start_board.board[3, 3], Piece.WP)                              

        #capture
        start_board.calculate_difference(np.array([[2, 2, 2, 0, 2, 2, 2, 2],
                                      [0, 2, 0, 0, 2, 2, 2, 2],
                                      [0, 0, 0, 0, 0, 0, 0, 0],
                                      [2, 0, 2, 2, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 1, 0, 0],
                                      [1, 1, 1, 1, 1, 0, 1, 1],
                                      [1, 1, 1, 1, 1, 1, 0, 1]]))

        self.assertAlmostEqual(start_board.turn, Turn.WHITE)
        self.assertAlmostEqual(start_board.board[0, 3], Piece.E)
        self.assertAlmostEqual(start_board.board[3, 3], Piece.BQ)



    def test_update_pieces(self):
        start_board = Board()

        start_board.update_pieces(3, 4, 1, 4)
        self.assertAlmostEqual(start_board.board[3, 4], Piece.BP)
        self.assertAlmostEqual(start_board.board[1, 4], Piece.E)

        start_board.update_pieces(5, 7, 7, 6)
        self.assertAlmostEqual(start_board.board[5, 7], Piece.WN)
        self.assertAlmostEqual(start_board.board[7, 6], Piece.E)

if __name__ == '__main__':
    unittest.main()